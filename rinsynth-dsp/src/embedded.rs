//! Lean embedded engine — a fixed, concrete voice pipeline for
//! resource-constrained targets (ESP32-S3 Xtensa, no pipelined FPU).
//!
//! `VoiceManager` is a desktop-grade engine: `dyn` dispatch everywhere, a
//! general mod matrix, 4 envelopes and 3 LFOs per voice, stereo SVFs. On
//! an in-order core that costs ~1.5-2k cycles/voice-sample — ~3 voices
//! per 44.1 kHz block. This engine renders the *same* `rinsynth-model`
//! presets through a fixed pipeline instead:
//!
//! - 2 oscillators per voice — PolyBLEP saw/pulse (the desktop oscs are
//!   naive wraps and alias badly; BLEP costs ~12 flops at the edge)
//! - amp ADSR + filter ADSR (gated on env amount)
//! - stereo-mixed oscillators through a shared-coefficient RBJ biquad
//!   (per-channel states; LP/HP/BP/notch)
//! - one LFO per voice at control rate, legacy `lfo_*` destination
//!   (pitch / cutoff / amplitude)
//! - portamento, pitch bend, per-osc equal-power pan resolved once per
//!   note — not per sample
//!
//! Not implemented (deliberately): the mod matrix, osc3/osc4, second
//! serial filter, env3/env4, lfo2/lfo3, FX. Presets that use them render
//! as their core voice.
//!
//! Target: ~700 cycles/voice-sample → 8 voices inside a 5.8 ms block.

use super::envelope::{ADSREnvelope, Envelope, EnvelopeState};
use super::filter::FilterMode;
use super::oscillator::sine_lut;
use rinsynth_model::event::{NoteEvent, SynthEventKind, WaveformType};
use rinsynth_model::lfo::{LfoDestination, LFO};
use rinsynth_model::presets::Preset;
use rinsynth_model::types::{
    midi_to_frequency, Amplitude, Frequency, MidiNote, Sample, SampleRate, StereoSample,
};
use rinsynth_model::voice::PolyphonyMode;
use std::f32::consts::PI;

/// Samples between control-rate updates (LFO tick, pitch, filter coeffs).
const CONTROL_DIV: u32 = 64;
const MAX_OSCS: usize = 2;

/// PolyBLEP residual for a discontinuity at phase 0 (normalized t ∈ [0,1)).
/// For pulse waves call again with the second edge's phase.
#[inline(always)]
fn poly_blep(t: f32, dt: f32) -> f32 {
    if t < dt {
        let x = t / dt;
        x + x - x * x - 1.0
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

#[inline(always)]
fn flush_denormal(x: &mut f32) {
    if x.abs() < 1e-18 {
        *x = 0.0;
    }
}

/// Fixed-pipeline oscillator: one struct, waveform chosen by a match —
/// statically dispatched and inlinable, unlike `Box<dyn Oscillator>`.
pub struct LeanOsc {
    kind: WaveformType,
    phase: f32, // 0..1
    inc: f32,
    duty: f32,
    rng: u32,
}

impl LeanOsc {
    fn new(kind: WaveformType) -> Self {
        Self {
            kind,
            phase: 0.0,
            inc: 0.0,
            duty: 0.5,
            rng: 0x9E37_79B9,
        }
    }

    #[inline]
    fn set_freq(&mut self, freq: Frequency, sample_rate: SampleRate) {
        // PolyBLEP needs dt < 0.5 — clamp rather than alias worse.
        self.inc = (freq / sample_rate as f32).clamp(0.0, 0.49);
    }

    #[inline]
    fn set_duty(&mut self, duty: f32) {
        self.duty = duty.clamp(0.01, 0.99);
    }

    #[inline]
    fn reset(&mut self) {
        self.phase = 0.0;
    }

    #[inline(always)]
    #[link_section = ".iram1.text"]
    fn next(&mut self) -> Sample {
        let t = self.phase;
        self.phase += self.inc;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        match self.kind {
            WaveformType::Sine => sine_lut(t * 2.0 * PI),
            WaveformType::Saw => {
                let v = 2.0 * t - 1.0;
                v - poly_blep(t, self.inc)
            }
            WaveformType::Square | WaveformType::Pulse => {
                let v = if t < self.duty { 1.0 } else { -1.0 };
                // second edge at t = duty
                let t2 = if t >= self.duty {
                    t - self.duty
                } else {
                    t + (1.0 - self.duty)
                };
                v + poly_blep(t, self.inc) - poly_blep(t2, self.inc)
            }
            WaveformType::Triangle => {
                // harmonics fall as 1/n² — no BLEP needed
                if t < 0.5 {
                    4.0 * t - 1.0
                } else {
                    3.0 - 4.0 * t
                }
            }
            WaveformType::Noise => {
                // xorshift32 → [-1, 1) via the mantissa trick — no division
                self.rng ^= self.rng << 13;
                self.rng ^= self.rng >> 17;
                self.rng ^= self.rng << 5;
                (f32::from_bits(0x3F80_0000 | (self.rng >> 9)) - 1.5) * 2.0
            }
        }
    }
}

/// RBJ cookbook biquad — LP/HP/BP/notch, stereo states sharing one set of
/// coefficients. ~9 flops/channel vs the saturated SVF's ~50.
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: [f32; 2],
    x2: [f32; 2],
    y1: [f32; 2],
    y2: [f32; 2],
}

impl Biquad {
    fn new() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: [0.0; 2],
            x2: [0.0; 2],
            y1: [0.0; 2],
            y2: [0.0; 2],
        }
    }

    /// Recompute coefficients — control rate only (sin/cos + a division).
    fn set(&mut self, mode: FilterMode, cutoff: Frequency, q: f32, fs: SampleRate) {
        let fc = cutoff.clamp(20.0, fs as f32 * 0.45);
        let w = 2.0 * PI * fc / fs as f32;
        let (sw, cw) = w.sin_cos();
        let alpha = sw / (2.0 * q.max(0.01));
        let a0 = 1.0 + alpha;
        let (b0, b1, b2) = match mode {
            FilterMode::LowPass => {
                let t = (1.0 - cw) * 0.5;
                (t, 2.0 * t, t)
            }
            FilterMode::HighPass => {
                let t = (1.0 + cw) * 0.5;
                (t, -2.0 * t, t)
            }
            FilterMode::BandPass => (alpha, 0.0, -alpha),
            FilterMode::Notch => (1.0, -2.0 * cw, 1.0),
        };
        let inv = 1.0 / a0;
        self.b0 = b0 * inv;
        self.b1 = b1 * inv;
        self.b2 = b2 * inv;
        self.a1 = -2.0 * cw * inv;
        self.a2 = (1.0 - alpha) * inv;
    }

    #[inline(always)]
    #[link_section = ".iram1.text"]
    fn process(&mut self, ch: usize, x: Sample) -> Sample {
        let mut y = self.b0 * x + self.b1 * self.x1[ch] + self.b2 * self.x2[ch]
            - self.a1 * self.y1[ch]
            - self.a2 * self.y2[ch];
        flush_denormal(&mut y);
        self.x2[ch] = self.x1[ch];
        self.x1[ch] = x;
        self.y2[ch] = self.y1[ch];
        self.y1[ch] = y;
        y
    }
}

/// Equal-power pan → constant (left, right) gains. Resolved once per note
/// or preset change — never per sample.
#[inline]
fn pan_gains(pan: f32) -> (f32, f32) {
    let angle = (pan.clamp(-1.0, 1.0) + 1.0) * 0.25 * PI;
    (angle.cos(), angle.sin())
}

pub struct LeanVoice {
    oscs: [LeanOsc; MAX_OSCS],
    /// (level × panL, level × panR) per osc — level folded into pan so the
    /// per-sample path is just two muls.
    gains: [(f32, f32); MAX_OSCS],
    osc_level: [f32; MAX_OSCS],
    detune: [f32; MAX_OSCS],
    amp_env: ADSREnvelope,
    fenv: ADSREnvelope,
    fenv_amount: f32,
    filter: Biquad,
    filter_mode: FilterMode,
    res_q: f32,
    base_cutoff: Frequency,
    lfo: LFO,
    lfo_dest: LfoDestination,
    base_freq: Frequency,
    glide_step: f32,
    glide_target: Frequency,
    gliding: bool,
    velocity: Amplitude,
    note: MidiNote,
    active: bool,
    age: u64,
    control_phase: u32,
    amp_lfo: f32,
    fenv_amp: f32,
    sample_rate: SampleRate,
}

impl LeanVoice {
    fn new(sample_rate: SampleRate) -> Self {
        Self {
            oscs: [
                LeanOsc::new(WaveformType::Saw),
                LeanOsc::new(WaveformType::Saw),
            ],
            gains: [(0.0, 0.0); MAX_OSCS],
            osc_level: [0.0; MAX_OSCS],
            detune: [1.0; MAX_OSCS],
            amp_env: ADSREnvelope::new(0.01, 0.1, 0.7, 0.2, sample_rate),
            fenv: ADSREnvelope::new(0.01, 0.3, 0.0, 0.3, sample_rate),
            fenv_amount: 0.0,
            filter: Biquad::new(),
            filter_mode: FilterMode::LowPass,
            res_q: 0.707,
            base_cutoff: 8000.0,
            lfo: LFO::new((sample_rate / CONTROL_DIV).max(1)),
            lfo_dest: LfoDestination::Off,
            base_freq: 440.0,
            glide_step: 0.0,
            glide_target: 440.0,
            gliding: false,
            velocity: 0.0,
            note: 0,
            active: false,
            age: 0,
            control_phase: 0,
            amp_lfo: 1.0,
            fenv_amp: 0.0,
            sample_rate,
        }
    }

    fn note_on(&mut self, note: MidiNote, velocity: Amplitude, portamento: f32, age: u64) {
        let freq = midi_to_frequency(note);
        self.note = note;
        self.velocity = velocity;
        self.active = true;
        self.age = age;
        if portamento > 0.005 && self.amp_env.state() != EnvelopeState::Idle {
            // legato glide — start from wherever the last note left us
            self.glide_target = freq;
            self.glide_step = (freq - self.base_freq) / (portamento * self.sample_rate as f32);
            self.gliding = self.glide_step != 0.0;
        } else {
            self.base_freq = freq;
            self.gliding = false;
        }
        self.amp_env.reset();
        self.amp_env.trigger();
        self.fenv.reset();
        self.fenv.trigger();
        self.lfo.reset();
        for o in &mut self.oscs {
            o.reset();
        }
        self.set_osc_freqs(0.0);
        self.control_phase = 0;
    }

    fn note_off(&mut self, note: MidiNote) {
        if self.active && self.note == note {
            self.amp_env.release();
            self.fenv.release();
        }
    }

    #[inline]
    fn set_osc_freqs(&mut self, semitone_offset: f32) {
        let bend = 2.0f32.powf(semitone_offset / 12.0);
        for i in 0..MAX_OSCS {
            self.oscs[i].set_freq(
                self.base_freq * self.detune[i] * bend,
                self.sample_rate,
            );
        }
    }

    /// One control tick: LFO → resolved pitch/cutoff/amp targets.
    #[inline]
    fn control_tick(&mut self, bend_semitones: f32) {
        let lfo = self.lfo.next_value(); // already depth-scaled, -depth..depth
        let lfo_pitch = if self.lfo_dest == LfoDestination::Pitch {
            lfo * 2.0 // ±2 semitones at full depth
        } else {
            0.0
        };
        self.set_osc_freqs(bend_semitones + lfo_pitch);

        let lfo_cutoff = if self.lfo_dest == LfoDestination::FilterCutoff {
            lfo * 2.0 // ±2 octaves at full depth
        } else {
            0.0
        };
        // Filter env: 6-octave sweep at full amount (matches the rich engine)
        let env_sweep = 6.0 * self.fenv_amount * self.fenv_amp;
        let cutoff = (self.base_cutoff * 2.0f32.powf(env_sweep + lfo_cutoff))
            .clamp(20.0, self.sample_rate as f32 * 0.45);
        self.filter
            .set(self.filter_mode, cutoff, self.res_q, self.sample_rate);

        self.amp_lfo = if self.lfo_dest == LfoDestination::Amplitude {
            1.0 + lfo * 0.5
        } else {
            1.0
        };
    }

    #[link_section = ".iram1.text"]
    fn next_stereo(&mut self, bend_semitones: f32) -> StereoSample {
        if !self.active {
            return StereoSample::ZERO;
        }
        let amp = self.amp_env.next_amplitude();
        if self.amp_env.is_finished() {
            self.active = false;
            return StereoSample::ZERO;
        }
        self.fenv_amp = if self.fenv_amount != 0.0 {
            self.fenv.next_amplitude()
        } else {
            0.0
        };

        if self.gliding {
            self.base_freq += self.glide_step;
            if (self.glide_target - self.base_freq).abs() <= self.glide_step.abs() {
                self.base_freq = self.glide_target;
                self.gliding = false;
            }
        }

        if self.control_phase == 0 {
            self.control_tick(bend_semitones);
        }
        self.control_phase = (self.control_phase + 1) % CONTROL_DIV;

        let mut l = 0.0f32;
        let mut r = 0.0f32;
        for i in 0..MAX_OSCS {
            if self.osc_level[i] > 0.0 {
                let s = self.oscs[i].next();
                l += s * self.gains[i].0;
                r += s * self.gains[i].1;
            }
        }

        let fl = self.filter.process(0, l);
        let fr = self.filter.process(1, r);
        let g = amp * self.velocity * self.amp_lfo;
        StereoSample::new(fl * g, fr * g)
    }
}

pub struct EmbeddedEngine {
    voices: Vec<LeanVoice>,
    sample_rate: SampleRate,
    master_volume: f32,
    pitch_bend: f32,      // -1..1
    bend_range: f32,      // semitones
    portamento: f32,
    polyphony: PolyphonyMode,
    age_counter: u64,
    stereo_width: f32,
}

impl EmbeddedEngine {
    pub fn new(max_voices: usize, sample_rate: SampleRate) -> Self {
        Self {
            voices: (0..max_voices).map(|_| LeanVoice::new(sample_rate)).collect(),
            sample_rate,
            master_volume: 1.0,
            pitch_bend: 0.0,
            bend_range: 12.0,
            portamento: 0.0,
            polyphony: PolyphonyMode::Poly,
            age_counter: 0,
            stereo_width: 1.0,
        }
    }

    pub fn set_master_volume(&mut self, v: Amplitude) {
        self.master_volume = v.clamp(0.0, 1.0);
    }

    pub fn active_voice_count(&self) -> usize {
        self.voices.iter().filter(|v| v.active).count()
    }

    pub fn apply_preset(&mut self, preset: &Preset) {
        self.portamento = preset.portamento_time;
        self.polyphony = preset.polyphony_mode;
        self.bend_range = preset.pitch_bend_range as f32;
        self.stereo_width = preset.stereo_width;
        self.master_volume = preset.master_volume.clamp(0.0, 1.0);
        let res_q = 0.7 * 20.0f32.powf(preset.filter_resonance.clamp(0.0, 1.0));
        for v in &mut self.voices {
            v.oscs[0].kind = preset.osc1_waveform;
            v.oscs[0].set_duty(preset.osc1_pw);
            v.oscs[1].kind = preset.osc2_waveform;
            v.oscs[1].set_duty(preset.osc2_pw);
            v.osc_level = [preset.osc1_level, preset.osc2_level];
            v.detune = [
                1.0,
                super::oscillator::detune_ratio(preset.osc2_semitones, preset.osc2_cents),
            ];
            let (p0l, p0r) = pan_gains(preset.osc1_pan);
            let (p1l, p1r) = pan_gains(preset.osc2_pan);
            v.gains = [
                (preset.osc1_level * p0l, preset.osc1_level * p0r),
                (preset.osc2_level * p1l, preset.osc2_level * p1r),
            ];
            v.amp_env = ADSREnvelope::new(
                preset.amp_attack,
                preset.amp_decay,
                preset.amp_sustain,
                preset.amp_release,
                v.sample_rate,
            );
            v.fenv = ADSREnvelope::new(
                preset.filter_attack,
                preset.filter_decay,
                preset.filter_sustain,
                preset.filter_release,
                v.sample_rate,
            );
            v.fenv_amount = preset.filter_env_amount;
            v.base_cutoff = preset.filter_cutoff;
            v.res_q = res_q;
            v.lfo.set_rate(preset.lfo_rate);
            v.lfo.set_depth(preset.lfo_depth);
            v.lfo.set_waveform(preset.lfo_waveform);
            v.lfo_dest = preset.lfo_destination;
        }
    }

    pub fn receive_event(&mut self, event: NoteEvent) {
        match event.kind {
            SynthEventKind::NoteOn => {
                self.age_counter += 1;
                let idx = self.pick_voice();
                self.voices[idx].note_on(
                    event.note,
                    event.velocity,
                    self.portamento,
                    self.age_counter,
                );
            }
            SynthEventKind::NoteOff => {
                for v in &mut self.voices {
                    v.note_off(event.note);
                }
            }
            SynthEventKind::PitchBend(v) => self.pitch_bend = v,
            _ => {}
        }
    }

    fn pick_voice(&mut self) -> usize {
        if self.polyphony != PolyphonyMode::Poly {
            return 0;
        }
        // free voice first, else steal the oldest
        self.voices
            .iter()
            .position(|v| !v.active)
            .unwrap_or_else(|| {
                self.voices
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, v)| v.age)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
    }

    #[link_section = ".iram1.text"]
    pub fn next_sample_stereo(&mut self) -> StereoSample {
        let bend_semi = self.pitch_bend * self.bend_range;
        let mut l = 0.0f32;
        let mut r = 0.0f32;
        for v in &mut self.voices {
            if v.active {
                let s = v.next_stereo(bend_semi);
                l += s.left;
                r += s.right;
            }
        }
        let mut out = StereoSample::new(l, r) * self.master_volume;
        if self.stereo_width != 1.0 {
            out = out.with_width(self.stereo_width);
        }
        StereoSample::new(soft_clip(out.left), soft_clip(out.right))
    }
}

#[inline]
fn soft_clip(x: f32) -> f32 {
    if x.abs() <= 1.0 {
        x
    } else {
        let t = (x.abs() - 1.0).min(2.0);
        x.signum() * (1.0 + t - t * t * 0.25)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> EmbeddedEngine {
        let mut e = EmbeddedEngine::new(8, 44100);
        let mut p = Preset::default();
        p.polyphony_mode = PolyphonyMode::Poly;
        e.apply_preset(&p);
        e
    }

    #[test]
    fn note_on_produces_sound() {
        let mut e = engine();
        e.receive_event(NoteEvent::note_on_midi(60, 100));
        let mut peak = 0.0f32;
        for _ in 0..2000 {
            let s = e.next_sample_stereo();
            peak = peak.max(s.left.abs()).max(s.right.abs());
        }
        assert!(peak > 0.05, "expected audible output, peak={peak}");
    }

    #[test]
    fn note_off_decays_to_silence() {
        let mut e = engine();
        e.receive_event(NoteEvent::note_on_midi(60, 100));
        for _ in 0..2000 {
            e.next_sample_stereo();
        }
        e.receive_event(NoteEvent::note_off(60));
        // release 0.3 s + margin
        for _ in 0..44100 {
            e.next_sample_stereo();
        }
        assert_eq!(e.active_voice_count(), 0);
        let s = e.next_sample_stereo();
        assert_eq!(s.left, 0.0);
    }

    #[test]
    fn polyphony_stacks_voices() {
        let mut e = engine();
        for n in [60u8, 64, 67] {
            e.receive_event(NoteEvent::note_on_midi(n, 100));
            for _ in 0..64 {
                e.next_sample_stereo();
            }
        }
        assert_eq!(e.active_voice_count(), 3);
    }

    #[test]
    fn no_nan_or_inf_in_output() {
        let mut e = engine();
        e.receive_event(NoteEvent::note_on_midi(96, 127));
        for i in 0..44100 {
            let s = e.next_sample_stereo();
            assert!(
                s.left.is_finite() && s.right.is_finite(),
                "non-finite output at sample {i}"
            );
        }
    }
}
