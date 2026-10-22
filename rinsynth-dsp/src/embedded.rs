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

use super::effects::{DistortionType, EffectsChain};
use super::envelope::{ADSREnvelope, Envelope, EnvelopeState};
use super::filter::FilterMode;
use super::oscillator::sine_lut;
use rinsynth_model::event::{NoteEvent, SynthEventKind, WaveformType};
use rinsynth_model::lfo::{LfoDestination, LfoWaveform, LFO};
use rinsynth_model::mod_matrix::{ModDest, ModSource};
use rinsynth_model::param_spec::cc_to_plain;
use rinsynth_model::params::{CCMapping, SynthParam};
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

/// Zavalishin topology-preserving transform (TPT) state-variable filter.
/// Unconditionally stable under fast cutoff/resonance modulation, 4 muls per
/// channel, and self-oscillates musically with bandpass state saturation.
struct TptSvf {
    a1: f32,
    a2: f32,
    a3: f32,
    k: f32,
    in_gain: f32,
    mode: FilterMode,
    ic1: [f32; 2],
    ic2: [f32; 2],
}

impl TptSvf {
    fn new() -> Self {
        Self {
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            k: 2.0,
            in_gain: 1.0,
            mode: FilterMode::LowPass,
            ic1: [0.0; 2],
            ic2: [0.0; 2],
        }
    }

    fn reset(&mut self) {
        self.ic1 = [0.0; 2];
        self.ic2 = [0.0; 2];
    }

    /// Recompute coefficients at control rate (`reso` in 0.0..=1.0).
    fn set(&mut self, mode: FilterMode, cutoff: Frequency, reso: f32, fs: SampleRate) {
        let fc = cutoff.clamp(25.0, fs as f32 * 0.42);
        let w = PI * fc / fs as f32;
        let (sw, cw) = w.sin_cos();
        let g = (sw / cw.max(1e-4)).min(10.0);
        // k = 2/Q: maps reso 0..1 -> k = 2.0..0.08 (Q = 0.5..12.5)
        let r = reso.clamp(0.0, 1.0);
        let k = 2.0 - 1.92 * r;
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        self.a1 = a1;
        self.a2 = a2;
        self.a3 = a3;
        self.k = k;
        // Moderate input headroom compensation at high Q so passband stays
        // full while high-velocity resonant peaks stay controlled.
        self.in_gain = 1.0 / (1.0 + 0.35 * r * r);
        self.mode = mode;
    }

    #[inline(always)]
    #[link_section = ".iram1.text"]
    fn process(&mut self, ch: usize, x: Sample) -> Sample {
        let xin = x * self.in_gain;
        let v3 = (xin + 1e-19) - self.ic2[ch];
        let v1 = self.a1 * self.ic1[ch] + self.a2 * v3;
        let v2 = self.ic2[ch] + self.a2 * self.ic1[ch] + self.a3 * v3;
        let mut ic1 = 2.0 * v1 - self.ic1[ch];
        let ic2 = 2.0 * v2 - self.ic2[ch];
        // Smooth division-free cubic saturation on the bandpass resonator state
        // starting at 0.60 (caps peak around ±0.90) so high Q self-oscillates
        // warmly without harsh clipping on high-velocity strikes.
        if ic1.abs() > 0.60 {
            let c = ic1.clamp(-1.35, 1.35);
            ic1 = c * (1.0 - 0.18 * c * c);
        }
        self.ic1[ch] = ic1;
        self.ic2[ch] = ic2;
        match self.mode {
            FilterMode::LowPass => v2,
            FilterMode::BandPass => v1,
            FilterMode::HighPass => xin - self.k * v1 - v2,
            FilterMode::Notch => xin - self.k * v1,
        }
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
    /// (level × 0.25 × panL, level × 0.25 × panR) per osc
    gains: [(f32, f32); MAX_OSCS],
    osc_level: [f32; MAX_OSCS],
    osc_pan: [f32; MAX_OSCS],
    osc2_semi: i8,
    osc2_cents: i8,
    detune: [f32; MAX_OSCS],
    amp_env: ADSREnvelope,
    fenv: ADSREnvelope,
    fenv_amount: f32,
    filter: TptSvf,
    filter_mode: FilterMode,
    resonance: f32, // 0.0..1.0
    base_cutoff: Frequency,
    lfo: LFO,
    lfo_depth: f32,
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
            osc_pan: [0.0; MAX_OSCS],
            osc2_semi: 0,
            osc2_cents: 0,
            detune: [1.0; MAX_OSCS],
            amp_env: ADSREnvelope::new(0.01, 0.1, 0.7, 0.2, sample_rate),
            fenv: ADSREnvelope::new(0.01, 0.3, 0.0, 0.3, sample_rate),
            fenv_amount: 0.0,
            filter: TptSvf::new(),
            filter_mode: FilterMode::LowPass,
            resonance: 0.2,
            base_cutoff: 8000.0,
            lfo: LFO::new((sample_rate / CONTROL_DIV).max(1)),
            lfo_depth: 0.0,
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

    fn update_gains(&mut self) {
        for i in 0..MAX_OSCS {
            let (pl, pr) = pan_gains(self.osc_pan[i]);
            let lvl = self.osc_level[i] * 0.25;
            self.gains[i] = (lvl * pl, lvl * pr);
        }
    }

    fn note_on(&mut self, note: MidiNote, velocity: Amplitude, portamento: f32, age: u64) {
        let was_active = self.active && self.amp_env.state() != EnvelopeState::Idle;
        let freq = midi_to_frequency(note);
        self.note = note;
        // Gentle velocity curve (concave power) so max-velocity hits stay
        // punchy without jumping disproportionately in peak level.
        self.velocity = velocity.clamp(0.0, 1.0).sqrt();
        self.active = true;
        self.age = age;
        if portamento > 0.005 && was_active {
            // legato glide — start from wherever the last note left us
            self.glide_target = freq;
            self.glide_step = (freq - self.base_freq) / (portamento * self.sample_rate as f32);
            self.gliding = self.glide_step != 0.0;
        } else {
            self.base_freq = freq;
            self.gliding = false;
            if !was_active {
                self.filter.reset();
            }
        }
        if !was_active {
            self.amp_env.reset();
            self.fenv.reset();
            self.lfo.reset();
            for o in &mut self.oscs {
                o.reset();
            }
        }
        self.amp_env.trigger();
        self.fenv.trigger();
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
    fn control_tick(&mut self, bend_semitones: f32, mod_wheel: f32) {
        let eff_depth = (self.lfo_depth + mod_wheel * 0.6).clamp(0.0, 1.0);
        self.lfo.set_depth(eff_depth);
        let lfo = self.lfo.next_value(); // -eff_depth..eff_depth
        let lfo_pitch = if self.lfo_dest == LfoDestination::Pitch {
            lfo * 2.0 // ±2 semitones at full depth
        } else {
            0.0
        };
        self.set_osc_freqs(bend_semitones + lfo_pitch);

        let lfo_cutoff = if self.lfo_dest == LfoDestination::FilterCutoff {
            lfo * 2.5 // ±2.5 octaves at full depth
        } else {
            0.0
        };
        // Filter env: 6-octave sweep at full amount (matches the rich engine)
        let env_sweep = 6.0 * self.fenv_amount * self.fenv_amp;
        let cutoff = (self.base_cutoff * 2.0f32.powf(env_sweep + lfo_cutoff))
            .clamp(25.0, self.sample_rate as f32 * 0.42);
        self.filter
            .set(self.filter_mode, cutoff, self.resonance, self.sample_rate);

        self.amp_lfo = if self.lfo_dest == LfoDestination::Amplitude {
            1.0 + lfo * 0.5
        } else {
            1.0
        };
    }

    #[link_section = ".iram1.text"]
    fn next_stereo(&mut self, bend_semitones: f32, mod_wheel: f32) -> StereoSample {
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
            self.control_tick(bend_semitones, mod_wheel);
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
    pub effects: EffectsChain,
    cc_mapping: CCMapping,
    master_volume: f32,
    mod_wheel: f32,       // 0..1 from CC1
    pitch_bend: f32,      // -1..1
    bend_range: f32,      // semitones
    portamento: f32,
    polyphony: PolyphonyMode,
    age_counter: u64,
    stereo_width: f32,
    comp_env: f32,        // stereo bus compressor/limiter envelope follower
}

impl EmbeddedEngine {
    pub fn new(max_voices: usize, sample_rate: SampleRate) -> Self {
        Self {
            voices: (0..max_voices).map(|_| LeanVoice::new(sample_rate)).collect(),
            effects: EffectsChain::new(sample_rate),
            cc_mapping: CCMapping::default_mappings(),
            master_volume: 0.7,
            mod_wheel: 0.0,
            pitch_bend: 0.0,
            bend_range: 12.0,
            portamento: 0.0,
            polyphony: PolyphonyMode::Poly,
            age_counter: 0,
            stereo_width: 1.0,
            comp_env: 0.0,
        }
    }

    pub fn set_master_volume(&mut self, v: Amplitude) {
        self.master_volume = v.clamp(0.0, 1.0);
    }

    pub fn set_filter_mode(&mut self, mode: FilterMode) {
        for v in &mut self.voices {
            v.filter_mode = mode;
        }
    }

    pub fn set_mod_wheel(&mut self, v: f32) {
        self.mod_wheel = v.clamp(0.0, 1.0);
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
        let reso = preset.filter_resonance.clamp(0.0, 1.0);

        // Fold primary mod-matrix routes into the concrete voice slots at
        // preset-load time (zero per-sample matrix cost).
        let mut fenv_amount = preset.filter_env_amount;
        let mut lfo_dest = preset.lfo_destination;
        let mut lfo_depth = preset.lfo_depth;
        for slot in &preset.mod_matrix.slots {
            if !slot.enabled {
                continue;
            }
            if slot.source == ModSource::FilterEnv
                && slot.dest == ModDest::FilterCutoff
                && fenv_amount == 0.0
            {
                fenv_amount = slot.amount;
            } else if slot.source == ModSource::Lfo1 && lfo_dest == LfoDestination::Off {
                match slot.dest {
                    ModDest::FilterCutoff => {
                        lfo_dest = LfoDestination::FilterCutoff;
                        if lfo_depth == 0.0 {
                            lfo_depth = slot.amount.abs().clamp(0.0, 1.0);
                        }
                    }
                    ModDest::AllPitch | ModDest::Osc1Pitch | ModDest::Osc2Pitch => {
                        lfo_dest = LfoDestination::Pitch;
                        if lfo_depth == 0.0 {
                            lfo_depth = slot.amount.abs().clamp(0.0, 1.0);
                        }
                    }
                    ModDest::Amplitude => {
                        lfo_dest = LfoDestination::Amplitude;
                        if lfo_depth == 0.0 {
                            lfo_depth = slot.amount.abs().clamp(0.0, 1.0);
                        }
                    }
                    _ => {}
                }
            }
        }
        // If no LFO destination is set at all, route LFO -> FilterCutoff at 0
        // base depth so the mod wheel (CC1) always has an audible target.
        if lfo_dest == LfoDestination::Off && lfo_depth == 0.0 {
            lfo_dest = LfoDestination::FilterCutoff;
        }

        for v in &mut self.voices {
            v.oscs[0].kind = preset.osc1_waveform;
            v.oscs[0].set_duty(preset.osc1_pw);
            v.oscs[1].kind = preset.osc2_waveform;
            v.oscs[1].set_duty(preset.osc2_pw);
            v.osc_level = [preset.osc1_level, preset.osc2_level];
            v.osc_pan = [preset.osc1_pan, preset.osc2_pan];
            v.osc2_semi = preset.osc2_semitones;
            v.osc2_cents = preset.osc2_cents;
            v.detune = [
                1.0,
                super::oscillator::detune_ratio(preset.osc2_semitones, preset.osc2_cents),
            ];
            v.update_gains();
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
            v.fenv_amount = fenv_amount;
            v.base_cutoff = preset.filter_cutoff;
            v.resonance = reso;
            v.lfo.set_rate(preset.lfo_rate);
            v.lfo_depth = lfo_depth;
            v.lfo.set_depth(lfo_depth);
            v.lfo.set_waveform(preset.lfo_waveform);
            v.lfo_dest = lfo_dest;
        }

        // Sync effects chain from preset
        self.effects.set_distortion_enabled(preset.distortion_enabled);
        self.effects.set_distortion_drive(preset.distortion_drive);
        self.effects.set_distortion_mix(preset.distortion_mix);
        self.effects.set_distortion_type(preset.distortion_type);

        self.effects.set_chorus_enabled(preset.chorus_enabled);
        self.effects.set_chorus_rate(preset.chorus_rate);
        self.effects.set_chorus_depth(preset.chorus_depth);
        self.effects.set_chorus_mix(preset.chorus_mix);

        self.effects.set_delay_enabled(preset.delay_enabled);
        self.effects.set_delay_time(preset.delay_time);
        self.effects.set_delay_feedback(preset.delay_feedback);
        self.effects.set_delay_mix(preset.delay_mix);
        self.effects.set_bpm(preset.tempo_bpm);
        self.effects
            .set_delay_tempo_sync(preset.delay_sync, preset.delay_beats);

        self.effects.set_reverb_enabled(preset.reverb_enabled);
        self.effects.set_reverb_room_size(preset.reverb_room_size);
        self.effects.set_reverb_damping(preset.reverb_damping);
        self.effects.set_reverb_mix(preset.reverb_mix);
    }

    /// Apply a plain-valued `SynthParam` update across all voices and effects.
    pub fn set_param(&mut self, param: SynthParam, plain: f32) {
        match param {
            SynthParam::Attack => {
                for v in &mut self.voices {
                    v.amp_env.set_attack(plain);
                }
            }
            SynthParam::Decay => {
                for v in &mut self.voices {
                    v.amp_env.set_decay(plain);
                }
            }
            SynthParam::Sustain => {
                for v in &mut self.voices {
                    v.amp_env.set_sustain(plain);
                }
            }
            SynthParam::Release => {
                for v in &mut self.voices {
                    v.amp_env.set_release(plain);
                }
            }
            SynthParam::FilterCutoff => {
                for v in &mut self.voices {
                    v.base_cutoff = plain.clamp(25.0, 20000.0);
                }
            }
            SynthParam::FilterResonance => {
                let r = plain.clamp(0.0, 1.0);
                for v in &mut self.voices {
                    v.resonance = r;
                }
            }
            SynthParam::FilterAttack => {
                for v in &mut self.voices {
                    v.fenv.set_attack(plain);
                }
            }
            SynthParam::FilterDecay => {
                for v in &mut self.voices {
                    v.fenv.set_decay(plain);
                }
            }
            SynthParam::FilterSustain => {
                for v in &mut self.voices {
                    v.fenv.set_sustain(plain);
                }
            }
            SynthParam::FilterRelease => {
                for v in &mut self.voices {
                    v.fenv.set_release(plain);
                }
            }
            SynthParam::FilterEnvAmount => {
                for v in &mut self.voices {
                    v.fenv_amount = plain.clamp(-1.0, 1.0);
                }
            }
            SynthParam::LfoRate => {
                for v in &mut self.voices {
                    v.lfo.set_rate(plain);
                }
            }
            SynthParam::LfoDepth => {
                for v in &mut self.voices {
                    v.lfo_depth = plain.clamp(0.0, 1.0);
                }
            }
            SynthParam::LfoWaveform => {
                let wf = match plain as i32 {
                    0 => LfoWaveform::Sine,
                    1 => LfoWaveform::Triangle,
                    2 => LfoWaveform::Square,
                    3 => LfoWaveform::Saw,
                    _ => LfoWaveform::Random,
                };
                for v in &mut self.voices {
                    v.lfo.set_waveform(wf);
                }
            }
            SynthParam::LfoDestination => {
                let dest = match plain as i32 {
                    0 => LfoDestination::Off,
                    1 => LfoDestination::Pitch,
                    2 => LfoDestination::FilterCutoff,
                    _ => LfoDestination::Amplitude,
                };
                for v in &mut self.voices {
                    v.lfo_dest = dest;
                }
            }
            SynthParam::PitchBendRange => {
                self.bend_range = plain.clamp(1.0, 24.0);
            }
            SynthParam::PortamentoTime => {
                self.portamento = plain.max(0.0);
            }
            SynthParam::Osc1Waveform => {
                let wf = WaveformType::from_index(plain as u8);
                for v in &mut self.voices {
                    v.oscs[0].kind = wf;
                }
            }
            SynthParam::Osc1Level => {
                let lvl = plain.clamp(0.0, 1.0);
                for v in &mut self.voices {
                    v.osc_level[0] = lvl;
                    v.update_gains();
                }
            }
            SynthParam::Osc1Pw => {
                for v in &mut self.voices {
                    v.oscs[0].set_duty(plain);
                }
            }
            SynthParam::Osc1Pan => {
                for v in &mut self.voices {
                    v.osc_pan[0] = plain.clamp(-1.0, 1.0);
                    v.update_gains();
                }
            }
            SynthParam::Osc2Waveform => {
                let wf = WaveformType::from_index(plain as u8);
                for v in &mut self.voices {
                    v.oscs[1].kind = wf;
                }
            }
            SynthParam::Osc2Level => {
                let lvl = plain.clamp(0.0, 1.0);
                for v in &mut self.voices {
                    v.osc_level[1] = lvl;
                    v.update_gains();
                }
            }
            SynthParam::Osc2Semitones => {
                let semi = plain.round() as i8;
                for v in &mut self.voices {
                    v.osc2_semi = semi;
                    v.detune[1] = super::oscillator::detune_ratio(v.osc2_semi, v.osc2_cents);
                }
            }
            SynthParam::Osc2Cents => {
                let cents = plain.round() as i8;
                for v in &mut self.voices {
                    v.osc2_cents = cents;
                    v.detune[1] = super::oscillator::detune_ratio(v.osc2_semi, v.osc2_cents);
                }
            }
            SynthParam::Osc2Pw => {
                for v in &mut self.voices {
                    v.oscs[1].set_duty(plain);
                }
            }
            SynthParam::Osc2Pan => {
                for v in &mut self.voices {
                    v.osc_pan[1] = plain.clamp(-1.0, 1.0);
                    v.update_gains();
                }
            }
            SynthParam::StereoWidth => {
                self.stereo_width = plain.clamp(0.0, 2.0);
            }
            SynthParam::MasterVolume => {
                self.master_volume = plain.clamp(0.0, 1.0);
            }
            SynthParam::DistortionType => {
                self.effects
                    .set_distortion_type(DistortionType::from_index(plain as u8));
            }
            _ => {}
        }
    }

    pub fn receive_event(&mut self, event: NoteEvent) {
        match event.kind {
            SynthEventKind::NoteOn => {
                self.age_counter += 1;
                let idx = self.pick_voice(event.note);
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
            SynthEventKind::ControlChange { cc, value } => {
                let norm = value as f32 / 127.0;
                match cc {
                    1 => self.mod_wheel = norm,
                    7 => self.master_volume = norm,
                    91 => {
                        self.effects.set_reverb_enabled(value > 0);
                        self.effects.set_reverb_mix(norm * 0.6);
                    }
                    93 => {
                        self.effects.set_chorus_enabled(value > 0);
                        self.effects.set_chorus_mix(norm * 0.6);
                    }
                    94 => {
                        self.effects.set_delay_enabled(value > 0);
                        self.effects.set_delay_mix(norm * 0.6);
                    }
                    _ => {}
                }
                if let Some(param) = self.cc_mapping.get_param(cc) {
                    let plain = cc_to_plain(param, value);
                    self.set_param(param, plain);
                }
            }
            _ => {}
        }
    }

    fn pick_voice(&mut self, note: MidiNote) -> usize {
        if self.polyphony != PolyphonyMode::Poly {
            return 0;
        }
        // 1. Retrigger: if a voice is already playing this exact MIDI note
        //    (whether held or in its release tail), reuse it so hammering notes
        //    never exhausts the polyphony pool with duplicate tails.
        if let Some(i) = self.voices.iter().position(|v| v.active && v.note == note) {
            return i;
        }
        // 2. Idle voice: pick the first completely free slot.
        if let Some(i) = self.voices.iter().position(|v| !v.active) {
            return i;
        }
        // 3. Releasing voice: steal the oldest releasing voice so held notes
        //    (Attack/Decay/Sustain) are never killed by release tails.
        if let Some((i, _)) = self
            .voices
            .iter()
            .enumerate()
            .filter(|(_, v)| v.amp_env.state() == EnvelopeState::Release)
            .min_by_key(|(_, v)| v.age)
        {
            return i;
        }
        // 4. All voices are physically held: protect the lowest held bass note
        //    and steal the oldest among the remaining held voices.
        let lowest_note = self.voices.iter().map(|v| v.note).min().unwrap_or(0);
        self.voices
            .iter()
            .enumerate()
            .filter(|(_, v)| v.note != lowest_note)
            .min_by_key(|(_, v)| v.age)
            .or_else(|| self.voices.iter().enumerate().min_by_key(|(_, v)| v.age))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[link_section = ".iram1.text"]
    pub fn next_sample_stereo(&mut self) -> StereoSample {
        let bend_semi = self.pitch_bend * self.bend_range;
        let mod_wheel = self.mod_wheel;
        let mut l = 0.0f32;
        let mut r = 0.0f32;
        for v in &mut self.voices {
            if v.active {
                let s = v.next_stereo(bend_semi, mod_wheel);
                l += s.left;
                r += s.right;
            }
        }
        let mut out = StereoSample::new(l, r);
        if self.stereo_width != 1.0 {
            out = out.with_width(self.stereo_width);
        }
        out = self.effects.process_stereo(out) * self.master_volume;

        // Stereo bus peak compressor/limiter: fast (~1.5 ms) attack, smooth
        // (~120 ms) release above 0.60 threshold so resonant or multi-voice
        // peaks are leveled cleanly without waveshaper buzz.
        let peak = out.left.abs().max(out.right.abs());
        let coeff = if peak > self.comp_env { 0.015 } else { 0.0002 };
        self.comp_env += coeff * (peak - self.comp_env);
        if self.comp_env < 1e-18 {
            self.comp_env = 0.0;
        }
        if self.comp_env > 0.60 {
            let excess = self.comp_env - 0.60;
            let target = 0.60 + 0.32 * (excess / (0.32 + excess));
            let gain = target / self.comp_env;
            out.left *= gain;
            out.right *= gain;
        }

        StereoSample::new(soft_clip(out.left), soft_clip(out.right))
    }
}

#[inline]
fn soft_clip(x: f32) -> f32 {
    if x.abs() <= 0.75 {
        x
    } else {
        // Smoothly compress above 0.75 into [-1.0, 1.0] (never exceeds ±1).
        let excess = x.abs() - 0.75;
        x.signum() * (0.75 + 0.25 * (excess / (0.25 + excess)))
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

    #[test]
    fn high_resonance_and_fx_chain_stay_finite_and_audible() {
        let mut e = engine();
        // Set cutoff to 800 Hz (CC 74 = 64) and max resonance (CC 71 = 127),
        // plus enable Chorus (CC 93), Delay (CC 94), and Reverb (CC 91).
        e.receive_event(NoteEvent::control_change(74, 64));
        e.receive_event(NoteEvent::control_change(71, 127));
        e.receive_event(NoteEvent::control_change(91, 60));
        e.receive_event(NoteEvent::control_change(93, 60));
        e.receive_event(NoteEvent::control_change(94, 60));
        e.receive_event(NoteEvent::note_on_midi(48, 120));
        let mut peak = 0.0f32;
        for i in 0..8820 {
            let s = e.next_sample_stereo();
            assert!(
                s.left.is_finite() && s.right.is_finite(),
                "non-finite output at sample {i}"
            );
            peak = peak.max(s.left.abs()).max(s.right.abs());
        }
        assert!(peak > 0.1 && peak <= 1.0, "unexpected peak={peak}");
    }

    #[test]
    fn held_bass_note_survives_repeated_chord_strums() {
        let mut e = engine();
        // Hold low C2 (MIDI 36)
        e.receive_event(NoteEvent::note_on_midi(36, 110));
        for _ in 0..4410 {
            e.next_sample_stereo();
        }
        // Repeatedly strum 6 different upper notes (releasing between strums)
        for round in 0..12 {
            let chord = [60u8, 64, 67, 71, 74, 76];
            for &n in &chord {
                e.receive_event(NoteEvent::note_on_midi(n, 120));
                for _ in 0..128 {
                    e.next_sample_stereo();
                }
            }
            for &n in &chord {
                e.receive_event(NoteEvent::note_off(n));
            }
            for _ in 0..512 {
                e.next_sample_stereo();
            }
            let bass_alive = e
                .voices
                .iter()
                .any(|v| v.active && v.note == 36 && v.amp_env.state() != EnvelopeState::Release);
            assert!(bass_alive, "held bass note 36 was killed at round {round}");
        }
    }
}
