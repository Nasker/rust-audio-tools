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
use crate::profile;
use super::envelope::{ADSREnvelope, Envelope, EnvelopeState};
use super::filter::FilterMode;
use super::oscillator::sine_lut;
use rinsynth_model::event::{NoteEvent, SynthEventKind, WaveformType};
use rinsynth_model::lfo::{LfoDestination, LfoWaveform, LFO};
use rinsynth_model::mod_matrix::{ModDest, ModMatrix, ModSlot, ModSource, MOD_SLOTS};
use rinsynth_model::param_spec::cc_to_plain;
use rinsynth_model::params::{CCMapping, SynthParam};
use rinsynth_model::presets::Preset;
use rinsynth_model::types::{
    midi_to_frequency, Amplitude, Frequency, MidiNote, Sample, SampleRate, StereoSample,
};
use rinsynth_model::voice::PolyphonyMode;
use std::f32::consts::PI;

/// Samples between control-rate updates (LFO tick, mod matrix, filter coeffs).
const CONTROL_DIV: u32 = 64;
const INV_CONTROL_DIV: f32 = 1.0 / (CONTROL_DIV as f32);
const MAX_OSCS: usize = 4;
/// Per-oscillator static gain staging (~-9.3 dB below the old 0.25 single-voice
/// scale). Leaves generous internal headroom for resonant low-frequency filters
/// and polyphonic chords; the post-FX Master Volume stage provides up to +3.5 dB
/// (1.5x) of clean digital makeup gain at the top of the throw.
const OSC_GAIN: f32 = 0.085;
/// Mono-mode key stack capacity (last-note priority, newest on top).
const KEY_STACK_CAP: usize = 16;
/// Per-sample decrement of the waveform-swap dip ramp — a full fade-out +
/// fade-in on one oscillator spans ~2.9 ms @ 44.1 kHz, masking the
/// discontinuity of switching shapes mid-note.
const WF_DIP_STEP: f32 = 1.0 / 64.0;

/// Cycle-count a pipeline stage when a profiler hook is installed; a single
/// predictable branch per call otherwise.
macro_rules! pstage {
    ($idx:expr, $body:expr) => {{
        if profile::enabled() {
            let _t0 = profile::now();
            let _r = $body;
            profile::add($idx, profile::now() - _t0);
            _r
        } else {
            $body
        }
    }};
}

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
    /// Previous output mode during a mode-switch crossfade.
    prev_mode: FilterMode,
    /// Remaining mode-crossfade samples (0 = fade complete).
    xfade: u8,
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
            prev_mode: FilterMode::LowPass,
            xfade: 0,
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
        // Input headroom compensation at high Q so low-frequency fundamentals
        // hitting the resonant peak don't overdrive the voice budget.
        self.in_gain = 1.0 / (1.0 + 1.1 * r * r);
        if mode != self.mode {
            self.prev_mode = self.mode;
            self.xfade = 64; // ~1.5 ms crossfade between mode outputs
        }
        self.mode = mode;
    }

    /// Filter output for a given mode — a linear combo of the resonator
    /// states already computed in `process`, so evaluating both modes during
    /// a switch crossfade costs only a few extra muls.
    #[inline(always)]
    fn mode_out(mode: FilterMode, xin: f32, k: f32, v1: f32, v2: f32) -> f32 {
        match mode {
            FilterMode::LowPass => v2,
            FilterMode::BandPass => v1,
            FilterMode::HighPass => xin - k * v1 - v2,
            FilterMode::Notch => xin - k * v1,
        }
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
        // C1-continuous division-free cubic soft saturation on the bandpass
        // resonator state above 0.24 (asymptotes cleanly to ±0.40). Eliminates
        // the old step discontinuity at 0.60 and keeps low-frequency resonant
        // peaks inside the per-voice headroom budget.
        let abs_ic1 = ic1.abs();
        if abs_ic1 > 0.24 {
            let u = ((abs_ic1 - 0.24) * 4.166_666_5).min(1.0);
            let sat = 0.24 + 0.24 * u * (1.0 - 0.333_333_34 * u * u);
            ic1 = ic1.signum() * sat;
        }
        self.ic1[ch] = ic1;
        self.ic2[ch] = ic2;
        let out = Self::mode_out(self.mode, xin, self.k, v1, v2);
        if self.xfade > 0 {
            self.xfade -= 1;
            let old = Self::mode_out(self.prev_mode, xin, self.k, v1, v2);
            let t = self.xfade as f32 * (1.0 / 64.0);
            out + (old - out) * t
        } else {
            out
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
    /// Smoothed (level × OSC_GAIN × panL, level × OSC_GAIN × panR) per osc
    gains: [(f32, f32); MAX_OSCS],
    /// Per-sample linear ramp increments across the 64-sample control tick
    gain_steps: [(f32, f32); MAX_OSCS],
    /// Target per-oscillator stereo gains (smoothed toward in `control_tick`)
    target_gains: [(f32, f32); MAX_OSCS],
    osc_level: [f32; MAX_OSCS],
    osc_pan: [f32; MAX_OSCS],
    osc_pw: [f32; MAX_OSCS],
    smooth_pw: [f32; MAX_OSCS],
    osc_semi: [i8; MAX_OSCS],
    osc_cents: [i8; MAX_OSCS],
    detune: [f32; MAX_OSCS],
    target_detune: [f32; MAX_OSCS],
    amp_env: ADSREnvelope,
    fenv: ADSREnvelope,
    env3: ADSREnvelope,
    /// Configured amp release — `shed_voice` temporarily shortens the live
    /// release for a fast fade; `note_on` restores this value.
    cfgd_release: f32,
    fenv_amount: f32,
    /// Smoothed Env Amount — the raw value scales the audio-rate fenv
    /// output, so a slider step would jump the cutoff while the env is hot.
    smooth_fenv_amt: f32,
    filter: TptSvf,
    filter_mode: FilterMode,
    /// Wide-open lowpass with ~no resonance is audibly transparent — skip the
    /// per-sample TPT state update entirely. Evaluated each control tick.
    filter_bypass: bool,
    resonance: f32, // 0.0..1.0
    smooth_reso: f32,
    base_cutoff: Frequency,
    smooth_cutoff: Frequency,
    smooth_cutoff_oct: f32,
    lfo: LFO,
    lfo2: LFO,
    lfo_rate: f32,
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
    amp_lfo_step: f32,
    last_amp: f32,
    fenv_amp: f32,
    env3_amp: f32,
    /// Audio-rate FM amount from Osc4 -> Osc1/AllPitch
    osc4_fm: f32,
    target_osc4_fm: f32,
    osc4_fm_step: f32,
    /// Audio-rate Ring-Mod amount from Osc4 -> Osc1Level/Amplitude
    osc4_rm: f32,
    target_osc4_rm: f32,
    osc4_rm_step: f32,
    /// Pending waveform swap per osc (0xFF = none) — dips that osc's output
    /// to silence over ~64 samples, swaps the shape, then fades back in.
    wf_pending: [u8; MAX_OSCS],
    wf_fade: [f32; MAX_OSCS],
    /// Smoothed whole-voice pitch offset in semitones (bend + LFO/matrix).
    smooth_pitch: f32,
    /// Smoothed per-osc pitch offsets in semitones (matrix Osc*Pitch).
    smooth_osc_pitch: [f32; MAX_OSCS],
    /// Unison pitch multiplier (mono-mode voice stacking) — folded into
    /// `set_osc_freqs`; 1.0 for a plain poly/mono voice.
    uni_mul: f32,
    target_uni_mul: f32,
    /// Unison stereo gain pair applied to the voice output — includes the
    /// 1/sqrt(n) loudness compensation so a unison stack doesn't slam the
    /// bus compressor. (1.0, 1.0) = unity stereo.
    uni_gain: (f32, f32),
    uni_gain_step: (f32, f32),
    target_uni_gain: (f32, f32),
    /// Bitmask of oscillators that may emit audio this control block
    /// (bit i = osc i renders) — recomputed each control tick so the
    /// per-sample osc gate is a single bit test.
    osc_live: u8,
    /// Any ramp steps currently nonzero — when false, the per-sample
    /// gain/amp/fm/rm additions are exact no-ops and get skipped wholesale.
    ramping: bool,
    /// Any waveform dip in progress or pending — skips the fade mul + state
    /// machine on the (overwhelmingly common) steady-state path.
    wf_fading: bool,
    /// Last inputs pushed into `filter.set` — the powf/sin_cos/divide chain
    /// is skipped while all three hold still (see control_tick).
    last_oct: f32,
    last_reso: f32,
    /// Effective cutoff from the last `filter.set` — reused by the bypass
    /// evaluation on ticks where coefficients aren't recomputed.
    last_cutoff: Frequency,
    sample_rate: SampleRate,
}

impl LeanVoice {
    fn new(sample_rate: SampleRate) -> Self {
        let ctrl_sr = (sample_rate / CONTROL_DIV).max(1);
        Self {
            oscs: [
                LeanOsc::new(WaveformType::Saw),
                LeanOsc::new(WaveformType::Saw),
                LeanOsc::new(WaveformType::Square),
                LeanOsc::new(WaveformType::Sine),
            ],
            gains: [(0.0, 0.0); MAX_OSCS],
            gain_steps: [(0.0, 0.0); MAX_OSCS],
            target_gains: [(0.0, 0.0); MAX_OSCS],
            osc_level: [0.0; MAX_OSCS],
            osc_pan: [0.0; MAX_OSCS],
            osc_pw: [0.5; MAX_OSCS],
            smooth_pw: [0.5; MAX_OSCS],
            osc_semi: [0, 0, -12, 0],
            osc_cents: [0, 7, 0, 0],
            detune: [1.0; MAX_OSCS],
            target_detune: [1.0; MAX_OSCS],
            amp_env: ADSREnvelope::new(0.01, 0.1, 0.7, 0.2, sample_rate),
            fenv: ADSREnvelope::new(0.01, 0.3, 0.0, 0.3, sample_rate),
            env3: ADSREnvelope::new(0.01, 0.3, 0.0, 0.3, ctrl_sr),
            cfgd_release: 0.2,
            fenv_amount: 0.0,
            smooth_fenv_amt: 0.0,
            filter: TptSvf::new(),
            filter_mode: FilterMode::LowPass,
            filter_bypass: false,
            resonance: 0.2,
            smooth_reso: 0.2,
            base_cutoff: 8000.0,
            smooth_cutoff: 8000.0,
            smooth_cutoff_oct: 0.0,
            lfo: LFO::new(ctrl_sr),
            lfo2: LFO::new(ctrl_sr),
            lfo_rate: 4.0,
            lfo_depth: 0.0,
            lfo_dest: LfoDestination::Off,
            base_freq: 440.0,
            glide_step: 0.0,
            glide_target: 440.0,
            gliding: false,
            velocity: 0.0,
            note: 0,
            active: false,
            age: u64::MAX,
            control_phase: 0,
            amp_lfo: 1.0,
            amp_lfo_step: 0.0,
            last_amp: 0.0,
            fenv_amp: 0.0,
            env3_amp: 0.0,
            osc4_fm: 0.0,
            target_osc4_fm: 0.0,
            osc4_fm_step: 0.0,
            osc4_rm: 0.0,
            target_osc4_rm: 0.0,
            osc4_rm_step: 0.0,
            wf_pending: [0xFF; MAX_OSCS],
            wf_fade: [1.0; MAX_OSCS],
            smooth_pitch: 0.0,
            smooth_osc_pitch: [0.0; MAX_OSCS],
            uni_mul: 1.0,
            target_uni_mul: 1.0,
            uni_gain: (1.0, 1.0),
            uni_gain_step: (0.0, 0.0),
            target_uni_gain: (1.0, 1.0),
            osc_live: 0,
            ramping: false,
            wf_fading: false,
            last_oct: 1e30,
            last_reso: 1e30,
            last_cutoff: 20000.0,
            sample_rate,
        }
    }

    #[inline]
    fn osc_level_gain(kind: WaveformType, level: f32) -> f32 {
        let x = level.clamp(0.0, 1.0);
        match kind {
            // Broadband white noise has high energy in the ear's 2–5 kHz
            // sensitivity band: use a quadratic audio taper (x^2) so the bottom
            // 15% of the slider gives fine "air"/breath control (-84 dB..-36 dB)
            // while full throw (1.0) reaches 0.65 * OSC_GAIN (equal-loudness
            // matched to a full-level Saw/Square wave).
            WaveformType::Noise => x * x * 0.65 * OSC_GAIN,
            // Gentle audio taper on pitched oscillators: fine control on subtle
            // sub/harmonic layers near the bottom, full 1.0 * OSC_GAIN at 1.0.
            _ => x * (0.4 + 0.6 * x) * OSC_GAIN,
        }
    }

    fn update_gains(&mut self) {
        for i in 0..MAX_OSCS {
            let (pl, pr) = pan_gains(self.osc_pan[i]);
            let lvl = Self::osc_level_gain(self.oscs[i].kind, self.osc_level[i]);
            self.target_gains[i] = (lvl * pl, lvl * pr);
            if !self.active {
                self.gains[i] = self.target_gains[i];
            }
        }
    }

    /// Change an oscillator's waveform without a click: an active voice dips
    /// that osc to silence over ~64 samples (`step_wf_fade` swaps the shape
    /// at the bottom), an idle voice switches immediately.
    fn set_osc_waveform(&mut self, i: usize, wf: WaveformType) {
        if self.active && self.oscs[i].kind != wf {
            self.wf_pending[i] = wf as u8;
            self.wf_fading = true;
        } else {
            self.oscs[i].kind = wf;
        }
    }

    /// Per-sample waveform-dip state machine for `set_osc_waveform`.
    #[inline(always)]
    fn step_wf_fade(&mut self, i: usize) {
        if self.wf_pending[i] != 0xFF {
            self.wf_fade[i] -= WF_DIP_STEP;
            if self.wf_fade[i] <= 0.0 {
                self.oscs[i].kind = WaveformType::from_index(self.wf_pending[i]);
                self.wf_pending[i] = 0xFF;
                self.wf_fade[i] = 0.0;
                // The gain taper is waveform-dependent (Noise uses x^2) —
                // refresh targets so the fade-in lands on the right level.
                self.update_gains();
            }
        } else if self.wf_fade[i] < 1.0 {
            self.wf_fade[i] = (self.wf_fade[i] + WF_DIP_STEP).min(1.0);
        }
    }

    fn note_on(&mut self, note: MidiNote, velocity: Amplitude, portamento: f32, age: u64) {
        let was_active = self.active && self.amp_env.state() != EnvelopeState::Idle;
        let freq = midi_to_frequency(note);
        self.note = note;
        self.velocity = velocity.clamp(0.0, 1.0);
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
            self.gains = self.target_gains;
            self.gain_steps = [(0.0, 0.0); MAX_OSCS];
            self.smooth_pw = self.osc_pw;
            self.detune = self.target_detune;
            self.uni_mul = self.target_uni_mul;
            self.uni_gain = self.target_uni_gain;
            self.uni_gain_step = (0.0, 0.0);
            self.amp_lfo = 1.0;
            self.amp_lfo_step = 0.0;
            self.smooth_cutoff = self.base_cutoff;
            self.smooth_cutoff_oct = 0.0;
            self.smooth_reso = self.resonance;
            self.smooth_pitch = 0.0;
            self.smooth_osc_pitch = [0.0; MAX_OSCS];
            self.smooth_fenv_amt = self.fenv_amount;
            self.osc4_fm = self.target_osc4_fm;
            self.osc4_rm = self.target_osc4_rm;
            self.osc4_fm_step = 0.0;
            self.osc4_rm_step = 0.0;
            self.ramping = false;
            self.last_oct = 1e30;
            self.last_reso = 1e30;
            let mut live = 0u8;
            for i in 0..MAX_OSCS {
                if self.target_gains[i].0 > 0.0 || self.target_gains[i].1 > 0.0 {
                    live |= 1 << i;
                }
            }
            if self.osc4_fm != 0.0 || self.osc4_rm != 0.0 {
                live |= 1 << 3;
            }
            self.osc_live = live;
            self.wf_fade = [1.0; MAX_OSCS];
            for i in 0..MAX_OSCS {
                if self.wf_pending[i] != 0xFF {
                    self.oscs[i].kind = WaveformType::from_index(self.wf_pending[i]);
                    self.wf_pending[i] = 0xFF;
                }
            }
            self.amp_env.reset();
            self.fenv.reset();
            self.env3.reset();
            self.lfo.reset();
            self.lfo2.reset();
            for o in &mut self.oscs {
                o.reset();
            }
        }
        // Restore the configured release — a previous governor shed may have
        // left a fast-fade release programmed on this envelope.
        self.amp_env.set_release(self.cfgd_release);
        self.amp_env.trigger();
        self.fenv.trigger();
        self.env3.trigger();
        self.set_osc_freqs(0.0, [0.0; MAX_OSCS]);
        self.control_phase = 0;
    }

    fn note_off(&mut self, note: MidiNote) {
        if self.active && self.note == note {
            self.force_release();
        }
    }

    /// Release regardless of the sounding note (mode switches, panic).
    fn force_release(&mut self) {
        if self.active {
            self.amp_env.release();
            self.fenv.release();
            self.env3.release();
        }
    }

    #[inline]
    fn set_osc_freqs(&mut self, all_semi_offset: f32, per_osc_semi: [f32; MAX_OSCS]) {
        let bend = 2.0f32.powf(all_semi_offset / 12.0);
        for i in 0..MAX_OSCS {
            let osc_mul = if per_osc_semi[i] == 0.0 {
                1.0
            } else {
                2.0f32.powf(per_osc_semi[i] / 12.0)
            };
            self.oscs[i].set_freq(
                self.base_freq * self.detune[i] * bend * osc_mul * self.uni_mul,
                self.sample_rate,
            );
        }
    }

    /// One control tick: evaluate LFO1/LFO2/Env3 + active ModMatrix slots.
    #[inline]
    fn control_tick(
        &mut self,
        bend_norm: f32,
        bend_semitones: f32,
        mod_wheel: f32,
        aftertouch: f32,
        active_slots: &[ModSlot],
    ) {
        self.env3_amp = self.env3.next_amplitude();
        let eff_depth = (self.lfo_depth + mod_wheel * 0.6).clamp(0.0, 1.0);
        self.lfo.set_depth(eff_depth);
        let lfo1 = self.lfo.next_value(); // -eff_depth..eff_depth
        let lfo2 = self.lfo2.next_value();

        let mut all_pitch_semi = if self.lfo_dest == LfoDestination::Pitch {
            lfo1 * 2.0
        } else {
            0.0
        };
        let mut osc_pitch_semi = [0.0f32; MAX_OSCS];
        let mut cutoff_oct = if self.lfo_dest == LfoDestination::FilterCutoff {
            lfo1 * 2.5
        } else {
            0.0
        };
        let mut reso_off = 0.0f32;
        let mut amp_mod = if self.lfo_dest == LfoDestination::Amplitude {
            lfo1 * 0.5
        } else {
            0.0
        };
        let mut lvl_off = [0.0f32; MAX_OSCS];
        let mut pan_off = [0.0f32; MAX_OSCS];
        let mut pw_off = [0.0f32; MAX_OSCS];
        let mut gains_dirty = false;

        if !active_slots.is_empty() {
            let key_track = ((self.note as f32 - 60.0) / 60.0).clamp(-1.0, 1.0);
            for s in active_slots {
                let src = match s.source {
                    ModSource::Off => 0.0,
                    ModSource::Lfo1 => lfo1,
                    ModSource::Lfo2 => lfo2,
                    ModSource::Lfo3 => lfo2,
                    ModSource::AmpEnv => self.last_amp,
                    ModSource::FilterEnv => self.fenv_amp,
                    ModSource::Env3 | ModSource::Env4 => self.env3_amp,
                    ModSource::Velocity => self.velocity,
                    ModSource::KeyTrack => key_track,
                    ModSource::ModWheel => mod_wheel,
                    ModSource::Aftertouch => aftertouch,
                    ModSource::PitchBend => bend_norm,
                    ModSource::Osc4 => 0.0,
                    ModSource::Constant => 1.0,
                };
                let v = src * s.amount;
                match s.dest {
                    ModDest::AllPitch => all_pitch_semi += v * 12.0,
                    ModDest::Osc1Pitch => osc_pitch_semi[0] += v * 12.0,
                    ModDest::Osc2Pitch => osc_pitch_semi[1] += v * 12.0,
                    ModDest::Osc3Pitch => osc_pitch_semi[2] += v * 12.0,
                    ModDest::FilterCutoff => cutoff_oct += v * 4.0,
                    ModDest::FilterResonance => reso_off += v,
                    ModDest::Amplitude => amp_mod += v,
                    ModDest::Osc1Level => {
                        lvl_off[0] += v;
                        gains_dirty = true;
                    }
                    ModDest::Osc2Level => {
                        lvl_off[1] += v;
                        gains_dirty = true;
                    }
                    ModDest::Osc3Level => {
                        lvl_off[2] += v;
                        gains_dirty = true;
                    }
                    ModDest::Osc1Pan => {
                        pan_off[0] += v;
                        gains_dirty = true;
                    }
                    ModDest::Osc2Pan => {
                        pan_off[1] += v;
                        gains_dirty = true;
                    }
                    ModDest::Osc3Pan => {
                        pan_off[2] += v;
                        gains_dirty = true;
                    }
                    ModDest::Osc1Pw => pw_off[0] += v,
                    ModDest::Osc2Pw => pw_off[1] += v,
                    ModDest::Osc3Pw => pw_off[2] += v,
                    ModDest::Lfo1Rate => {
                        self.lfo
                            .set_rate((self.lfo_rate * 2.0f32.powf(v * 3.0)).clamp(0.05, 30.0));
                    }
                    _ => {}
                }
            }
        }

        if gains_dirty {
            for i in 0..MAX_OSCS {
                let (pl, pr) = pan_gains(self.osc_pan[i] + pan_off[i]);
                let lvl = Self::osc_level_gain(self.oscs[i].kind, self.osc_level[i] + lvl_off[i]);
                self.target_gains[i] = (lvl * pl, lvl * pr);
            }
        }
        // Compute per-sample linear ramp increments across the next 64 samples
        // (with a 0.45 one-pole slew filter so even Square/Saw LFO edges glide
        // over ~3 ms instead of snapping, and Sine/Tri LFOs on Pan/Level never
        // produce 689 Hz control-rate staircase "shizzle").
        let mut pw_dirty = false;
        let mut detune_dirty = false;
        for i in 0..MAX_OSCS {
            let dl = 0.45 * (self.target_gains[i].0 - self.gains[i].0);
            let dr = 0.45 * (self.target_gains[i].1 - self.gains[i].1);
            self.gain_steps[i] = (dl * INV_CONTROL_DIV, dr * INV_CONTROL_DIV);

            // Smooth pulse-width and fine detune (cents/semitones) across
            // control ticks so slider drags and PWM never click.
            let target_pw = (self.osc_pw[i] + pw_off[i]).clamp(0.01, 0.99);
            self.smooth_pw[i] += 0.25 * (target_pw - self.smooth_pw[i]);
            if (target_pw - self.smooth_pw[i]).abs() > 1e-6 {
                pw_dirty = true;
            }
            self.detune[i] += 0.25 * (self.target_detune[i] - self.detune[i]);
            if (self.target_detune[i] - self.detune[i]).abs() > 1e-7 {
                detune_dirty = true;
            }
        }
        if pw_dirty {
            for i in 0..MAX_OSCS {
                self.oscs[i].set_duty(self.smooth_pw[i]);
            }
        }
        self.uni_mul += 0.25 * (self.target_uni_mul - self.uni_mul);
        let uni_dirty = (self.target_uni_mul - self.uni_mul).abs() > 1e-7;
        let udl = 0.45 * (self.target_uni_gain.0 - self.uni_gain.0);
        let udr = 0.45 * (self.target_uni_gain.1 - self.uni_gain.1);
        self.uni_gain_step = (udl * INV_CONTROL_DIV, udr * INV_CONTROL_DIV);

        // Osc4 audio-rate FM/RM depths ramp per sample like amp_lfo.
        self.osc4_fm_step = 0.45 * (self.target_osc4_fm - self.osc4_fm) * INV_CONTROL_DIV;
        self.osc4_rm_step = 0.45 * (self.target_osc4_rm - self.osc4_rm) * INV_CONTROL_DIV;

        // Smooth pitch offsets (bend + LFO/matrix pitch) at ~0.5 pole per
        // tick: pitch bend and vibrato stay clean while tracking stays fast.
        let target_pitch = bend_semitones + all_pitch_semi;
        self.smooth_pitch += 0.5 * (target_pitch - self.smooth_pitch);
        let mut pitch_dirty = (target_pitch - self.smooth_pitch).abs() > 1e-7;
        for i in 0..MAX_OSCS {
            self.smooth_osc_pitch[i] += 0.5 * (osc_pitch_semi[i] - self.smooth_osc_pitch[i]);
            if (osc_pitch_semi[i] - self.smooth_osc_pitch[i]).abs() > 1e-7 {
                pitch_dirty = true;
            }
        }
        // Only pay for the powf/divides in set_osc_freqs while a pitch input
        // is actually moving — sustained notes re-derive identical incs.
        if pitch_dirty || detune_dirty || uni_dirty {
            self.set_osc_freqs(self.smooth_pitch, self.smooth_osc_pitch);
        }
        // Bitmask of oscillators that may emit audio next block — one bit
        // test per osc per sample instead of re-reading all the gain fields.
        let mut live = 0u8;
        for i in 0..MAX_OSCS {
            if self.gains[i].0 > 1e-6
                || self.gains[i].1 > 1e-6
                || self.target_gains[i].0 > 0.0
                || self.target_gains[i].1 > 0.0
                || self.wf_pending[i] != 0xFF
            {
                live |= 1 << i;
            }
        }
        if self.osc4_fm != 0.0
            || self.osc4_rm != 0.0
            || self.target_osc4_fm != 0.0
            || self.target_osc4_rm != 0.0
        {
            live |= 1 << 3;
        }
        self.osc_live = live;

        // Smooth base_cutoff, cutoff modulation, and resonance at control rate
        // to prevent staircase clicks when sweeping filter sliders/knobs.
        // The filter envelope itself runs at audio rate upstream, so only the
        // control-rate offsets are smoothed — fenv attack stays crisp.
        self.smooth_cutoff += 0.22 * (self.base_cutoff - self.smooth_cutoff);
        self.smooth_fenv_amt += 0.25 * (self.fenv_amount - self.smooth_fenv_amt);
        let env_sweep = 6.0 * self.smooth_fenv_amt * self.fenv_amp;
        self.smooth_cutoff_oct += 0.35 * (cutoff_oct - self.smooth_cutoff_oct);
        let eff_reso = (self.resonance + reso_off).clamp(0.0, 1.0);
        self.smooth_reso += 0.22 * (eff_reso - self.smooth_reso);
        // Only pay for the powf + sin_cos + divides in filter.set while a
        // coefficient input is actually moving (or the mode changed) —
        // sustained notes re-derive identical coefficients otherwise.
        let oct_total = env_sweep + self.smooth_cutoff_oct;
        if (oct_total - self.last_oct).abs() > 1e-7
            || (self.smooth_reso - self.last_reso).abs() > 1e-7
            || self.filter.mode != self.filter_mode
        {
            self.last_oct = oct_total;
            self.last_reso = self.smooth_reso;
            let cutoff = (self.smooth_cutoff * 2.0f32.powf(oct_total))
                .clamp(25.0, self.sample_rate as f32 * 0.42);
            self.last_cutoff = cutoff;
            self.filter
                .set(self.filter_mode, cutoff, self.smooth_reso, self.sample_rate);
        }
        // Bypass the per-sample SVF when it would be a no-op audibly:
        // lowpass, cutoff pinned at the top clamp, ~no resonance — AND no
        // modulator currently able to pull it back down. apply_preset
        // auto-routes LFO1->FilterCutoff for the mod wheel even with depth 0,
        // so gate on the *effective* wheel depth (lfo_depth + CC1), not just
        // the destination. Slots routing to cutoff/reso with nonzero amount
        // and nonzero fenv stay live — no mid-note bypass flapping. On exit,
        // reset integrator state; at wide-open cutoff re-entry is clickless.
        let lfo_to_cutoff = self.lfo_dest == LfoDestination::FilterCutoff
            && (self.lfo_depth + mod_wheel * 0.6) > 0.01;
        let filter_modulated = self.fenv_amount.abs() > 0.01
            || lfo_to_cutoff
            || active_slots.iter().any(|s| {
                matches!(s.dest, ModDest::FilterCutoff | ModDest::FilterResonance)
                    && s.amount.abs() > 0.01
            });
        let bypass = self.filter_mode == FilterMode::LowPass
            && self.smooth_reso < 0.01
            && self.last_cutoff >= self.sample_rate as f32 * 0.42 * 0.999
            && self.filter.xfade == 0
            && !filter_modulated;
        if self.filter_bypass && !bypass {
            self.filter.reset();
        }
        self.filter_bypass = bypass;

        let target_amp_lfo = (1.0 + amp_mod).clamp(0.0, 2.0);
        self.amp_lfo_step = 0.45 * (target_amp_lfo - self.amp_lfo) * INV_CONTROL_DIV;

        // Per-sample fast-path flags: with all steps converged the ramp adds
        // are exact no-ops, so the hot loop can skip them and their write-back.
        self.ramping = self.gain_steps.iter().any(|s| s.0 != 0.0 || s.1 != 0.0)
            || self.uni_gain_step.0 != 0.0
            || self.uni_gain_step.1 != 0.0
            || self.osc4_fm_step != 0.0
            || self.osc4_rm_step != 0.0
            || self.amp_lfo_step != 0.0;
        self.wf_fading = self.wf_fade.iter().any(|f| *f < 1.0)
            || self.wf_pending.iter().any(|p| *p != 0xFF);
    }

    #[link_section = ".iram1.text"]
    fn next_stereo(
        &mut self,
        bend_norm: f32,
        bend_semitones: f32,
        mod_wheel: f32,
        aftertouch: f32,
        active_slots: &[ModSlot],
    ) -> StereoSample {
        if !self.active {
            return StereoSample::ZERO;
        }
        let amp = pstage!(profile::ENV, self.amp_env.next_amplitude());
        if self.amp_env.is_finished() {
            self.active = false;
            return StereoSample::ZERO;
        }
        self.last_amp = amp;
        self.fenv_amp = if self.fenv_amount != 0.0 || !active_slots.is_empty() {
            pstage!(profile::ENV, self.fenv.next_amplitude())
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
            pstage!(
                profile::CONTROL,
                self.control_tick(bend_norm, bend_semitones, mod_wheel, aftertouch, active_slots)
            );
        }
        self.control_phase = (self.control_phase + 1) % CONTROL_DIV;

        let mut l = 0.0f32;
        let mut r = 0.0f32;
        let _osct = profile::now();
        // Hoist per-sample ramp state into locals: the `&mut oscs[i]` borrows
        // in the loop otherwise force a load+store round-trip on every gain
        // field every sample. `osc_live` is a bitmask refreshed at each
        // control tick — one bit test per osc instead of a 5-way OR over
        // gain/pending fields. When `ramping` is false all steps are exact
        // zeros and the adds/write-back are skipped wholesale.
        let live = self.osc_live;
        let fading = self.wf_fading;
        let mut gs = self.gains;
        if self.ramping {
            let st = self.gain_steps;
            for i in 0..MAX_OSCS {
                gs[i].0 += st[i].0;
                gs[i].1 += st[i].1;
            }
            self.gains = gs;
            self.osc4_fm += self.osc4_fm_step;
            self.osc4_rm += self.osc4_rm_step;
            self.amp_lfo += self.amp_lfo_step;
            self.uni_gain.0 += self.uni_gain_step.0;
            self.uni_gain.1 += self.uni_gain_step.1;
        }
        // Advance Osc 4 first when active (either audible or routed as an
        // audio-rate FM / Ring-Mod source in the mod matrix).
        let s4 = if live & (1 << 3) != 0 {
            let mut v = self.oscs[3].next();
            if fading {
                v *= self.wf_fade[3];
                self.step_wf_fade(3);
            }
            l += v * gs[3].0;
            r += v * gs[3].1;
            if self.osc4_fm != 0.0 {
                let mut p = self.oscs[0].phase + v * self.osc4_fm * self.oscs[0].inc * 4.0;
                if p >= 1.0 {
                    p -= 1.0;
                } else if p < 0.0 {
                    p += 1.0;
                }
                self.oscs[0].phase = p;
            }
            v
        } else {
            0.0
        };
        let rm_gain = if self.osc4_rm != 0.0 {
            1.0 + s4 * self.osc4_rm
        } else {
            1.0
        };
        for i in 0..3 {
            if live & (1 << i) != 0 {
                let mut s = self.oscs[i].next();
                if fading {
                    s *= self.wf_fade[i];
                    self.step_wf_fade(i);
                }
                let m = if i == 0 { rm_gain } else { 1.0 };
                l += s * gs[i].0 * m;
                r += s * gs[i].1 * m;
            }
        }
        if profile::enabled() {
            profile::add(profile::OSC, profile::now() - _osct);
        }

        let _ft = profile::now();
        let (fl, fr) = if self.filter_bypass {
            (l, r)
        } else {
            (self.filter.process(0, l), self.filter.process(1, r))
        };
        if profile::enabled() {
            profile::add(profile::FILTER, profile::now() - _ft);
        }
        let g = amp * self.velocity * self.amp_lfo;
        StereoSample::new(fl * g * self.uni_gain.0, fr * g * self.uni_gain.1)
    }
}

pub struct EmbeddedEngine {
    voices: Vec<LeanVoice>,
    pub effects: EffectsChain,
    pub mod_matrix: ModMatrix,
    active_slots: [ModSlot; MOD_SLOTS],
    num_active_slots: usize,
    cc_mapping: CCMapping,
    master_volume: f32,
    /// Post-FX master gain after audio-taper + one-pole per-sample smoothing
    /// (maps master_volume 0..1 -> 0..1.5 so the top of the throw gives up to
    /// +3.5 dB of clean digital makeup gain before the DAC).
    smooth_master_gain: f32,
    mod_wheel: f32,       // 0..1 from CC1
    aftertouch: f32,      // 0..1
    pitch_bend: f32,      // -1..1
    bend_range: f32,      // semitones
    portamento: f32,
    polyphony: PolyphonyMode,
    /// Mono-mode key stack (last-note priority — newest on top), matching
    /// the desktop `VoiceManager::key_stack` semantics.
    key_stack: [(MidiNote, Amplitude); KEY_STACK_CAP],
    key_stack_len: usize,
    /// Mono unison stack: voices triggered per note (1 = classic mono).
    unison_count: usize,
    /// Max unison pitch deviation in cents at the stack edges.
    unison_detune: f32,
    /// Stereo spread 0..1 across unison voices.
    unison_spread: f32,
    age_counter: u64,
    stereo_width: f32,
    /// Per-sample smoothed stereo width — moving the Width slider/CC steps
    /// the mid/side gain instantly otherwise.
    smooth_width: f32,
    voice_cap: usize,     // effective polyphony cap (CPU governor / user limit)
    bpm: f32,             // tempo in BPM (20..300) for synced LFOs and delay
    lfo1_rate: f32,       // unsynced LFO1 rate in Hz
    lfo1_sync: bool,
    lfo1_beats: f32,      // LFO1 period in quarter-note beats when synced
    lfo2_rate: f32,       // unsynced LFO2 rate in Hz
    lfo2_sync: bool,
    lfo2_beats: f32,      // LFO2 period in quarter-note beats when synced
}

impl EmbeddedEngine {
    pub fn new(max_voices: usize, sample_rate: SampleRate) -> Self {
        Self {
            voices: (0..max_voices).map(|_| LeanVoice::new(sample_rate)).collect(),
            effects: EffectsChain::new(sample_rate),
            mod_matrix: ModMatrix::default(),
            active_slots: [ModSlot::default(); MOD_SLOTS],
            num_active_slots: 0,
            cc_mapping: CCMapping::default_mappings(),
            master_volume: 0.7,
            smooth_master_gain: 0.7 * (0.75 + 0.75 * 0.7),
            mod_wheel: 0.0,
            aftertouch: 0.0,
            pitch_bend: 0.0,
            bend_range: 12.0,
            portamento: 0.0,
            polyphony: PolyphonyMode::Poly,
            key_stack: [(0, 0.0); KEY_STACK_CAP],
            key_stack_len: 0,
            unison_count: 1,
            unison_detune: 15.0,
            unison_spread: 0.7,
            age_counter: 0,
            stereo_width: 1.0,
            smooth_width: 1.0,
            voice_cap: max_voices,
            bpm: 120.0,
            lfo1_rate: 4.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_rate: 3.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
        }
    }

    /// Switch the internal render rate (e.g. 44100 ↔ 22050 for a runtime
    /// A/B). Re-derives every rate-dependent coefficient in place: filter and
    /// osc increments recompute on the next control tick from `sample_rate`;
    /// LFOs, envelopes and the FX chain get explicit `set_sample_rate`. Voices
    /// keep sounding across the switch (envelope positions are preserved).
    /// NOTE: `Delay::set_sample_rate` reallocates its delay lines — call from
    /// a user-initiated toggle, not the per-block hot path.
    pub fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        let ctrl_sr = (sample_rate / CONTROL_DIV).max(1);
        for v in &mut self.voices {
            v.sample_rate = sample_rate;
            v.amp_env.set_sample_rate(sample_rate);
            v.fenv.set_sample_rate(sample_rate);
            v.env3.set_sample_rate(ctrl_sr);
            v.lfo.set_sample_rate(ctrl_sr);
            v.lfo2.set_sample_rate(ctrl_sr);
        }
        self.effects.set_sample_rate(sample_rate);
    }

    /// Effective polyphony limit — `pick_voice` only allocates from the
    /// first `cap` slots. Voices above the cap that are already sounding keep
    /// their tails; shed them explicitly with `shed_voice`.
    pub fn set_voice_cap(&mut self, n: usize) {
        self.voice_cap = n.clamp(1, self.voices.len());
    }

    pub fn voice_cap(&self) -> usize {
        self.voice_cap
    }

    /// Switch between Poly and Mono. Desktop `VoiceManager` semantics: no-op
    /// when unchanged; on a real switch the key stack is cleared and every
    /// voice is released (envelopes release, not killed).
    pub fn set_polyphony_mode(&mut self, mode: PolyphonyMode) {
        if self.polyphony == mode {
            return;
        }
        self.polyphony = mode;
        self.key_stack_len = 0;
        for v in &mut self.voices {
            v.force_release();
        }
    }

    pub fn polyphony_mode(&self) -> PolyphonyMode {
        self.polyphony
    }

    /// Mono-mode unison stack size (1..=voice count).
    pub fn set_unison_count(&mut self, n: usize) {
        self.unison_count = n.clamp(1, self.voices.len());
    }

    /// Unison pitch spread in cents (max deviation at the stack edges).
    pub fn set_unison_detune(&mut self, cents: f32) {
        self.unison_detune = cents.clamp(0.0, 50.0);
    }

    /// Unison stereo spread 0..1.
    pub fn set_unison_spread(&mut self, spread: f32) {
        self.unison_spread = spread.clamp(0.0, 1.0);
    }

    pub fn unison(&self) -> (usize, f32, f32) {
        (self.unison_count, self.unison_detune, self.unison_spread)
    }

    /// Trigger the mono voice stack: `unison` voice slots on the same note
    /// with symmetric detune/pan offsets and decorrelated start phases.
    /// Legato glide + envelope retrigger come from `LeanVoice::note_on`,
    /// matching the desktop mono behaviour.
    fn trigger_mono(&mut self, note: MidiNote, velocity: Amplitude) {
        let n = self
            .unison_count
            .clamp(1, self.voice_cap.min(self.voices.len()));
        let det = self.unison_detune;
        let spr = self.unison_spread;
        let portamento = self.portamento;
        let age = self.age_counter;
        // 1/sqrt(n) keeps a unison stack at roughly single-voice loudness.
        let comp = (n as f32).sqrt().recip();
        for (k, v) in self.voices[..n].iter_mut().enumerate() {
            let off = if n > 1 {
                (k as f32 / (n - 1) as f32) * 2.0 - 1.0
            } else {
                0.0
            };
            v.target_uni_mul = 2.0f32.powf(off * det / 1200.0);
            // Linear pan taper on an already-stereo voice (centre = unity).
            let pan = off * spr;
            v.target_uni_gain = (
                (1.0 - pan.max(0.0)) * comp,
                (1.0 + pan.min(0.0)) * comp,
            );
            let was = v.active;
            v.note_on(note, velocity, portamento, age);
            if !was {
                // Decorrelate start phases — all-unison voices at phase 0
                // sum as one loud voice instead of a detuned chorus.
                for (j, o) in v.oscs.iter_mut().enumerate() {
                    o.phase = ((k + 1) as f32 * 0.618_034 + j as f32 * 0.377).fract();
                }
            }
        }
    }

    /// Monophonic note-on with last-note-priority key stacking (port of the
    /// desktop `handle_note_on_mono`).
    fn note_on_mono(&mut self, note: MidiNote, velocity: Amplitude) {
        // Remove the note if already held, then push on top of the stack.
        self.stack_remove(note);
        if self.key_stack_len == KEY_STACK_CAP {
            self.key_stack.copy_within(1.., 0); // drop the oldest held note
            self.key_stack_len -= 1;
        }
        self.key_stack[self.key_stack_len] = (note, velocity);
        self.key_stack_len += 1;
        self.trigger_mono(note, velocity);
    }

    /// Monophonic note-off: pop the note; if it was the sounding one,
    /// fall back to the previous held note (legato retrigger) or release.
    fn note_off_mono(&mut self, note: MidiNote) {
        let was_current =
            self.key_stack_len > 0 && self.key_stack[self.key_stack_len - 1].0 == note;
        self.stack_remove(note);
        if !was_current {
            return;
        }
        if self.key_stack_len > 0 {
            let (note, vel) = self.key_stack[self.key_stack_len - 1];
            self.trigger_mono(note, vel);
        } else {
            for v in &mut self.voices {
                v.note_off(note);
            }
        }
    }

    fn stack_remove(&mut self, note: MidiNote) {
        let mut w = 0;
        for i in 0..self.key_stack_len {
            if self.key_stack[i].0 != note {
                self.key_stack[w] = self.key_stack[i];
                w += 1;
            }
        }
        self.key_stack_len = w;
    }

    /// Fast-fade the least-important active voice to reclaim render budget.
    /// Priority: the quietest meaningful release tail, then the quietest held
    /// non-bass voice *above the cap* (the actual excess), then the quietest
    /// held non-bass anywhere. Attack-state voices are never held-candidates —
    /// a fresh note reads as quiet mid-ramp but is the most audible loss.
    /// ~70 ms fade: fast relief, but reads as a release rather than a death.
    /// Returns the MIDI note shed, if any.
    pub fn shed_voice(&mut self) -> Option<u8> {
        let lowest = self.voices.iter().filter(|v| v.active).map(|v| v.note).min();
        // Held candidates only — Decay/Sustain. Attack-state voices read as
        // quiet mid-ramp but are the most audible loss, and Release voices
        // are already dying (re-picking wastes the streak without freeing
        // another voice).
        let quietest = |vs: &[LeanVoice]| {
            vs.iter()
                .enumerate()
                .filter(|(_, v)| {
                    v.active
                        && Some(v.note) != lowest
                        && matches!(
                            v.amp_env.state(),
                            EnvelopeState::Decay | EnvelopeState::Sustain
                        )
                })
                .min_by(|(_, a), (_, b)| {
                    a.amp_env
                        .level()
                        .partial_cmp(&b.amp_env.level())
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.age.cmp(&b.age))
                })
                .map(|(i, _)| i)
        };
        // Prefer the quietest *meaningful* release tail — skip tails already
        // fading within ~100 ms (naturally short releases and voices this
        // governor just shed at the 70 ms fade). Re-picking a fading voice is
        // worse than wasted: release() recomputes its decrement from the
        // current amplitude, so each re-shed restarts a full 70 ms fade — the
        // voice asymptotically approaches zero without ever deactivating.
        let idx = self
            .voices
            .iter()
            .enumerate()
            .filter(|(_, v)| {
                v.active
                    && v.amp_env.state() == EnvelopeState::Release
                    && v.amp_env.release_time() > 0.1
            })
            .min_by(|(_, a), (_, b)| {
                a.amp_env
                    .level()
                    .partial_cmp(&b.amp_env.level())
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.age.cmp(&b.age))
            })
            .map(|(i, _)| i)
            .or_else(|| quietest(&self.voices[self.voice_cap..]).map(|i| self.voice_cap + i))
            .or_else(|| quietest(&self.voices))
            .or_else(|| self.voices.iter().position(|v| v.active))?;
        let note = self.voices[idx].note;
        // ~70 ms fade — CPU relief within a few blocks, perceived as a quick
        // release rather than a note vanishing. Apply it to held notes AND
        // long natural tails (a 2 s release tail still burns CPU for 2 s —
        // accelerate it). Voices picked here either aren't in Release or have
        // release_time > 0.1, so re-releasing never extends a fast fade —
        // the zombie-fade bug came from re-arming an already-short release.
        self.voices[idx].amp_env.set_release(0.07);
        self.voices[idx].amp_env.release();
        self.voices[idx].fenv.release();
        self.voices[idx].env3.release();
        Some(note)
    }

    fn repack_mod_slots(&mut self) {
        let mut n = 0;
        let mut fm = 0.0f32;
        let mut rm = 0.0f32;
        for slot in &self.mod_matrix.slots {
            if slot.enabled && slot.source != ModSource::Off && slot.dest != ModDest::Off && slot.amount != 0.0 {
                if slot.source == ModSource::Osc4 {
                    match slot.dest {
                        ModDest::AllPitch | ModDest::Osc1Pitch | ModDest::Osc2Pitch => {
                            fm += slot.amount;
                        }
                        ModDest::Osc1Level | ModDest::Osc2Level | ModDest::Amplitude => {
                            rm += slot.amount;
                        }
                        _ => {}
                    }
                } else {
                    self.active_slots[n] = *slot;
                    n += 1;
                }
            }
        }
        self.num_active_slots = n;
        for v in &mut self.voices {
            v.target_osc4_fm = fm;
            v.target_osc4_rm = rm;
            if !v.active {
                v.osc4_fm = fm;
                v.osc4_rm = rm;
            }
            v.update_gains();
        }
    }

    pub fn set_mod_slot(&mut self, idx: usize, slot: ModSlot) {
        if idx < MOD_SLOTS {
            self.mod_matrix.slots[idx] = slot;
            self.repack_mod_slots();
        }
    }

    #[inline]
    fn effective_lfo1_rate(&self) -> f32 {
        if self.lfo1_sync {
            (self.bpm / 60.0) / self.lfo1_beats.max(0.01)
        } else {
            self.lfo1_rate
        }
        .clamp(0.05, 20.0)
    }

    #[inline]
    fn effective_lfo2_rate(&self) -> f32 {
        if self.lfo2_sync {
            (self.bpm / 60.0) / self.lfo2_beats.max(0.01)
        } else {
            self.lfo2_rate
        }
        .clamp(0.05, 20.0)
    }

    fn update_lfo_rates(&mut self) {
        let r1 = self.effective_lfo1_rate();
        let r2 = self.effective_lfo2_rate();
        for v in &mut self.voices {
            v.lfo_rate = r1;
            v.lfo.set_rate(r1);
            v.lfo2.set_rate(r2);
        }
    }

    /// Set tempo in BPM (20..300) and recompute tempo-synced LFOs + Delay.
    pub fn set_bpm(&mut self, bpm: f32) {
        self.bpm = bpm.clamp(20.0, 300.0);
        self.effects.set_bpm(self.bpm);
        self.update_lfo_rates();
    }

    pub fn bpm(&self) -> f32 {
        self.bpm
    }

    pub fn set_lfo1_sync(&mut self, sync: bool, beats: f32) {
        self.lfo1_sync = sync;
        self.lfo1_beats = beats.clamp(0.05, 16.0);
        self.update_lfo_rates();
    }

    pub fn set_lfo1_sync_enabled(&mut self, sync: bool) {
        self.lfo1_sync = sync;
        self.update_lfo_rates();
    }

    pub fn set_lfo1_beats(&mut self, beats: f32) {
        self.lfo1_beats = beats.clamp(0.05, 16.0);
        self.update_lfo_rates();
    }

    pub fn lfo1_sync(&self) -> (bool, f32) {
        (self.lfo1_sync, self.lfo1_beats)
    }

    pub fn set_lfo2_sync(&mut self, sync: bool, beats: f32) {
        self.lfo2_sync = sync;
        self.lfo2_beats = beats.clamp(0.05, 16.0);
        self.update_lfo_rates();
    }

    pub fn set_lfo2_sync_enabled(&mut self, sync: bool) {
        self.lfo2_sync = sync;
        self.update_lfo_rates();
    }

    pub fn set_lfo2_beats(&mut self, beats: f32) {
        self.lfo2_beats = beats.clamp(0.05, 16.0);
        self.update_lfo_rates();
    }

    pub fn lfo2_sync(&self) -> (bool, f32) {
        (self.lfo2_sync, self.lfo2_beats)
    }

    pub fn set_delay_sync(&mut self, sync: bool) {
        let beats = self.effects.delay_sync_beats();
        self.effects.set_delay_tempo_sync(sync, beats);
    }

    pub fn set_delay_beats(&mut self, beats: f32) {
        let sync = self.effects.delay_tempo_sync();
        self.effects.set_delay_tempo_sync(sync, beats);
    }

    pub fn set_lfo2_rate(&mut self, rate: f32) {
        self.lfo2_rate = rate.clamp(0.05, 30.0);
        self.update_lfo_rates();
    }

    pub fn set_lfo2_depth(&mut self, depth: f32) {
        let d = depth.clamp(0.0, 1.0);
        for v in &mut self.voices {
            v.lfo2.set_depth(d);
        }
    }

    pub fn set_lfo2_waveform(&mut self, wf: LfoWaveform) {
        for v in &mut self.voices {
            v.lfo2.set_waveform(wf);
        }
    }

    pub fn set_env3_attack(&mut self, a: f32) {
        for v in &mut self.voices {
            v.env3.set_attack(a);
        }
    }

    pub fn set_env3_decay(&mut self, d: f32) {
        for v in &mut self.voices {
            v.env3.set_decay(d);
        }
    }

    pub fn set_env3_sustain(&mut self, s: f32) {
        for v in &mut self.voices {
            v.env3.set_sustain(s);
        }
    }

    pub fn set_env3_release(&mut self, r: f32) {
        for v in &mut self.voices {
            v.env3.set_release(r);
        }
    }

    #[inline]
    fn master_taper(v: f32) -> f32 {
        let v = v.clamp(0.0, 1.0);
        v * (0.75 + 0.75 * v)
    }

    pub fn set_master_volume(&mut self, v: Amplitude) {
        self.master_volume = v.clamp(0.0, 1.0);
        if self.active_voice_count() == 0 {
            self.smooth_master_gain = Self::master_taper(self.master_volume);
        }
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
        self.set_polyphony_mode(preset.polyphony_mode);
        self.set_unison_count(preset.unison_voices as usize);
        self.set_unison_detune(preset.unison_detune);
        self.set_unison_spread(preset.unison_spread);
        self.bend_range = preset.pitch_bend_range as f32;
        self.stereo_width = preset.stereo_width;
        self.set_master_volume(preset.master_volume);
        self.mod_matrix = preset.mod_matrix;
        let reso = preset.filter_resonance.clamp(0.0, 1.0);

        let mut lfo_dest = preset.lfo_destination;
        // If neither legacy LFO dest nor mod matrix is configured, route LFO1 ->
        // FilterCutoff at 0 base depth so the mod wheel (CC1) is always active.
        let any_matrix = preset.mod_matrix.slots.iter().any(|s| s.enabled);
        if lfo_dest == LfoDestination::Off && preset.lfo_depth == 0.0 && !any_matrix {
            lfo_dest = LfoDestination::FilterCutoff;
        }

        self.bpm = preset.tempo_bpm.clamp(20.0, 300.0);
        self.lfo1_rate = preset.lfo_rate;
        self.lfo1_sync = preset.lfo1_sync;
        self.lfo1_beats = preset.lfo1_beats.clamp(0.05, 16.0);
        self.lfo2_rate = preset.lfo2_rate.max(0.05);
        self.lfo2_sync = preset.lfo2_sync;
        self.lfo2_beats = preset.lfo2_beats.clamp(0.05, 16.0);
        let eff_lfo1_rate = self.effective_lfo1_rate();
        let eff_lfo2_rate = self.effective_lfo2_rate();

        for v in &mut self.voices {
            v.set_osc_waveform(0, preset.osc1_waveform);
            v.set_osc_waveform(1, preset.osc2_waveform);
            v.set_osc_waveform(2, preset.osc3_waveform);
            v.set_osc_waveform(3, preset.osc4_waveform);
            v.osc_pw = [preset.osc1_pw, preset.osc2_pw, preset.osc3_pw, preset.osc4_pw];
            v.osc_level = [
                preset.osc1_level,
                preset.osc2_level,
                preset.osc3_level,
                preset.osc4_level,
            ];
            v.osc_pan = [
                preset.osc1_pan,
                preset.osc2_pan,
                preset.osc3_pan,
                preset.osc4_pan,
            ];
            v.osc_semi = [
                0,
                preset.osc2_semitones,
                preset.osc3_semitones,
                preset.osc4_semitones,
            ];
            v.osc_cents = [0, preset.osc2_cents, preset.osc3_cents, preset.osc4_cents];
            v.target_detune = [
                1.0,
                super::oscillator::detune_ratio(preset.osc2_semitones, preset.osc2_cents),
                super::oscillator::detune_ratio(preset.osc3_semitones, preset.osc3_cents),
                super::oscillator::detune_ratio(preset.osc4_semitones, preset.osc4_cents),
            ];
            v.update_gains();
            // Update ADSR parameters in place so active voices keep sounding
            // through preset changes instead of dropping to Idle.
            v.amp_env.set_attack(preset.amp_attack);
            v.amp_env.set_decay(preset.amp_decay);
            v.amp_env.set_sustain(preset.amp_sustain);
            v.amp_env.set_release(preset.amp_release);
            v.cfgd_release = preset.amp_release;
            v.fenv.set_attack(preset.filter_attack);
            v.fenv.set_decay(preset.filter_decay);
            v.fenv.set_sustain(preset.filter_sustain);
            v.fenv.set_release(preset.filter_release);
            v.env3.set_attack(preset.env3_attack);
            v.env3.set_decay(preset.env3_decay);
            v.env3.set_sustain(preset.env3_sustain);
            v.env3.set_release(preset.env3_release);
            v.fenv_amount = preset.filter_env_amount;
            v.base_cutoff = preset.filter_cutoff;
            v.resonance = reso;
            if !v.active {
                v.smooth_cutoff = preset.filter_cutoff;
                v.smooth_reso = reso;
                v.smooth_pw = v.osc_pw;
                v.detune = v.target_detune;
                for i in 0..MAX_OSCS {
                    v.oscs[i].set_duty(v.smooth_pw[i]);
                }
            }
            v.lfo_rate = eff_lfo1_rate;
            v.lfo.set_rate(eff_lfo1_rate);
            v.lfo_depth = preset.lfo_depth;
            v.lfo.set_depth(preset.lfo_depth);
            v.lfo.set_waveform(preset.lfo_waveform);
            v.lfo_dest = lfo_dest;
            v.lfo2.set_rate(eff_lfo2_rate);
            v.lfo2.set_depth(preset.lfo2_depth.clamp(0.0, 1.0));
            v.lfo2.set_waveform(preset.lfo2_waveform);
        }
        self.repack_mod_slots();

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
                    v.cfgd_release = plain;
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
                self.lfo1_rate = plain;
                self.update_lfo_rates();
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
                    v.set_osc_waveform(0, wf);
                    v.update_gains();
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
                    v.osc_pw[0] = plain;
                    if !v.active {
                        v.smooth_pw[0] = plain;
                        v.oscs[0].set_duty(plain);
                    }
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
                    v.set_osc_waveform(1, wf);
                    v.update_gains();
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
                    v.osc_semi[1] = semi;
                    v.target_detune[1] =
                        super::oscillator::detune_ratio(v.osc_semi[1], v.osc_cents[1]);
                    if !v.active {
                        v.detune[1] = v.target_detune[1];
                    }
                }
            }
            SynthParam::Osc2Cents => {
                let cents = plain.round() as i8;
                for v in &mut self.voices {
                    v.osc_cents[1] = cents;
                    v.target_detune[1] =
                        super::oscillator::detune_ratio(v.osc_semi[1], v.osc_cents[1]);
                    if !v.active {
                        v.detune[1] = v.target_detune[1];
                    }
                }
            }
            SynthParam::Osc2Pw => {
                for v in &mut self.voices {
                    v.osc_pw[1] = plain;
                    if !v.active {
                        v.smooth_pw[1] = plain;
                        v.oscs[1].set_duty(plain);
                    }
                }
            }
            SynthParam::Osc2Pan => {
                for v in &mut self.voices {
                    v.osc_pan[1] = plain.clamp(-1.0, 1.0);
                    v.update_gains();
                }
            }
            SynthParam::Osc3Waveform => {
                let wf = WaveformType::from_index(plain as u8);
                for v in &mut self.voices {
                    v.set_osc_waveform(2, wf);
                    v.update_gains();
                }
            }
            SynthParam::Osc3Level => {
                let lvl = plain.clamp(0.0, 1.0);
                for v in &mut self.voices {
                    v.osc_level[2] = lvl;
                    v.update_gains();
                }
            }
            SynthParam::Osc3Semitones => {
                let semi = plain.round() as i8;
                for v in &mut self.voices {
                    v.osc_semi[2] = semi;
                    v.target_detune[2] =
                        super::oscillator::detune_ratio(v.osc_semi[2], v.osc_cents[2]);
                    if !v.active {
                        v.detune[2] = v.target_detune[2];
                    }
                }
            }
            SynthParam::Osc3Cents => {
                let cents = plain.round() as i8;
                for v in &mut self.voices {
                    v.osc_cents[2] = cents;
                    v.target_detune[2] =
                        super::oscillator::detune_ratio(v.osc_semi[2], v.osc_cents[2]);
                    if !v.active {
                        v.detune[2] = v.target_detune[2];
                    }
                }
            }
            SynthParam::Osc3Pw => {
                for v in &mut self.voices {
                    v.osc_pw[2] = plain;
                    if !v.active {
                        v.smooth_pw[2] = plain;
                        v.oscs[2].set_duty(plain);
                    }
                }
            }
            SynthParam::Osc3Pan => {
                for v in &mut self.voices {
                    v.osc_pan[2] = plain.clamp(-1.0, 1.0);
                    v.update_gains();
                }
            }
            SynthParam::Osc4Waveform => {
                let wf = WaveformType::from_index(plain as u8);
                for v in &mut self.voices {
                    v.set_osc_waveform(3, wf);
                    v.update_gains();
                }
            }
            SynthParam::Osc4Level => {
                let lvl = plain.clamp(0.0, 1.0);
                for v in &mut self.voices {
                    v.osc_level[3] = lvl;
                    v.update_gains();
                }
            }
            SynthParam::Osc4Semitones => {
                let semi = plain.round() as i8;
                for v in &mut self.voices {
                    v.osc_semi[3] = semi;
                    v.target_detune[3] =
                        super::oscillator::detune_ratio(v.osc_semi[3], v.osc_cents[3]);
                    if !v.active {
                        v.detune[3] = v.target_detune[3];
                    }
                }
            }
            SynthParam::Osc4Cents => {
                let cents = plain.round() as i8;
                for v in &mut self.voices {
                    v.osc_cents[3] = cents;
                    v.target_detune[3] =
                        super::oscillator::detune_ratio(v.osc_semi[3], v.osc_cents[3]);
                    if !v.active {
                        v.detune[3] = v.target_detune[3];
                    }
                }
            }
            SynthParam::Osc4Pw => {
                for v in &mut self.voices {
                    v.osc_pw[3] = plain;
                    if !v.active {
                        v.smooth_pw[3] = plain;
                        v.oscs[3].set_duty(plain);
                    }
                }
            }
            SynthParam::Osc4Pan => {
                for v in &mut self.voices {
                    v.osc_pan[3] = plain.clamp(-1.0, 1.0);
                    v.update_gains();
                }
            }
            SynthParam::StereoWidth => {
                self.stereo_width = plain.clamp(0.0, 2.0);
            }
            SynthParam::MasterVolume => {
                self.set_master_volume(plain);
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
                match self.polyphony {
                    PolyphonyMode::Mono => self.note_on_mono(event.note, event.velocity),
                    PolyphonyMode::Poly => {
                        let idx = self.pick_voice(event.note);
                        let v = &mut self.voices[idx];
                        v.target_uni_mul = 1.0;
                        v.target_uni_gain = (1.0, 1.0);
                        v.note_on(event.note, event.velocity, self.portamento, self.age_counter);
                    }
                }
            }
            SynthEventKind::NoteOff => match self.polyphony {
                PolyphonyMode::Mono => self.note_off_mono(event.note),
                PolyphonyMode::Poly => {
                    for v in &mut self.voices {
                        v.note_off(event.note);
                    }
                }
            },
            SynthEventKind::PitchBend(v) => self.pitch_bend = v,
            SynthEventKind::Aftertouch(v) => self.aftertouch = v,
            SynthEventKind::ControlChange { cc, value } => {
                let norm = value as f32 / 127.0;
                match cc {
                    1 => self.mod_wheel = norm,
                    7 => self.set_master_volume(norm),
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
        let pool = &self.voices[..self.voice_cap];
        // 1. Retrigger: if a voice is already playing this exact MIDI note
        //    (whether held or in its release tail), reuse it so hammering notes
        //    never exhausts the polyphony pool with duplicate tails.
        if let Some(i) = pool.iter().position(|v| v.active && v.note == note) {
            return i;
        }
        // 2. Idle voice: pick the first completely free slot.
        if let Some(i) = pool.iter().position(|v| !v.active) {
            return i;
        }
        // 3. Releasing voice: steal the oldest releasing voice so held notes
        //    (Attack/Decay/Sustain) are never killed by release tails.
        if let Some((i, _)) = pool
            .iter()
            .enumerate()
            .filter(|(_, v)| v.amp_env.state() == EnvelopeState::Release)
            .min_by_key(|(_, v)| v.age)
        {
            return i;
        }
        // 4. Over-cap voices are already marked for death — steal one before
        //    touching a held in-pool voice. Prefer releasing tails, then the
        //    oldest held non-bass note in the excess slots. Stealing an excess
        //    slot keeps the active count flat (the replaced voice dies with
        //    it) and the fresh age protects the new note from the next shed.
        //    An idle excess slot is NOT taken: a note there would be the only
        //    held excess voice and thus the governor's first shed candidate.
        if self.voice_cap < self.voices.len() {
            let excess = &self.voices[self.voice_cap..];
            let lowest = self.voices.iter().filter(|v| v.active).map(|v| v.note).min();
            if let Some((i, _)) = excess
                .iter()
                .enumerate()
                .filter(|(_, v)| v.active && v.amp_env.state() == EnvelopeState::Release)
                .min_by_key(|(_, v)| v.age)
                .or_else(|| {
                    excess
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| v.active && Some(v.note) != lowest)
                        .min_by_key(|(_, v)| v.age)
                })
            {
                return self.voice_cap + i;
            }
        }
        // 5. All in-pool voices are physically held: protect the lowest held
        //    bass note and steal the oldest among the remaining held voices.
        let lowest_note = pool.iter().map(|v| v.note).min().unwrap_or(0);
        pool.iter()
            .enumerate()
            .filter(|(_, v)| v.note != lowest_note)
            .min_by_key(|(_, v)| v.age)
            .or_else(|| pool.iter().enumerate().min_by_key(|(_, v)| v.age))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[link_section = ".iram1.text"]
    pub fn next_sample_stereo(&mut self) -> StereoSample {
        let bend_norm = self.pitch_bend;
        let bend_semi = bend_norm * self.bend_range;
        let mod_wheel = self.mod_wheel;
        let aftertouch = self.aftertouch;
        let active_slots = &self.active_slots[..self.num_active_slots];
        let mut l = 0.0f32;
        let mut r = 0.0f32;
        for v in &mut self.voices {
            if v.active {
                let s = v.next_stereo(bend_norm, bend_semi, mod_wheel, aftertouch, active_slots);
                l += s.left;
                r += s.right;
            }
        }
        let mut out = StereoSample::new(l, r);
        // Per-sample one-pole on stereo width — a stepped mid/side gain is an
        // output discontinuity just like any other level jump. Snaps on
        // converge so the compare-guard skips the update in steady state.
        if self.smooth_width != self.stereo_width {
            self.smooth_width += 0.002 * (self.stereo_width - self.smooth_width);
            if (self.stereo_width - self.smooth_width).abs() < 1e-5 {
                self.smooth_width = self.stereo_width;
            }
        }
        if (self.smooth_width - 1.0).abs() > 1e-4 {
            out = out.with_width(self.smooth_width);
        }
        // Per-sample one-pole de-zipper on Master Volume (~15 ms time constant)
        // with an audio-taper makeup curve: 0.0 -> 0.0, 0.5 -> 0.56, 0.7 -> 0.89,
        // 1.0 -> 1.50 (+3.5 dB post-FX digital makeup gain before the DAC).
        let target_vol = Self::master_taper(self.master_volume);
        if self.smooth_master_gain != target_vol {
            self.smooth_master_gain += 0.0015 * (target_vol - self.smooth_master_gain);
            if (target_vol - self.smooth_master_gain).abs() < 1e-5 {
                self.smooth_master_gain = target_vol;
            }
        }

        if profile::enabled() {
            profile::add(profile::SAMPLES, 1);
            let _ft = profile::now();
            out = self.effects.process_stereo(out) * self.smooth_master_gain;
            profile::add(profile::FX, profile::now() - _ft);
        } else {
            out = self.effects.process_stereo(out) * self.smooth_master_gain;
        }

        StereoSample::new(soft_clip(out.left), soft_clip(out.right))
    }
}

#[inline]
pub fn soft_clip(x: f32) -> f32 {
    if x.abs() <= 0.90 {
        x
    } else {
        // C1-smooth safety shoulder above 0.90 into [-1.0, 1.0]: completely
        // linear up to 0.90 so normal polyphonic playing is untouched.
        let excess = x.abs() - 0.90;
        x.signum() * (0.90 + 0.10 * (excess / (0.10 + excess)))
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
        assert!(peak > 0.02, "expected audible output, peak={peak}");
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
        assert!(peak > 0.03 && peak <= 1.0, "unexpected peak={peak}");
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

    #[test]
    fn tempo_sync_updates_lfos_and_delay_across_bpm_and_presets() {
        let mut e = engine();
        // Unsynced baseline
        e.set_param(SynthParam::LfoRate, 4.5);
        e.set_lfo2_rate(3.25);
        assert!((e.voices[0].lfo.rate() - 4.5).abs() < 1e-4);
        assert!((e.voices[0].lfo2.rate() - 3.25).abs() < 1e-4);

        // Sync LFO1 to 1/8 note (0.5 beats) and LFO2 to 1 bar (4.0 beats) at 120 BPM (2 beats/sec)
        e.set_bpm(120.0);
        e.set_lfo1_sync(true, 0.5);
        e.set_lfo2_sync(true, 4.0);
        e.effects.set_delay_tempo_sync(true, 0.75); // dotted eighth = 0.375 s @ 120 BPM
        assert!((e.voices[0].lfo.rate() - 4.0).abs() < 1e-4);
        assert!((e.voices[0].lfo2.rate() - 0.5).abs() < 1e-4);
        assert!((e.effects.delay_time() - 0.375).abs() < 1e-4);

        // Changing BPM rescales LFO1, LFO2, and Delay simultaneously
        e.set_bpm(150.0); // 2.5 beats/sec
        assert!((e.voices[0].lfo.rate() - 5.0).abs() < 1e-4);
        assert!((e.voices[0].lfo2.rate() - 0.625).abs() < 1e-4);
        assert!((e.effects.delay_time() - 0.30).abs() < 1e-4);

        // Disabling sync restores the stored unsynced Hz rates
        e.set_lfo1_sync_enabled(false);
        e.set_lfo2_sync_enabled(false);
        assert!((e.voices[0].lfo.rate() - 4.5).abs() < 1e-4);
        assert!((e.voices[0].lfo2.rate() - 3.25).abs() < 1e-4);

        // Preset with tempo sync applies synced rates directly
        let mut p = Preset::default();
        p.tempo_bpm = 140.0;
        p.lfo1_sync = true;
        p.lfo1_beats = 0.25; // 1/16 -> (140/60)/0.25 = 9.3333 Hz
        p.lfo2_sync = true;
        p.lfo2_beats = 2.0; // 1/2 -> (140/60)/2.0 = 1.1667 Hz
        p.delay_sync = true;
        p.delay_beats = 0.5; // 1/8 -> 0.5 * 60 / 140 = 0.2142857 s
        e.apply_preset(&p);
        assert!((e.bpm() - 140.0).abs() < 1e-4);
        assert!((e.voices[0].lfo.rate() - (140.0 / 60.0 / 0.25)).abs() < 1e-3);
        assert!((e.voices[0].lfo2.rate() - (140.0 / 60.0 / 2.0)).abs() < 1e-3);
        assert!((e.effects.delay_time() - (0.5 * 60.0 / 140.0)).abs() < 1e-4);
    }

    #[test]
    fn mono_key_stack_returns_to_previous_note() {
        let mut e = engine();
        e.set_polyphony_mode(PolyphonyMode::Mono);
        e.receive_event(NoteEvent::note_on_midi(60, 100));
        for _ in 0..64 {
            e.next_sample_stereo();
        }
        e.receive_event(NoteEvent::note_on_midi(64, 100));
        for _ in 0..64 {
            e.next_sample_stereo();
        }
        assert!(e.voices[0].active && e.voices[0].note == 64);
        // Releasing the top note falls back to the still-held 60.
        e.receive_event(NoteEvent::note_off(64));
        for _ in 0..64 {
            e.next_sample_stereo();
        }
        assert!(e.voices[0].active);
        assert_eq!(e.voices[0].note, 60);
        assert_ne!(e.voices[0].amp_env.state(), EnvelopeState::Release);
        // Releasing a non-top held note changes nothing audible.
        e.receive_event(NoteEvent::note_on_midi(67, 100));
        e.receive_event(NoteEvent::note_off(60));
        for _ in 0..64 {
            e.next_sample_stereo();
        }
        assert!(e.voices[0].active && e.voices[0].note == 67);
        // Releasing the last note releases the voice.
        e.receive_event(NoteEvent::note_off(67));
        for _ in 0..44100 {
            e.next_sample_stereo();
        }
        assert_eq!(e.active_voice_count(), 0);
    }

    #[test]
    fn mono_unison_stacks_detuned_voices() {
        let mut e = engine();
        e.set_polyphony_mode(PolyphonyMode::Mono);
        e.set_unison_count(3);
        e.set_unison_detune(24.0);
        e.set_unison_spread(1.0);
        e.receive_event(NoteEvent::note_on_midi(60, 100));
        assert_eq!(e.active_voice_count(), 3);
        assert!(e.voices[..3].iter().all(|v| v.note == 60));
        // Symmetric detune: -24, 0, +24 cents.
        let edge = 2.0f32.powf(24.0 / 1200.0);
        assert!((e.voices[0].uni_mul - 1.0 / edge).abs() < 1e-4);
        assert!((e.voices[1].uni_mul - 1.0).abs() < 1e-4);
        assert!((e.voices[2].uni_mul - edge).abs() < 1e-4);
        // Hard spread puts the edge voices fully L/R (with 1/sqrt(3) comp).
        let comp = 3.0f32.sqrt().recip();
        assert!((e.voices[0].uni_gain.1).abs() < 1e-6);
        assert!((e.voices[0].uni_gain.0 - comp).abs() < 1e-4);
        assert!((e.voices[2].uni_gain.0).abs() < 1e-6);
        // Stack still returns to the previous held note across all voices.
        e.receive_event(NoteEvent::note_on_midi(64, 90));
        e.receive_event(NoteEvent::note_off(64));
        for _ in 0..64 {
            e.next_sample_stereo();
        }
        assert!(e.voices[..3].iter().all(|v| v.note == 60 && v.active));
        e.receive_event(NoteEvent::note_off(60));
        for _ in 0..44100 {
            e.next_sample_stereo();
        }
        assert_eq!(e.active_voice_count(), 0);
    }

    #[test]
    fn unison_ignored_in_poly_and_mode_switch_releases() {
        let mut e = engine(); // Poly
        e.set_unison_count(4);
        e.receive_event(NoteEvent::note_on_midi(60, 100));
        assert_eq!(e.active_voice_count(), 1);
        // Switching to mono releases held poly voices.
        e.set_polyphony_mode(PolyphonyMode::Mono);
        for _ in 0..44100 {
            e.next_sample_stereo();
        }
        assert_eq!(e.active_voice_count(), 0);
        // Idempotent: no release when the mode doesn't change.
        e.receive_event(NoteEvent::note_on_midi(62, 100));
        e.set_polyphony_mode(PolyphonyMode::Mono);
        for _ in 0..64 {
            e.next_sample_stereo();
        }
        assert_eq!(e.active_voice_count(), e.unison().0);
    }

    #[test]
    fn static_gain_staging_preserves_chord_dynamics_without_clipping() {
        // Drawbar-style Organ patch (3 sine oscs at 0.9 / 0.7 / 0.5) at full
        // MasterVolume = 1.0 and max velocity 127.
        let measure_chord_peak = |notes: &[u8]| -> f32 {
            let mut e = engine();
            e.set_param(SynthParam::MasterVolume, 1.0);
            for p in [
                SynthParam::Osc1Waveform,
                SynthParam::Osc2Waveform,
                SynthParam::Osc3Waveform,
            ] {
                e.set_param(p, 0.0); // Sine
            }
            e.set_param(SynthParam::Osc1Level, 0.9);
            e.set_param(SynthParam::Osc2Level, 0.7);
            e.set_param(SynthParam::Osc2Semitones, 12.0);
            e.set_param(SynthParam::Osc3Level, 0.5);
            e.set_param(SynthParam::Osc3Semitones, 19.0);
            for &n in notes {
                e.receive_event(NoteEvent::note_on_midi(n, 127));
            }
            let mut peak = 0.0f32;
            for _ in 0..8820 {
                let s = e.next_sample_stereo();
                assert!(
                    s.left.is_finite()
                        && s.right.is_finite()
                        && s.left.abs() <= 1.0
                        && s.right.abs() <= 1.0,
                    "output out of bounds: {s:?}"
                );
                peak = peak.max(s.left.abs()).max(s.right.abs());
            }
            peak
        };

        let p1 = measure_chord_peak(&[60]);
        let p2 = measure_chord_peak(&[60, 64]);
        let p4 = measure_chord_peak(&[60, 64, 67, 72]);

        // Chords must scale up naturally (no auto-dimming) while keeping
        // 4-note full-velocity chords inside the linear range (<= 0.90).
        assert!(p2 > p1 * 1.5, "2 notes ({p2}) should be clearly louder than 1 note ({p1})");
        assert!(p4 > p2 * 1.4, "4 notes ({p4}) should be clearly louder than 2 notes ({p2})");
        assert!(p4 <= 0.90, "4-note organ chord ({p4}) should stay in the linear region <= 0.90");
    }

    #[test]
    fn low_freq_high_resonance_stays_controlled_and_volume_dezippers() {
        let mut e = engine();
        // Low bass notes (C1=24, G1=31, C2=36) with cutoff tuned near the
        // fundamental (110 Hz), maximum resonance (1.0), and full volume (1.0).
        e.set_param(SynthParam::MasterVolume, 1.0);
        e.set_param(SynthParam::Osc1Level, 1.0);
        e.set_param(SynthParam::Osc2Level, 0.8);
        e.set_param(SynthParam::FilterCutoff, 110.0);
        e.set_param(SynthParam::FilterResonance, 1.0);
        for &n in &[24u8, 31, 36] {
            e.receive_event(NoteEvent::note_on_midi(n, 127));
        }
        let mut peak = 0.0f32;
        for _ in 0..8820 {
            let s = e.next_sample_stereo();
            peak = peak.max(s.left.abs()).max(s.right.abs());
        }
        assert!(
            peak > 0.10 && peak <= 0.95,
            "resonant low-freq bass peak ({peak}) out of controlled range"
        );

        // Now test volume de-zippering on a sustained sine: jumping
        // MasterVolume from 0.1 to 1.0 mid-note must not cause a step jump.
        let mut e2 = engine();
        e2.set_param(SynthParam::Osc1Waveform, 0.0); // Sine
        e2.set_param(SynthParam::MasterVolume, 0.1);
        e2.receive_event(NoteEvent::note_on_midi(60, 127));
        for _ in 0..2000 {
            e2.next_sample_stereo();
        }
        let prev = e2.next_sample_stereo().left;
        // Abrupt MIDI/Web volume jump while voice is active
        e2.set_param(SynthParam::MasterVolume, 1.0);
        let next = e2.next_sample_stereo().left;
        assert!(
            (next - prev).abs() < 0.02,
            "volume change caused sample discontinuity: prev={prev}, next={next}"
        );
    }

    #[test]
    fn lfo_pan_and_amp_modulation_ramps_per_sample_without_staircase_jumps() {
        let mut e = engine();
        // Route a fast Square LFO2 to both Osc1Pan and Amplitude so every LFO
        // flip would be a worst-case step jump if not interpolated per sample.
        e.set_lfo2_rate(12.0);
        e.set_lfo2_depth(1.0);
        e.set_lfo2_waveform(LfoWaveform::Square);
        e.set_mod_slot(
            0,
            ModSlot {
                enabled: true,
                source: ModSource::Lfo2,
                dest: ModDest::Osc1Pan,
                amount: 1.0,
            },
        );
        e.set_mod_slot(
            1,
            ModSlot {
                enabled: true,
                source: ModSource::Lfo2,
                dest: ModDest::Amplitude,
                amount: 0.8,
            },
        );
        e.receive_event(NoteEvent::note_on_midi(60, 127));
        let mut prev_g = e.voices[0].gains[0].0;
        let mut prev_amp = e.voices[0].amp_lfo;
        let mut max_dg = 0.0f32;
        let mut max_damp = 0.0f32;
        for _ in 0..8820 {
            e.next_sample_stereo();
            let g = e.voices[0].gains[0].0;
            let a = e.voices[0].amp_lfo;
            max_dg = max_dg.max((g - prev_g).abs());
            max_damp = max_damp.max((a - prev_amp).abs());
            prev_g = g;
            prev_amp = a;
        }
        // Per-sample step must be tiny (~1/64 of the control-tick slew), never
        // a 64-sample staircase jump.
        assert!(max_dg < 0.002, "pan gain stepped too abruptly: max_dg={max_dg}");
        assert!(max_damp < 0.015, "amp_lfo stepped too abruptly: max_damp={max_damp}");
    }

    #[test]
    fn noise_oscillator_taper_gives_subtle_low_end_and_strong_max() {
        let low_noise = LeanVoice::osc_level_gain(WaveformType::Noise, 0.05);
        let low_saw = LeanVoice::osc_level_gain(WaveformType::Saw, 0.05);
        let max_noise = LeanVoice::osc_level_gain(WaveformType::Noise, 1.0);
        let max_saw = LeanVoice::osc_level_gain(WaveformType::Saw, 1.0);
        // At 0.05 ("air" region), noise must be >10x quieter than a pitched osc
        assert!(low_noise * 10.0 < low_saw, "low_noise={low_noise}, low_saw={low_saw}");
        // At 1.0, noise still reaches 65% of full oscillator scale
        assert!((max_noise - max_saw * 0.65).abs() < 1e-6);
    }

    /// Max |sample-to-sample| delta over `n` rendered samples — the metric a
    /// listener hears as a click or scratch when a parameter steps.
    fn max_step(e: &mut EmbeddedEngine, n: usize) -> f32 {
        let mut prev = e.next_sample_stereo();
        let mut maxd = 0.0f32;
        for _ in 1..n {
            let s = e.next_sample_stereo();
            maxd = maxd
                .max((s.left - prev.left).abs())
                .max((s.right - prev.right).abs());
            prev = s;
        }
        maxd
    }

    #[test]
    fn synth_wide_slider_moves_do_not_click() {
        let mut e = engine();
        // Every sounding osc must be a clean low sine for the per-sample
        // delta metric — saw/square/noise edges have legitimate fast deltas
        // of their own that swamp the zipper measurement.
        e.set_param(SynthParam::Osc1Waveform, 0.0); // Sine (voice idle -> direct)
        e.set_param(SynthParam::Osc2Waveform, 0.0);
        e.set_param(SynthParam::Osc2Semitones, 0.0);
        e.set_param(SynthParam::Osc2Cents, 0.0);
        e.set_param(SynthParam::Osc3Level, 0.0);
        e.set_param(SynthParam::Osc4Level, 0.0);
        e.receive_event(NoteEvent::note_on_midi(60, 110));
        for _ in 0..8820 {
            e.next_sample_stereo();
        }
        let changes: &[(SynthParam, f32)] = &[
            (SynthParam::MasterVolume, 0.95),
            (SynthParam::Osc2Level, 0.9),
            (SynthParam::Osc2Pan, -0.85),
            (SynthParam::Osc2Pw, 0.15),
            (SynthParam::Osc2Semitones, 7.0),
            (SynthParam::Osc2Cents, -30.0),
            (SynthParam::FilterCutoff, 900.0),
            (SynthParam::FilterResonance, 0.85),
            (SynthParam::FilterEnvAmount, 0.8),
            (SynthParam::LfoDepth, 0.8),
            (SynthParam::LfoRate, 14.0),
            (SynthParam::StereoWidth, 1.8),
        ];
        for &(p, v) in changes {
            e.set_param(p, v);
            let mut prev = e.next_sample_stereo();
            let mut d = 0.0f32;
            let mut at = 0usize;
            let mut ch = 'l';
            for i in 1..192 {
                let s = e.next_sample_stereo();
                let dl = (s.left - prev.left).abs();
                let dr = (s.right - prev.right).abs();
                if dl > d {
                    d = dl;
                    at = i;
                    ch = 'l';
                }
                if dr > d {
                    d = dr;
                    at = i;
                    ch = 'r';
                }
                prev = s;
            }
            assert!(
                d < 0.05,
                "{p:?} -> {v} clicked: max step {d} at i={at} ch={ch} prev={prev:?}"
            );
        }
        // MIDI-style continuous controller sweeps: pitch bend, mod wheel,
        // aftertouch and CC7 volume — all through the same event path.
        for (kind, val) in [(0u8, 1.0f32), (1, 0.9), (2, 0.8)] {
            match kind {
                0 => e.receive_event(NoteEvent {
                    kind: SynthEventKind::PitchBend(val),
                    note: 0,
                    velocity: 0.0,
                }),
                1 => e.receive_event(NoteEvent::control_change(1, 115)),
                _ => e.receive_event(NoteEvent {
                    kind: SynthEventKind::Aftertouch(val),
                    note: 0,
                    velocity: 0.0,
                }),
            }
            let d = max_step(&mut e, 192);
            assert!(d < 0.05, "continuous controller {kind} clicked: {d}");
        }
        e.receive_event(NoteEvent::control_change(7, 127));
        let d = max_step(&mut e, 192);
        assert!(d < 0.05, "CC7 volume clicked: {d}");
    }

    #[test]
    fn fx_toggle_and_param_jumps_declick() {
        let mut e = engine();
        // Clean-sine source so the delta metric isolates the FX gates, not
        // the saw/square edges in the default patch.
        e.set_param(SynthParam::Osc1Waveform, 0.0);
        e.set_param(SynthParam::Osc2Level, 0.0);
        e.set_param(SynthParam::Osc3Level, 0.0);
        e.set_param(SynthParam::Osc4Level, 0.0);
        e.receive_event(NoteEvent::note_on_midi(60, 110));
        for _ in 0..8820 {
            e.next_sample_stereo();
        }
        // Stomp each effect on mid-note — the bypass gate must fade in.
        for (name, action) in [
            ("delay", 0),
            ("reverb", 1),
            ("chorus", 2),
            ("distortion", 3),
        ] {
            match action {
                0 => e.effects.set_delay_enabled(true),
                1 => e.effects.set_reverb_enabled(true),
                2 => e.effects.set_chorus_enabled(true),
                _ => e.effects.set_distortion_enabled(true),
            }
            let d = max_step(&mut e, 192);
            assert!(d < 0.05, "{name} enable clicked: {d}");
        }
        // Hard jumps on every effect slider while audio is running.
        e.effects.set_delay_time(0.05);
        e.effects.set_delay_feedback(0.9);
        e.effects.set_delay_mix(0.8);
        e.effects.set_reverb_room_size(0.95);
        e.effects.set_reverb_damping(0.9);
        e.effects.set_reverb_mix(0.8);
        e.effects.set_chorus_rate(4.5);
        e.effects.set_chorus_depth(9.0);
        e.effects.set_chorus_mix(0.9);
        e.effects.set_distortion_drive(0.95);
        e.effects.set_distortion_mix(0.9);
        e.effects.set_distortion_type(DistortionType::Fuzz);
        for _ in 0..32 {
            let d = max_step(&mut e, 64);
            assert!(d < 0.08, "fx param jump clicked: {d}");
        }
        // Stomp everything off — fade-outs must replace the old hard cut.
        // A hard-clipped Fuzz signal has large legitimate per-sample deltas of
        // its own, so the post-disable step must not exceed the pre-disable
        // wet step (the fade cannot add more than the signal itself).
        // Long window: rare hard-clipped Fuzz edges must be captured so the
        // comparison baseline reflects the true wet-signal step size.
        let wet_step = max_step(&mut e, 2048);
        e.effects.set_delay_enabled(false);
        e.effects.set_reverb_enabled(false);
        e.effects.set_chorus_enabled(false);
        e.effects.set_distortion_enabled(false);
        for _ in 0..64 {
            let d = max_step(&mut e, 64);
            assert!(
                d <= wet_step + 0.02,
                "fx disable added a step beyond the wet signal: {d} vs {wet_step}"
            );
        }
        // After the ~11 ms gates converge, output converges to the clean dry
        // path again (tight delta on a sine).
        for _ in 0..4410 {
            e.next_sample_stereo();
        }
        let d = max_step(&mut e, 256);
        assert!(d < 0.05, "post-disable output not clean: {d}");
    }

    #[test]
    fn waveform_filter_mode_and_preset_switches_stay_smooth() {
        let mut e = engine();
        e.receive_event(NoteEvent::note_on_midi(57, 120));
        for _ in 0..8820 {
            e.next_sample_stereo();
        }
        // Waveform swaps dip that osc instead of discontinuing the output.
        e.set_param(SynthParam::Osc1Waveform, 2.0); // Sine -> Saw
        let d = max_step(&mut e, 256);
        assert!(d < 0.05, "waveform swap clicked: {d}");
        // Filter mode switch crossfades between mode outputs.
        e.set_filter_mode(FilterMode::HighPass);
        let d = max_step(&mut e, 256);
        assert!(d < 0.05, "filter mode switch clicked: {d}");
        // Preset load while notes ring: everything glides, output bounded.
        let mut p = Preset::default();
        p.osc1_waveform = WaveformType::Saw;
        p.filter_cutoff = 1200.0;
        p.filter_resonance = 0.6;
        p.master_volume = 1.0;
        p.delay_enabled = true;
        p.reverb_enabled = true;
        e.apply_preset(&p);
        let mut peak = 0.0f32;
        for _ in 0..4096 {
            let s = e.next_sample_stereo();
            assert!(s.left.is_finite() && s.right.is_finite());
            peak = peak.max(s.left.abs()).max(s.right.abs());
        }
        assert!(peak <= 1.0, "post-preset peak {peak} out of bounds");
    }
}
