use std::f32::consts::PI;

use rinsynth_model::event::WaveformType;
use rinsynth_model::types::{Amplitude, Frequency, Sample, SampleRate};

/// 2048-entry sine table with linear interpolation — replaces `sin()` in
/// per-sample paths. Software libm on targets like Xtensa costs hundreds
/// of cycles per call; the LUT costs ~4.
const SINE_LUT_SIZE: usize = 2048;

fn sine_lut_table() -> &'static [f32; SINE_LUT_SIZE] {
    use std::sync::OnceLock;
    static LUT: OnceLock<Box<[f32; SINE_LUT_SIZE]>> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = Box::new([0.0f32; SINE_LUT_SIZE]);
        for (i, v) in t.iter_mut().enumerate() {
            *v = (i as f32 * 2.0 * PI / SINE_LUT_SIZE as f32).sin();
        }
        t
    })
}

/// sin(phase) for phase in radians; wraps to [0, 2π).
/// Callers keep phase ∈ [0, 2π) and offset ∈ [0, 2π) so the sum is < 4π —
/// conditional wraps avoid fmodf (soft libm on Xtensa).
#[inline]
#[link_section = ".iram1.text"]

fn sine_lut(phase: f32) -> f32 {
    let table = sine_lut_table();
    let mut p = phase;
    if p >= 2.0 * PI {
        p -= 2.0 * PI;
    }
    if p >= 2.0 * PI {
        p -= 2.0 * PI;
    }
    if p < 0.0 {
        p += 2.0 * PI;
    }
    let pos = p * (SINE_LUT_SIZE as f32 / (2.0 * PI));
    let idx = pos as usize & (SINE_LUT_SIZE - 1);
    let frac = pos - idx as f32;
    let a = table[idx];
    let b = table[(idx + 1) & (SINE_LUT_SIZE - 1)];
    a + (b - a) * frac
}

/// Cheap deterministic white noise — xorshift32. `white_noise()` in the
/// noise oscillator used to hash `SystemTime::now()` per sample.
#[inline]
fn xorshift32(state: &mut u32) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x as i32 as f32 / i32::MAX as f32
}

/// Trait for all oscillator types.
/// Oscillators generate audio samples at a given frequency.
pub trait Oscillator: Send + Sync {
    /// Generate the next sample
    #[link_section = ".iram1.text"]

    fn next_sample(&mut self) -> Sample;

    /// Set the oscillator frequency
    fn set_frequency(&mut self, frequency: Frequency);

    /// Get the current frequency
    fn frequency(&self) -> Frequency;

    /// Reset the oscillator phase to initial state
    fn reset(&mut self);

    /// Set the sample rate (called when audio engine initializes)
    fn set_sample_rate(&mut self, sample_rate: SampleRate);

    /// Set phase offset (0.0 to 1.0, where 1.0 = full cycle)
    fn set_phase_offset(&mut self, offset: f32);

    /// Get current phase offset
    fn phase_offset(&self) -> f32;

    /// Set pulse duty cycle (0.0 to 1.0). No-op for non-pulse waveforms.
    fn set_duty_cycle(&mut self, _duty: f32) {}
}

/// Sine wave oscillator - the purest waveform
pub struct SineOscillator {
    frequency: Frequency,
    phase: f32,
    phase_offset: f32, // 0.0 to 2π
    phase_increment: f32,
    sample_rate: SampleRate,
}

impl SineOscillator {
    pub fn new(frequency: Frequency, sample_rate: SampleRate) -> Self {
        let mut osc = Self {
            frequency,
            phase: 0.0,
            phase_offset: 0.0,
            phase_increment: 0.0,
            sample_rate,
        };
        osc.update_phase_increment();
        osc
    }

    fn update_phase_increment(&mut self) {
        self.phase_increment = 2.0 * PI * self.frequency / self.sample_rate as f32;
    }
}

impl Oscillator for SineOscillator {
    #[link_section = ".iram1.text"]

    fn next_sample(&mut self) -> Sample {
        // LUT lookup instead of sin() — software libm makes the per-sample
        // call prohibitive on targets without sin in hardware (Xtensa).
        let sample = sine_lut(self.phase + self.phase_offset);
        self.phase += self.phase_increment;
        if self.phase >= 2.0 * PI {
            self.phase -= 2.0 * PI;
        }
        sample
    }

    fn set_frequency(&mut self, frequency: Frequency) {
        self.frequency = frequency;
        self.update_phase_increment();
    }

    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
        self.update_phase_increment();
    }

    fn set_phase_offset(&mut self, offset: f32) {
        self.phase_offset = offset.clamp(0.0, 1.0) * 2.0 * PI;
    }

    fn phase_offset(&self) -> f32 {
        self.phase_offset / (2.0 * PI)
    }
}

/// Square wave oscillator - rich in odd harmonics
pub struct SquareOscillator {
    frequency: Frequency,
    phase: f32,
    phase_offset: f32, // 0.0 to 1.0
    phase_increment: f32,
    sample_rate: SampleRate,
    duty_cycle: f32,
}

impl SquareOscillator {
    pub fn new(frequency: Frequency, sample_rate: SampleRate) -> Self {
        let mut osc = Self {
            frequency,
            phase: 0.0,
            phase_offset: 0.0,
            phase_increment: 0.0,
            sample_rate,
            duty_cycle: 0.5,
        };
        osc.update_phase_increment();
        osc
    }

    pub fn set_duty_cycle(&mut self, duty: f32) {
        self.duty_cycle = duty.clamp(0.01, 0.99);
    }

    fn update_phase_increment(&mut self) {
        self.phase_increment = self.frequency / self.sample_rate as f32;
    }
}

impl Oscillator for SquareOscillator {
    #[link_section = ".iram1.text"]

    fn next_sample(&mut self) -> Sample {
        // phase + offset ∈ [0, 2): conditional wrap avoids fmodf (soft libm).
        let mut effective_phase = self.phase + self.phase_offset;
        if effective_phase >= 1.0 {
            effective_phase -= 1.0;
        }
        let sample = if effective_phase < self.duty_cycle {
            1.0
        } else {
            -1.0
        };
        self.phase += self.phase_increment;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        sample
    }

    fn set_frequency(&mut self, frequency: Frequency) {
        self.frequency = frequency;
        self.update_phase_increment();
    }

    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
        self.update_phase_increment();
    }

    fn set_phase_offset(&mut self, offset: f32) {
        self.phase_offset = offset.clamp(0.0, 1.0);
    }

    fn phase_offset(&self) -> f32 {
        self.phase_offset
    }

    fn set_duty_cycle(&mut self, duty: f32) {
        SquareOscillator::set_duty_cycle(self, duty);
    }
}

/// Sawtooth wave oscillator - all harmonics
pub struct SawOscillator {
    frequency: Frequency,
    phase: f32,
    phase_offset: f32, // 0.0 to 1.0
    phase_increment: f32,
    sample_rate: SampleRate,
}

impl SawOscillator {
    pub fn new(frequency: Frequency, sample_rate: SampleRate) -> Self {
        let mut osc = Self {
            frequency,
            phase: 0.0,
            phase_offset: 0.0,
            phase_increment: 0.0,
            sample_rate,
        };
        osc.update_phase_increment();
        osc
    }

    fn update_phase_increment(&mut self) {
        self.phase_increment = self.frequency / self.sample_rate as f32;
    }
}

impl Oscillator for SawOscillator {
    #[link_section = ".iram1.text"]

    fn next_sample(&mut self) -> Sample {
        let mut effective_phase = self.phase + self.phase_offset;
        if effective_phase >= 1.0 {
            effective_phase -= 1.0;
        }
        let sample = 2.0 * effective_phase - 1.0;
        self.phase += self.phase_increment;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        sample
    }

    fn set_frequency(&mut self, frequency: Frequency) {
        self.frequency = frequency;
        self.update_phase_increment();
    }

    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
        self.update_phase_increment();
    }

    fn set_phase_offset(&mut self, offset: f32) {
        self.phase_offset = offset.clamp(0.0, 1.0);
    }

    fn phase_offset(&self) -> f32 {
        self.phase_offset
    }
}

/// Triangle wave oscillator - odd harmonics, softer than square
pub struct TriangleOscillator {
    frequency: Frequency,
    phase: f32,
    phase_offset: f32, // 0.0 to 1.0
    phase_increment: f32,
    sample_rate: SampleRate,
}

impl TriangleOscillator {
    pub fn new(frequency: Frequency, sample_rate: SampleRate) -> Self {
        let mut osc = Self {
            frequency,
            phase: 0.0,
            phase_offset: 0.0,
            phase_increment: 0.0,
            sample_rate,
        };
        osc.update_phase_increment();
        osc
    }

    fn update_phase_increment(&mut self) {
        self.phase_increment = self.frequency / self.sample_rate as f32;
    }
}

impl Oscillator for TriangleOscillator {
    #[link_section = ".iram1.text"]

    fn next_sample(&mut self) -> Sample {
        let mut effective_phase = self.phase + self.phase_offset;
        if effective_phase >= 1.0 {
            effective_phase -= 1.0;
        }
        let sample = if effective_phase < 0.5 {
            4.0 * effective_phase - 1.0
        } else {
            3.0 - 4.0 * effective_phase
        };
        self.phase += self.phase_increment;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        sample
    }

    fn set_frequency(&mut self, frequency: Frequency) {
        self.frequency = frequency;
        self.update_phase_increment();
    }

    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
        self.update_phase_increment();
    }

    fn set_phase_offset(&mut self, offset: f32) {
        self.phase_offset = offset.clamp(0.0, 1.0);
    }

    fn phase_offset(&self) -> f32 {
        self.phase_offset
    }
}

/// Noise oscillator - generates white noise
/// Unlike other oscillators, frequency controls the "color" of noise (filter cutoff)
pub struct NoiseOscillator {
    frequency: Frequency, // Used as filter cutoff for noise color
    sample_rate: SampleRate,
    phase_offset: f32,
    // Simple state for filtering (one-pole lowpass for pink noise option)
    filter_state: f32,
    rng: u32,
}

impl NoiseOscillator {
    pub fn new(frequency: Frequency, sample_rate: SampleRate) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEED: AtomicU32 = AtomicU32::new(0x9E3779B9);
        Self {
            frequency,
            sample_rate,
            phase_offset: 0.0,
            filter_state: 0.0,
            rng: SEED.fetch_add(0x6D2B79F5, Ordering::Relaxed) | 1,
        }
    }
}

impl Oscillator for NoiseOscillator {
    #[link_section = ".iram1.text"]

    fn next_sample(&mut self) -> Sample {
        // Generate white noise
        let white = xorshift32(&mut self.rng);

        // Apply simple one-pole filter for "color" based on frequency
        // Higher frequency = less filtering (brighter noise)
        // cutoff_ratio: 0.0 = full filter (darker), 1.0 = no filter (white)
        let cutoff_ratio = (self.frequency / 20000.0).clamp(0.0, 1.0);
        let filter_coeff = 1.0 - cutoff_ratio; // Lower cutoff = more filtering

        // One-pole lowpass filter
        self.filter_state = white * (1.0 - filter_coeff) + self.filter_state * filter_coeff;

        self.filter_state
    }

    fn set_frequency(&mut self, frequency: Frequency) {
        self.frequency = frequency;
    }

    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn reset(&mut self) {
        self.filter_state = 0.0;
    }

    fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
    }

    fn set_phase_offset(&mut self, _offset: f32) {
        // Noise doesn't use phase, but we store it for API compatibility
        self.phase_offset = 0.0;
    }

    fn phase_offset(&self) -> f32 {
        0.0 // Noise has no phase
    }
}

/// Factory function to create an oscillator from WaveformType
pub fn create_oscillator(
    waveform: WaveformType,
    frequency: Frequency,
    sample_rate: SampleRate,
) -> Box<dyn Oscillator> {
    match waveform {
        WaveformType::Sine => Box::new(SineOscillator::new(frequency, sample_rate)),
        WaveformType::Square => Box::new(SquareOscillator::new(frequency, sample_rate)),
        WaveformType::Saw => Box::new(SawOscillator::new(frequency, sample_rate)),
        WaveformType::Triangle => Box::new(TriangleOscillator::new(frequency, sample_rate)),
        WaveformType::Noise => Box::new(NoiseOscillator::new(frequency, sample_rate)),
        WaveformType::Pulse => Box::new(SquareOscillator::new(frequency, sample_rate)),
    }
}

/// Convert semitones and cents to frequency ratio
/// semitones: -24 to +24 (2 octaves)
/// cents: -100 to +100 (fine tune within semitone)
#[inline]
pub fn detune_ratio(semitones: i8, cents: i8) -> f32 {
    let total_semitones = semitones as f32 + (cents as f32 / 100.0);
    2.0_f32.powf(total_semitones / 12.0)
}

/// A single oscillator unit with waveform, detune, phase, and level controls
pub struct OscillatorUnit {
    oscillator: Box<dyn Oscillator>,
    waveform: WaveformType,
    semitones: i8,     // -24 to +24
    cents: i8,         // -100 to +100
    phase_offset: f32, // 0.0 to 1.0
    level: Amplitude,  // 0.0 to 1.0
    /// Per-sample pitch modulation in semitones (mod matrix / FM).
    pitch_mod: f32,
    base_frequency: Frequency,
    sample_rate: SampleRate,
    /// Pulse width base + modulation (0.0..=1.0, clamped by oscillator).
    pulse_width: f32,
}

impl OscillatorUnit {
    pub fn new(sample_rate: SampleRate) -> Self {
        Self {
            oscillator: Box::new(SineOscillator::new(440.0, sample_rate)),
            waveform: WaveformType::Sine,
            semitones: 0,
            cents: 0,
            phase_offset: 0.0,
            level: 1.0,
            pitch_mod: 0.0,
            base_frequency: 440.0,
            sample_rate,
            pulse_width: 0.5,
        }
    }

    /// Set the waveform type
    pub fn set_waveform(&mut self, waveform: WaveformType) {
        if self.waveform != waveform {
            self.waveform = waveform;
            let freq = self.oscillator.frequency();
            let phase = self.phase_offset;
            self.oscillator = create_oscillator(waveform, freq, self.sample_rate);
            self.oscillator.set_phase_offset(phase);
        }
    }

    /// Get current waveform
    pub fn waveform(&self) -> WaveformType {
        self.waveform
    }

    /// Set detune in semitones (-24 to +24)
    pub fn set_semitones(&mut self, semitones: i8) {
        self.semitones = semitones.clamp(-24, 24);
        self.update_frequency();
    }

    /// Get current semitone detune
    pub fn semitones(&self) -> i8 {
        self.semitones
    }

    /// Set fine detune in cents (-100 to +100)
    pub fn set_cents(&mut self, cents: i8) {
        self.cents = cents.clamp(-100, 100);
        self.update_frequency();
    }

    /// Get current cents detune
    pub fn cents(&self) -> i8 {
        self.cents
    }

    /// Set phase offset (0.0 to 1.0, where 1.0 = full cycle)
    pub fn set_phase_offset(&mut self, offset: f32) {
        self.phase_offset = offset.clamp(0.0, 1.0);
        self.oscillator.set_phase_offset(self.phase_offset);
    }

    /// Get current phase offset
    pub fn phase_offset(&self) -> f32 {
        self.phase_offset
    }

    /// Set the level (0.0 to 1.0)
    pub fn set_level(&mut self, level: Amplitude) {
        self.level = level.clamp(0.0, 1.0);
    }

    /// Get current level
    pub fn level(&self) -> Amplitude {
        self.level
    }

    /// Set the base frequency (before detune)
    pub fn set_base_frequency(&mut self, frequency: Frequency) {
        self.base_frequency = frequency;
        self.update_frequency();
    }

    /// Set per-oscillator pitch modulation in semitones.
    ///
    /// Called once per sample from the voice's modulation stage; cheap when
    /// unchanged because it early-returns without touching the oscillator.
    pub fn set_pitch_mod(&mut self, semitones: f32) {
        if self.pitch_mod == semitones {
            return;
        }
        self.pitch_mod = semitones;
        self.update_frequency();
    }

    /// Update oscillator frequency based on base + detune + pitch mod
    fn update_frequency(&mut self) {
        let ratio = detune_ratio(self.semitones, self.cents) * 2.0_f32.powf(self.pitch_mod / 12.0);
        self.oscillator.set_frequency(self.base_frequency * ratio);
    }

    /// Set pulse width (0.0 to 1.0). Only affects pulse/square oscillators.
    pub fn set_pulse_width(&mut self, width: f32) {
        self.pulse_width = width.clamp(0.0, 1.0);
    }

    /// Generate next sample (with level applied)
    #[link_section = ".iram1.text"]

    pub fn next_sample(&mut self) -> Sample {
        self.oscillator.set_duty_cycle(self.pulse_width);
        self.oscillator.next_sample() * self.level
    }

    /// Generate next sample WITHOUT level applied (for the modulation matrix,
    /// which feeds osc4's raw output back in as an FM/RM source).
    #[link_section = ".iram1.text"]

    pub fn next_sample_unscaled(&mut self) -> Sample {
        self.oscillator.next_sample()
    }

    /// Reset oscillator phase
    pub fn reset(&mut self) {
        self.oscillator.reset();
    }

    /// Set sample rate
    pub fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
        self.oscillator.set_sample_rate(sample_rate);
    }
}

/// Minimoog-style oscillator bank with 4 oscillators
/// OSC 1: Main oscillator (tuned to note)
/// OSC 2: Detunable relative to OSC 1
/// OSC 3: Detunable relative to OSC 1
/// OSC 4: Detunable; its raw output doubles as an FM/RM mod source
pub struct OscillatorBank {
    osc1: OscillatorUnit,
    osc2: OscillatorUnit,
    osc3: OscillatorUnit,
    osc4: OscillatorUnit,
    base_frequency: Frequency,
    pitch_bend: f32,      // -1.0 to +1.0
    pitch_bend_range: u8, // semitones (default 12 = 1 octave)
    sample_rate: SampleRate,
    /// Per-oscillator tick mask — false entries produce silence and cost
    /// nothing. Set by the voice each control tick (level-0 oscillators with
    /// no mod route to their level contribute nothing).
    active: [bool; 4],
}

impl OscillatorBank {
    pub fn new(sample_rate: SampleRate) -> Self {
        // Osc4 defaults to silent: it exists primarily as an FM/RM modulation
        // source, audible only when its level is raised.
        let mut osc4 = OscillatorUnit::new(sample_rate);
        osc4.set_level(0.0);
        Self {
            osc1: OscillatorUnit::new(sample_rate),
            osc2: OscillatorUnit::new(sample_rate),
            osc3: OscillatorUnit::new(sample_rate),
            osc4,
            base_frequency: 440.0,
            pitch_bend: 0.0,
            pitch_bend_range: 12, // 1 octave default
            sample_rate,
            active: [true; 4],
        }
    }

    /// Set pitch bend (-1.0 to +1.0)
    pub fn set_pitch_bend(&mut self, bend: f32) {
        self.pitch_bend = bend.clamp(-1.0, 1.0);
        self.update_frequencies();
    }

    /// Get current pitch bend
    pub fn pitch_bend(&self) -> f32 {
        self.pitch_bend
    }

    /// Set pitch bend range in semitones (1-24)
    pub fn set_pitch_bend_range(&mut self, semitones: u8) {
        self.pitch_bend_range = semitones.clamp(1, 24);
    }

    /// Get pitch bend range
    pub fn pitch_bend_range(&self) -> u8 {
        self.pitch_bend_range
    }

    /// Update all oscillator frequencies with pitch bend applied
    fn update_frequencies(&mut self) {
        let bend_semitones = self.pitch_bend * self.pitch_bend_range as f32;
        let bend_ratio = 2.0_f32.powf(bend_semitones / 12.0);
        let bent_freq = self.base_frequency * bend_ratio;
        self.osc1.set_base_frequency(bent_freq);
        self.osc2.set_base_frequency(bent_freq);
        self.osc3.set_base_frequency(bent_freq);
        self.osc4.set_base_frequency(bent_freq);
    }

    /// Apply per-oscillator pitch modulation (semitones) for this sample.
    /// Each `OscillatorUnit::set_pitch_mod` is a no-op when unchanged.
    pub fn set_pitch_mods(&mut self, mods: [f32; 4]) {
        self.osc1.set_pitch_mod(mods[0]);
        self.osc2.set_pitch_mod(mods[1]);
        self.osc3.set_pitch_mod(mods[2]);
        self.osc4.set_pitch_mod(mods[3]);
    }

    /// Get mutable reference to oscillator 1 (main)
    pub fn osc1_mut(&mut self) -> &mut OscillatorUnit {
        &mut self.osc1
    }

    /// Get mutable reference to oscillator 2
    pub fn osc2_mut(&mut self) -> &mut OscillatorUnit {
        &mut self.osc2
    }

    /// Get mutable reference to oscillator 3
    pub fn osc3_mut(&mut self) -> &mut OscillatorUnit {
        &mut self.osc3
    }

    /// Get mutable reference to oscillator 4
    pub fn osc4_mut(&mut self) -> &mut OscillatorUnit {
        &mut self.osc4
    }

    /// Get reference to oscillator 1
    pub fn osc1(&self) -> &OscillatorUnit {
        &self.osc1
    }

    /// Get reference to oscillator 2
    pub fn osc2(&self) -> &OscillatorUnit {
        &self.osc2
    }

    /// Get reference to oscillator 3
    pub fn osc3(&self) -> &OscillatorUnit {
        &self.osc3
    }

    /// Get reference to oscillator 4
    pub fn osc4(&self) -> &OscillatorUnit {
        &self.osc4
    }

    /// Set waveform for a specific oscillator (1-4)
    pub fn set_waveform(&mut self, osc_num: u8, waveform: WaveformType) {
        match osc_num {
            1 => self.osc1.set_waveform(waveform),
            2 => self.osc2.set_waveform(waveform),
            3 => self.osc3.set_waveform(waveform),
            4 => self.osc4.set_waveform(waveform),
            _ => {}
        }
    }

    /// Set detune for a specific oscillator
    pub fn set_detune(&mut self, osc_num: u8, semitones: i8, cents: i8) {
        match osc_num {
            1 => {
                self.osc1.set_semitones(semitones);
                self.osc1.set_cents(cents);
            }
            2 => {
                self.osc2.set_semitones(semitones);
                self.osc2.set_cents(cents);
            }
            3 => {
                self.osc3.set_semitones(semitones);
                self.osc3.set_cents(cents);
            }
            4 => {
                self.osc4.set_semitones(semitones);
                self.osc4.set_cents(cents);
            }
            _ => {}
        }
    }

    /// Set level for a specific oscillator
    pub fn set_level(&mut self, osc_num: u8, level: Amplitude) {
        match osc_num {
            1 => self.osc1.set_level(level),
            2 => self.osc2.set_level(level),
            3 => self.osc3.set_level(level),
            4 => self.osc4.set_level(level),
            _ => {}
        }
    }

    /// Set phase offset for a specific oscillator (0.0 to 1.0)
    pub fn set_phase(&mut self, osc_num: u8, phase: f32) {
        match osc_num {
            1 => self.osc1.set_phase_offset(phase),
            2 => self.osc2.set_phase_offset(phase),
            3 => self.osc3.set_phase_offset(phase),
            4 => self.osc4.set_phase_offset(phase),
            _ => {}
        }
    }

    /// Set pulse widths for all oscillators at once (per-sample modulation).
    pub fn set_pulse_widths(&mut self, widths: [f32; 4]) {
        self.osc1.set_pulse_width(widths[0]);
        self.osc2.set_pulse_width(widths[1]);
        self.osc3.set_pulse_width(widths[2]);
        self.osc4.set_pulse_width(widths[3]);
    }
}

impl OscillatorBank {
    /// Get individual oscillator samples (for stereo panning), normalized
    /// by the 4-oscillator count.
    ///
    /// Returns `(samples, osc4_raw)` where `samples` are the level-scaled
    /// per-oscillator outputs and `osc4_raw` is oscillator 4's unscaled
    /// output — the modulation matrix feeds it back as an FM/RM source.
    /// Per-oscillator activity mask, refreshed at control rate by the voice.
    pub fn set_active(&mut self, active: [bool; 4]) {
        self.active = active;
    }

    #[link_section = ".iram1.text"]

    pub fn next_samples_individual(&mut self) -> ([Sample; 4], Sample) {
        // Advance osc4 first so its raw output can be reused for both the
        // audio mix and the modulation source without double-stepping.
        let osc4_raw = if self.active[3] {
            self.osc4.next_sample_unscaled()
        } else {
            0.0
        };
        let s1 = if self.active[0] {
            self.osc1.next_sample() / 4.0
        } else {
            0.0
        };
        let s2 = if self.active[1] {
            self.osc2.next_sample() / 4.0
        } else {
            0.0
        };
        let s3 = if self.active[2] {
            self.osc3.next_sample() / 4.0
        } else {
            0.0
        };
        let s4 = osc4_raw * self.osc4.level() / 4.0;
        ([s1, s2, s3, s4], osc4_raw)
    }
}

impl Oscillator for OscillatorBank {
    #[link_section = ".iram1.text"]

    fn next_sample(&mut self) -> Sample {
        // Mix all four oscillators (masked units produce 0)
        let s1 = if self.active[0] { self.osc1.next_sample() } else { 0.0 };
        let s2 = if self.active[1] { self.osc2.next_sample() } else { 0.0 };
        let s3 = if self.active[2] { self.osc3.next_sample() } else { 0.0 };
        let s4 = if self.active[3] { self.osc4.next_sample() } else { 0.0 };
        (s1 + s2 + s3 + s4) / 4.0
    }

    fn set_frequency(&mut self, frequency: Frequency) {
        self.base_frequency = frequency;
        self.update_frequencies();
    }

    fn frequency(&self) -> Frequency {
        self.base_frequency
    }

    fn reset(&mut self) {
        self.osc1.reset();
        self.osc2.reset();
        self.osc3.reset();
        self.osc4.reset();
    }

    fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
        self.osc1.set_sample_rate(sample_rate);
        self.osc2.set_sample_rate(sample_rate);
        self.osc3.set_sample_rate(sample_rate);
        self.osc4.set_sample_rate(sample_rate);
    }

    fn set_phase_offset(&mut self, offset: f32) {
        // Set phase for all oscillators (typically you'd use set_phase per-osc)
        self.osc1.set_phase_offset(offset);
        self.osc2.set_phase_offset(offset);
        self.osc3.set_phase_offset(offset);
        self.osc4.set_phase_offset(offset);
    }

    fn phase_offset(&self) -> f32 {
        // Return OSC1's phase as reference
        self.osc1.phase_offset()
    }
}

#[cfg(test)]
mod synth_plot_tests {
    use super::*;
    use crate::envelope::{ADSREnvelope, AREnvelope, Envelope};
    use std::path::PathBuf;

    /// Synthesize audio by multiplying oscillator samples with envelope amplitude.
    /// Returns (audio_samples, envelope_samples) for plotting.
    fn synthesize<O: Oscillator, E: Envelope>(
        osc: &mut O,
        env: &mut E,
        events: &[(f32, bool)], // (time_seconds, is_note_on)
        duration_seconds: f32,
        sample_rate: u32,
    ) -> (Vec<f32>, Vec<f32>, Vec<(usize, bool)>) {
        let total_samples = (duration_seconds * sample_rate as f32) as usize;
        let mut audio = Vec::with_capacity(total_samples);
        let mut envelope_curve = Vec::with_capacity(total_samples);
        let mut event_markers = Vec::new();

        let mut event_index = 0;
        for sample in 0..total_samples {
            // Fire events at the correct sample
            while event_index < events.len() {
                let event_sample = (events[event_index].0 * sample_rate as f32) as usize;
                if event_sample == sample {
                    let is_on = events[event_index].1;
                    if is_on {
                        env.trigger();
                    } else {
                        env.release();
                    }
                    event_markers.push((sample, is_on));
                    event_index += 1;
                } else {
                    break;
                }
            }
            let amp = env.next_amplitude();
            let raw = osc.next_sample();
            envelope_curve.push(amp);
            audio.push(raw * amp);
        }
        (audio, envelope_curve, event_markers)
    }

    fn plot_synth_output(
        audio: &[f32],
        envelope: &[f32],
        events: &[(usize, bool)],
        sample_rate: u32,
        filename: &str,
        title: &str,
    ) {
        use plotters::prelude::*;

        let path = PathBuf::from(filename);
        {
            let root = BitMapBackend::new(&path, (1400, 800)).into_drawing_area();
            root.fill(&WHITE).unwrap();

            let duration = audio.len() as f32 / sample_rate as f32;
            let (top, bottom) = root.split_vertically(400);

            // --- TOP PANEL: full waveform (subsampled) + envelope overlay ---
            let subsample = (audio.len() / 4000).max(1);

            let mut chart = ChartBuilder::on(&top)
                .caption(title, ("sans-serif", 22))
                .margin(10)
                .x_label_area_size(30)
                .y_label_area_size(50)
                .build_cartesian_2d(0.0f32..duration, -1.1f32..1.1f32)
                .unwrap();

            chart
                .configure_mesh()
                .x_desc("Time (s)")
                .y_desc("Amplitude")
                .draw()
                .unwrap();

            // Audio waveform
            chart
                .draw_series(LineSeries::new(
                    audio
                        .iter()
                        .enumerate()
                        .step_by(subsample)
                        .map(|(i, &s)| (i as f32 / sample_rate as f32, s)),
                    BLUE.mix(0.5),
                ))
                .unwrap()
                .label("Sine * Envelope")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

            // Envelope upper bound
            chart
                .draw_series(LineSeries::new(
                    envelope
                        .iter()
                        .enumerate()
                        .step_by(subsample)
                        .map(|(i, &e)| (i as f32 / sample_rate as f32, e)),
                    RED.stroke_width(2),
                ))
                .unwrap()
                .label("Envelope")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

            // Envelope lower bound (mirror)
            chart
                .draw_series(LineSeries::new(
                    envelope
                        .iter()
                        .enumerate()
                        .step_by(subsample)
                        .map(|(i, &e)| (i as f32 / sample_rate as f32, -e)),
                    RED.stroke_width(2),
                ))
                .unwrap();

            // Note on/off markers
            for (idx, is_on) in events {
                let x = *idx as f32 / sample_rate as f32;
                let color = if *is_on { GREEN } else { RED };
                chart
                    .draw_series(std::iter::once(TriangleMarker::new(
                        (x, 1.05),
                        10,
                        color.filled(),
                    )))
                    .unwrap();
            }

            chart
                .configure_series_labels()
                .background_style(WHITE.mix(0.8))
                .border_style(BLACK)
                .draw()
                .unwrap();

            // --- BOTTOM PANEL: zoomed attack (first 200ms) ---
            let zoom_samples = ((sample_rate as f32 * 0.2) as usize).min(audio.len());
            let zoom_dur = zoom_samples as f32 / sample_rate as f32;

            let mut chart2 = ChartBuilder::on(&bottom)
                .caption("Zoomed: Attack region (first 200 ms)", ("sans-serif", 18))
                .margin(10)
                .x_label_area_size(30)
                .y_label_area_size(50)
                .build_cartesian_2d(0.0f32..zoom_dur, -1.1f32..1.1f32)
                .unwrap();

            chart2
                .configure_mesh()
                .x_desc("Time (s)")
                .y_desc("Amplitude")
                .draw()
                .unwrap();

            chart2
                .draw_series(LineSeries::new(
                    audio[..zoom_samples]
                        .iter()
                        .enumerate()
                        .map(|(i, &s)| (i as f32 / sample_rate as f32, s)),
                    BLUE.mix(0.6),
                ))
                .unwrap();

            chart2
                .draw_series(LineSeries::new(
                    envelope[..zoom_samples]
                        .iter()
                        .enumerate()
                        .map(|(i, &e)| (i as f32 / sample_rate as f32, e)),
                    &RED,
                ))
                .unwrap();

            chart2
                .draw_series(LineSeries::new(
                    envelope[..zoom_samples]
                        .iter()
                        .enumerate()
                        .map(|(i, &e)| (i as f32 / sample_rate as f32, -e)),
                    &RED,
                ))
                .unwrap();

            root.present().unwrap();
        }
        println!("Saved: {}", path.display());
    }

    fn check_waveform_continuity(
        audio: &[f32],
        events: &[(usize, bool)],
    ) -> Vec<(usize, f32, f32)> {
        let mut jumps = Vec::new();
        for i in 1..audio.len() {
            let delta = (audio[i] - audio[i - 1]).abs();
            // Flag only at event boundaries — these are the most likely click sources
            let near_event = events
                .iter()
                .any(|(idx, _)| (*idx as isize - i as isize).abs() <= 5);
            if near_event && delta > 0.1 {
                jumps.push((i, audio[i - 1], audio[i]));
            }
        }
        jumps
    }

    /// Simple ADSR-enveloped 440 Hz sine: one note on, one note off
    #[test]
    fn plot_sine_adsr_simple() {
        let sample_rate = 44100u32;
        let mut osc = SineOscillator::new(440.0, sample_rate);
        let mut env = ADSREnvelope::new(0.1, 0.2, 0.7, 0.3, sample_rate);

        let events = vec![(0.0, true), (1.0, false)];
        let (audio, envelope, markers) = synthesize(&mut osc, &mut env, &events, 1.8, sample_rate);

        plot_synth_output(
            &audio,
            &envelope,
            &markers,
            sample_rate,
            "test_synth_sine_adsr_simple.png",
            "Sine + ADSR Envelope (440 Hz)",
        );

        let jumps = check_waveform_continuity(&audio, &markers);
        assert!(
            jumps.is_empty(),
            "Detected {} amplitude jumps near note events (potential clicks):\n{:?}",
            jumps.len(),
            jumps
        );
    }

    /// Fast attack (1 ms) — checks for onset click
    #[test]
    fn plot_sine_adsr_fast_attack() {
        let sample_rate = 44100u32;
        let mut osc = SineOscillator::new(440.0, sample_rate);
        let mut env = ADSREnvelope::new(0.001, 0.1, 0.6, 0.2, sample_rate);

        let events = vec![(0.0, true), (0.5, false)];
        let (audio, envelope, markers) = synthesize(&mut osc, &mut env, &events, 0.9, sample_rate);

        plot_synth_output(
            &audio,
            &envelope,
            &markers,
            sample_rate,
            "test_synth_sine_fast_attack.png",
            "Sine + ADSR — Fast Attack (1 ms)",
        );

        let jumps = check_waveform_continuity(&audio, &markers);
        assert!(
            jumps.is_empty(),
            "Fast attack caused {} click-like jumps:\n{:?}",
            jumps.len(),
            jumps
        );
    }

    /// Re-trigger during release — the most common source of clicks
    #[test]
    fn plot_sine_adsr_retrigger_during_release() {
        let sample_rate = 44100u32;
        let mut osc = SineOscillator::new(440.0, sample_rate);
        let mut env = ADSREnvelope::new(0.05, 0.1, 0.6, 0.3, sample_rate);

        // Note off, then note on again before release finishes
        let events = vec![
            (0.0, true),
            (0.3, false),
            (0.45, true), // re-trigger mid-release
            (0.9, false),
        ];
        let (audio, envelope, markers) = synthesize(&mut osc, &mut env, &events, 1.4, sample_rate);

        plot_synth_output(
            &audio,
            &envelope,
            &markers,
            sample_rate,
            "test_synth_sine_retrigger.png",
            "Sine + ADSR — Re-trigger During Release",
        );

        println!("\n=== RETRIGGER CLICK ANALYSIS ===");
        let jumps = check_waveform_continuity(&audio, &markers);
        if jumps.is_empty() {
            println!("✓ No clicks detected at re-trigger boundaries");
        } else {
            for (idx, prev, curr) in &jumps {
                let t = *idx as f32 / sample_rate as f32;
                println!(
                    "  ⚠ Click at {:.4}s: {:.4} -> {:.4} (Δ={:.4})",
                    t,
                    prev,
                    curr,
                    (curr - prev).abs()
                );
            }
        }

        assert!(
            jumps.is_empty(),
            "Re-trigger caused {} click-like jumps:\n{:?}",
            jumps.len(),
            jumps
        );
    }

    /// AR envelope on a sine — compare with ADSR
    #[test]
    fn plot_sine_ar_vs_adsr() {
        let sample_rate = 44100u32;
        let events = vec![(0.0, true), (0.6, false)];

        // AR
        let mut osc_ar = SineOscillator::new(440.0, sample_rate);
        let mut ar = AREnvelope::new(0.1, 0.3, sample_rate);
        let (ar_audio, ar_env, ar_markers) =
            synthesize(&mut osc_ar, &mut ar, &events, 1.2, sample_rate);
        plot_synth_output(
            &ar_audio,
            &ar_env,
            &ar_markers,
            sample_rate,
            "test_synth_sine_ar.png",
            "Sine + AR Envelope (440 Hz)",
        );

        // ADSR
        let mut osc_adsr = SineOscillator::new(440.0, sample_rate);
        let mut adsr = ADSREnvelope::new(0.1, 0.15, 0.7, 0.3, sample_rate);
        let (adsr_audio, adsr_env, adsr_markers) =
            synthesize(&mut osc_adsr, &mut adsr, &events, 1.2, sample_rate);
        plot_synth_output(
            &adsr_audio,
            &adsr_env,
            &adsr_markers,
            sample_rate,
            "test_synth_sine_adsr.png",
            "Sine + ADSR Envelope (440 Hz)",
        );

        println!("AR plot and ADSR plot both saved.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_oscillator_range() {
        let mut osc = SineOscillator::new(440.0, 44100);
        for _ in 0..1000 {
            let sample = osc.next_sample();
            assert!(sample >= -1.0 && sample <= 1.0);
        }
    }

    #[test]
    fn test_square_oscillator_values() {
        let mut osc = SquareOscillator::new(440.0, 44100);
        for _ in 0..1000 {
            let sample = osc.next_sample();
            assert!(sample == 1.0 || sample == -1.0);
        }
    }

    #[test]
    fn test_detune_ratio() {
        // No detune = ratio of 1.0
        assert!((detune_ratio(0, 0) - 1.0).abs() < 0.001);
        // +12 semitones = octave up = ratio of 2.0
        assert!((detune_ratio(12, 0) - 2.0).abs() < 0.001);
        // -12 semitones = octave down = ratio of 0.5
        assert!((detune_ratio(-12, 0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_oscillator_bank() {
        let mut bank = OscillatorBank::new(44100);

        // Set different waveforms
        bank.set_waveform(1, WaveformType::Saw);
        bank.set_waveform(2, WaveformType::Square);
        bank.set_waveform(3, WaveformType::Sine);

        // Detune osc 2 and 3
        bank.set_detune(2, 0, 10); // +10 cents
        bank.set_detune(3, -12, 0); // -1 octave

        bank.set_frequency(440.0);

        // Generate samples - should be in valid range
        for _ in 0..1000 {
            let sample = bank.next_sample();
            assert!(sample >= -1.0 && sample <= 1.0);
        }
    }
}
