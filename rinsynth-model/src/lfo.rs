use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

use super::types::SampleRate;

/// LFO routing destinations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LfoDestination {
    #[default]
    Off,
    Pitch,        // Modulate oscillator pitch (vibrato)
    FilterCutoff, // Modulate filter cutoff
    Amplitude,    // Modulate amplitude (tremolo)
}

impl LfoDestination {
    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            LfoDestination::Off => "Off",
            LfoDestination::Pitch => "Pitch",
            LfoDestination::FilterCutoff => "Filter",
            LfoDestination::Amplitude => "Amplitude",
        }
    }

    /// Get all destinations
    pub fn all() -> &'static [LfoDestination] {
        &[
            LfoDestination::Off,
            LfoDestination::Pitch,
            LfoDestination::FilterCutoff,
            LfoDestination::Amplitude,
        ]
    }
}

/// LFO waveform types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LfoWaveform {
    #[default]
    Sine,
    Triangle,
    Square,
    Saw,
    Random, // Sample & hold
}

impl LfoWaveform {
    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            LfoWaveform::Sine => "Sine",
            LfoWaveform::Triangle => "Triangle",
            LfoWaveform::Square => "Square",
            LfoWaveform::Saw => "Saw",
            LfoWaveform::Random => "Random",
        }
    }
}

/// Low Frequency Oscillator for modulation
/// Generates sub-audio rate waveforms (0.1-20 Hz) for pitch/filter/amplitude modulation
pub struct LFO {
    rate: f32,  // Frequency in Hz (0.1-20)
    depth: f32, // Modulation depth 0.0-1.0
    waveform: LfoWaveform,
    destination: LfoDestination,
    phase: f32, // 0.0 to 2π
    phase_increment: f32,
    sample_rate: SampleRate,
    // Random/S&H state
    last_random_value: f32,
    last_random_phase: f32,
}

impl LFO {
    pub fn new(sample_rate: SampleRate) -> Self {
        let mut lfo = Self {
            rate: 6.0,  // Default 6 Hz (medium vibrato)
            depth: 0.0, // Off by default
            waveform: LfoWaveform::Sine,
            destination: LfoDestination::Off,
            phase: 0.0,
            phase_increment: 0.0,
            sample_rate,
            last_random_value: 0.0,
            last_random_phase: 0.0,
        };
        lfo.update_phase_increment();
        lfo
    }

    /// Set LFO rate in Hz (0.1 to 20)
    pub fn set_rate(&mut self, rate: f32) {
        self.rate = rate.clamp(0.1, 20.0);
        self.update_phase_increment();
    }

    /// Get current rate
    pub fn rate(&self) -> f32 {
        self.rate
    }

    /// Set modulation depth (0.0 to 1.0)
    pub fn set_depth(&mut self, depth: f32) {
        self.depth = depth.clamp(0.0, 1.0);
    }

    /// Get current depth
    pub fn depth(&self) -> f32 {
        self.depth
    }

    /// Set waveform
    pub fn set_waveform(&mut self, waveform: LfoWaveform) {
        self.waveform = waveform;
        // Reset random state when switching to/from random
        if waveform == LfoWaveform::Random {
            self.last_random_value = random_value();
        }
    }

    /// Get current waveform
    pub fn waveform(&self) -> LfoWaveform {
        self.waveform
    }

    /// Set destination
    pub fn set_destination(&mut self, destination: LfoDestination) {
        self.destination = destination;
    }

    /// Get current destination
    pub fn destination(&self) -> LfoDestination {
        self.destination
    }

    /// Generate next modulation value (-1.0 to +1.0).
    /// Routing is handled by the mod matrix, so the LFO always runs once a
    /// nonzero depth is set.
    pub fn next_value(&mut self) -> f32 {
        if self.depth == 0.0 {
            return 0.0;
        }

        let raw_value = match self.waveform {
            LfoWaveform::Sine => self.next_sine(),
            LfoWaveform::Triangle => self.next_triangle(),
            LfoWaveform::Square => self.next_square(),
            LfoWaveform::Saw => self.next_saw(),
            LfoWaveform::Random => self.next_random(),
        };

        // Apply depth
        raw_value * self.depth
    }

    /// Generate next sine wave value (-1.0 to +1.0)
    fn next_sine(&mut self) -> f32 {
        let value = self.phase.sin();
        self.advance_phase();
        value
    }

    /// Generate next triangle wave value (-1.0 to +1.0)
    fn next_triangle(&mut self) -> f32 {
        // Convert phase to triangle: peak at π/2, valley at 3π/2
        let normalized_phase = self.phase / (2.0 * PI);
        let value = if normalized_phase < 0.5 {
            // Rising: 0 -> 1 over 0 to 0.5
            4.0 * normalized_phase - 1.0
        } else {
            // Falling: 1 -> -1 over 0.5 to 1.0
            3.0 - 4.0 * normalized_phase
        };
        self.advance_phase();
        value
    }

    /// Generate next square wave value (-1.0 to +1.0)
    fn next_square(&mut self) -> f32 {
        let value = if self.phase < PI { 1.0 } else { -1.0 };
        self.advance_phase();
        value
    }

    /// Generate next saw wave value (-1.0 to +1.0, falling)
    fn next_saw(&mut self) -> f32 {
        let value = 1.0 - 2.0 * self.phase / (2.0 * PI);
        self.advance_phase();
        value
    }

    /// Generate next random/S&H value (-1.0 to +1.0)
    fn next_random(&mut self) -> f32 {
        // Sample new value when phase wraps
        if self.phase < self.last_random_phase {
            // Phase wrapped, sample new value
            self.last_random_value = random_value();
        }
        self.last_random_phase = self.phase;
        self.advance_phase();
        self.last_random_value
    }

    /// Advance phase by one sample
    fn advance_phase(&mut self) {
        self.phase += self.phase_increment;
        if self.phase >= 2.0 * PI {
            self.phase -= 2.0 * PI;
        }
    }

    /// Update phase increment based on rate
    fn update_phase_increment(&mut self) {
        self.phase_increment = 2.0 * PI * self.rate / self.sample_rate as f32;
    }

    /// Reset LFO phase
    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.last_random_phase = 0.0;
        self.last_random_value = random_value();
    }

    /// Set sample rate
    pub fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
        self.update_phase_increment();
    }
}

/// Generate random value between -1.0 and 1.0
fn random_value() -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let mut hasher = DefaultHasher::new();
    now.hash(&mut hasher);
    let hash = hasher.finish();

    // Convert hash to f32 in range [-1.0, 1.0]
    (hash as f32 / u64::MAX as f32) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lfo_range() {
        let mut lfo = LFO::new(44100);
        lfo.set_depth(1.0);
        lfo.set_destination(LfoDestination::Pitch);

        for _ in 0..1000 {
            let value = lfo.next_value();
            assert!(
                value >= -1.0 && value <= 1.0,
                "LFO value out of range: {}",
                value
            );
        }
    }

    #[test]
    fn test_lfo_depth() {
        let mut lfo = LFO::new(44100);
        lfo.set_depth(0.5);
        lfo.set_destination(LfoDestination::Pitch);

        for _ in 0..1000 {
            let value = lfo.next_value();
            assert!(
                value >= -0.5 && value <= 0.5,
                "LFO value exceeds depth: {}",
                value
            );
        }
    }

    #[test]
    fn test_lfo_off() {
        let mut lfo = LFO::new(44100);
        lfo.set_destination(LfoDestination::Off);

        for _ in 0..100 {
            assert_eq!(lfo.next_value(), 0.0);
        }
    }

    #[test]
    fn test_lfo_rate() {
        let mut lfo = LFO::new(44100);
        lfo.set_rate(1.0); // 1 Hz = 44100 samples per cycle
        lfo.set_depth(1.0);
        lfo.set_destination(LfoDestination::Pitch);

        // Collect one cycle
        let mut zero_crossings = 0;
        let mut last_value = lfo.next_value();

        for _ in 0..50000 {
            let value = lfo.next_value();
            if last_value < 0.0 && value >= 0.0 {
                zero_crossings += 1;
            }
            last_value = value;
        }

        // Should have approximately 1 zero crossing (one full cycle)
        assert!(zero_crossings >= 1, "Expected at least 1 cycle at 1Hz");
    }
}
