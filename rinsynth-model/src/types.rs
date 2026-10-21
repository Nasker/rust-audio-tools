/// A single audio sample value, typically in range [-1.0, 1.0]
pub type Sample = f32;

/// A stereo audio sample (left, right)
#[derive(Debug, Clone, Copy, Default)]
pub struct StereoSample {
    pub left: Sample,
    pub right: Sample,
}

impl StereoSample {
    pub const ZERO: StereoSample = StereoSample {
        left: 0.0,
        right: 0.0,
    };

    pub fn new(left: Sample, right: Sample) -> Self {
        Self { left, right }
    }

    /// Create stereo from mono with equal power panning
    /// pan: -1.0 = full left, 0.0 = center, 1.0 = full right
    pub fn from_mono_panned(sample: Sample, pan: f32) -> Self {
        // Equal power panning using sine/cosine
        let pan_normalized = (pan + 1.0) * 0.5; // 0.0 to 1.0
        let angle = pan_normalized * std::f32::consts::FRAC_PI_2;
        Self {
            left: sample * angle.cos(),
            right: sample * angle.sin(),
        }
    }

    /// Create stereo from mono (centered)
    pub fn from_mono(sample: Sample) -> Self {
        Self {
            left: sample,
            right: sample,
        }
    }

    /// Mix two stereo samples
    pub fn mix(self, other: StereoSample) -> Self {
        Self {
            left: self.left + other.left,
            right: self.right + other.right,
        }
    }

    /// Scale by amplitude
    pub fn scale(self, amp: f32) -> Self {
        Self {
            left: self.left * amp,
            right: self.right * amp,
        }
    }

    /// Apply stereo width (0.0 = mono, 1.0 = normal, 2.0 = extra wide)
    pub fn with_width(self, width: f32) -> Self {
        let mid = (self.left + self.right) * 0.5;
        let side = (self.left - self.right) * 0.5;
        Self {
            left: mid + side * width,
            right: mid - side * width,
        }
    }
}

impl std::ops::Add for StereoSample {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            left: self.left + other.left,
            right: self.right + other.right,
        }
    }
}

impl std::ops::AddAssign for StereoSample {
    fn add_assign(&mut self, other: Self) {
        self.left += other.left;
        self.right += other.right;
    }
}

impl std::ops::Mul<f32> for StereoSample {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Self {
            left: self.left * scalar,
            right: self.right * scalar,
        }
    }
}

/// Sample rate in Hz (e.g., 44100, 48000)
pub type SampleRate = u32;

/// Frequency in Hz
pub type Frequency = f32;

/// Amplitude/volume level, typically in range [0.0, 1.0]
pub type Amplitude = f32;

/// MIDI note number (0-127)
pub type MidiNote = u8;

/// Converts a MIDI note number to frequency in Hz
/// A4 (note 69) = 440 Hz
pub fn midi_to_frequency(note: MidiNote) -> Frequency {
    440.0 * 2.0_f32.powf((note as f32 - 69.0) / 12.0)
}

/// Converts frequency in Hz to the nearest MIDI note number
pub fn frequency_to_midi(freq: Frequency) -> MidiNote {
    (69.0 + 12.0 * (freq / 440.0).log2()).round() as MidiNote
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a4_conversion() {
        let freq = midi_to_frequency(69);
        assert!((freq - 440.0).abs() < 0.01);
    }

    #[test]
    fn test_frequency_to_midi() {
        assert_eq!(frequency_to_midi(440.0), 69);
    }
}
