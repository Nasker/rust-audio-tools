//! Schema types for the effects chain — parameters/enums only.
//! The DSP implementations stay in the host `effects` module.

use serde::{Deserialize, Serialize};

/// Distortion waveshaper topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistortionType {
    /// Warm tanh soft-clip, level-compensated.
    Tube,
    /// Aggressive hard clip with a slight asymmetric bias.
    Fuzz,
    /// Sample-hold decimation + bit-depth quantizer (lo-fi crunch).
    Bitcrush,
    /// Sine wavefolder: folds peaks back on themselves at high drive.
    Fold,
}

impl DistortionType {
    pub const ALL: [DistortionType; 4] = [
        DistortionType::Tube,
        DistortionType::Fuzz,
        DistortionType::Bitcrush,
        DistortionType::Fold,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            DistortionType::Tube => "Tube",
            DistortionType::Fuzz => "Fuzz",
            DistortionType::Bitcrush => "Bitcrush",
            DistortionType::Fold => "Fold",
        }
    }

    pub fn from_index(i: u8) -> Self {
        match i {
            1 => DistortionType::Fuzz,
            2 => DistortionType::Bitcrush,
            3 => DistortionType::Fold,
            _ => DistortionType::Tube,
        }
    }

    pub fn index(self) -> u8 {
        match self {
            DistortionType::Tube => 0,
            DistortionType::Fuzz => 1,
            DistortionType::Bitcrush => 2,
            DistortionType::Fold => 3,
        }
    }
}

impl Default for DistortionType {
    fn default() -> Self {
        DistortionType::Tube
    }
}
