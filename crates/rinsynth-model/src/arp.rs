use serde::{Deserialize, Serialize};

/// Arpeggiator pattern direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ArpMode {
    /// Held notes played low -> high, repeating.
    #[default]
    Up,
    /// Held notes played high -> low, repeating.
    Down,
    /// Held notes ping-ponged low -> high -> low (endpoints not repeated).
    UpDown,
    /// Random pick from the expanded note sequence each step.
    Random,
}

impl ArpMode {
    pub fn from_index(index: u8) -> Self {
        match index % 4 {
            0 => ArpMode::Up,
            1 => ArpMode::Down,
            2 => ArpMode::UpDown,
            _ => ArpMode::Random,
        }
    }

    pub fn index(&self) -> u8 {
        match self {
            ArpMode::Up => 0,
            ArpMode::Down => 1,
            ArpMode::UpDown => 2,
            ArpMode::Random => 3,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ArpMode::Up => "Up",
            ArpMode::Down => "Down",
            ArpMode::UpDown => "Up/Down",
            ArpMode::Random => "Random",
        }
    }
}
