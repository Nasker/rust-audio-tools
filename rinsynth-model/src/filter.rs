//! Schema types for filters — topology selection only.
//! The DSP implementations stay in the host `filter` module.

/// Selects which filter topology a voice uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum FilterType {
    #[default]
    Svf,
    Ladder,
    Comb,
}

impl FilterType {
    pub fn name(&self) -> &'static str {
        match self {
            FilterType::Svf => "SVF",
            FilterType::Ladder => "Ladder",
            FilterType::Comb => "Comb",
        }
    }
    pub fn from_index(i: u8) -> Self {
        match i {
            1 => FilterType::Ladder,
            2 => FilterType::Comb,
            _ => FilterType::Svf,
        }
    }
    pub fn index(&self) -> u8 {
        match self {
            FilterType::Svf => 0,
            FilterType::Ladder => 1,
            FilterType::Comb => 2,
        }
    }
    pub const ALL: [FilterType; 3] = [FilterType::Svf, FilterType::Ladder, FilterType::Comb];
}
