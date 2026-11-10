//! Voice-related schema: polyphony mode + legacy preset migration helpers.
//! `Voice`/`VoiceManager` DSP stays in the host `voice` module.

use serde::{Deserialize, Serialize};

use crate::lfo::LfoDestination;
use crate::mod_matrix::{ModDest, ModSlot, ModSource};
use crate::presets::Preset;

/// Polyphony mode for the voice manager
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolyphonyMode {
    /// Single voice with key stacking (returns to previous note)
    Mono,
    /// Multiple independent voices for chords
    Poly,
}

impl Default for PolyphonyMode {
    fn default() -> Self {
        PolyphonyMode::Mono
    }
}

/// Convert a legacy `lfo_destination` preset field into an equivalent
/// matrix slot, preserving the old modulation depth semantics.
pub fn legacy_lfo_slot(preset: &Preset) -> Option<ModSlot> {
    if preset.lfo_depth <= 0.0 {
        return None;
    }
    let (dest, amount) = match preset.lfo_destination {
        // Old vibrato: ±1 semitone at depth 1.0 -> depth already scales the
        // LFO output, so amount maps to the 12-semitone destination range.
        LfoDestination::Pitch => (ModDest::AllPitch, preset.lfo_depth / 12.0),
        // Old filter LFO: ±2 octaves at full depth (depth was applied twice).
        LfoDestination::FilterCutoff => (ModDest::FilterCutoff, preset.lfo_depth * 0.4),
        // Old tremolo: ±50% amplitude at depth 1.0.
        LfoDestination::Amplitude => (ModDest::Amplitude, preset.lfo_depth * 0.5),
        LfoDestination::Off => return None,
    };
    Some(ModSlot {
        source: ModSource::Lfo1,
        dest,
        amount,
        enabled: true,
    })
}
