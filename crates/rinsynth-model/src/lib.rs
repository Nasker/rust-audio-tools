//! rinsynth-model — the portable model of RustInSynth.
//!
//! Contains the parameter specification, preset schema, modulation matrix,
//! events, and factory bank data. This crate holds *no* DSP and *no*
//! host-side persistence by default; it is the vocabulary shared between the
//! desktop host/plugin and the ESP32-S3 firmware port.
//!
//! Features:
//! - `fs`      — `Preset::save_to_file`/`load_from_file` against an explicit
//!               path (usable on any `std` target, incl. ESP-IDF VFS).
//! - `host-fs` — `fs` + desktop conveniences (`dirs`-based preset directory,
//!               `load_preset`/`save_preset`/`list_*`/`install_factory_presets`).

pub mod arp;
pub mod effects;
pub mod event;
pub mod factory_bank;
pub mod filter;
pub mod lfo;
pub mod mod_matrix;
pub mod param_spec;
pub mod params;
pub mod presets;
pub mod types;
pub mod voice;

pub use arp::ArpMode;
pub use effects::DistortionType;
pub use event::{
    NoteEvent, NoteEventKind, SynthEvent, SynthEventKind, SynthEventReceiver, WaveformType,
};
pub use filter::FilterType;
pub use lfo::{LfoDestination, LfoWaveform, LFO};
pub use mod_matrix::{ModDest, ModMatrix, ModSlot, ModSource, ModSourceValues, MOD_SLOTS};
pub use params::{
    cc, cc_to_cents, cc_to_filter_env_amount, cc_to_level, cc_to_lfo_depth, cc_to_lfo_destination,
    cc_to_lfo_rate, cc_to_lfo_waveform, cc_to_phase, cc_to_pitch_bend_range, cc_to_semitones,
    cc_to_sustain, cc_to_time, cc_to_waveform, CCMapping, SynthParam,
};
pub use presets::{Preset, PresetEntry, PresetError, PresetFile, PRESET_BANKS};
#[cfg(feature = "host-fs")]
pub use presets::{
    default_presets_dir, ensure_presets_dir, install_factory_presets, list_library, list_presets,
    load_preset, save_preset,
};
pub use types::{Amplitude, Frequency, Sample, SampleRate, StereoSample};
pub use voice::{legacy_lfo_slot, PolyphonyMode};
