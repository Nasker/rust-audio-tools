use serde::{Deserialize, Serialize};
#[cfg(feature = "fs")]
use std::fs;
#[cfg(feature = "fs")]
use std::path::Path;

use super::arp::ArpMode;
use super::effects::DistortionType;
use super::event::WaveformType;
use super::filter::FilterType;
use super::lfo::LfoDestination;
use super::lfo::LfoWaveform;
use super::mod_matrix::{ModDest, ModMatrix, ModSlot, ModSource, MOD_SLOTS};
use super::voice::PolyphonyMode;

/// A complete synthesizer preset/patch
/// Contains all parameters needed to recreate a sound
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Preset {
    pub name: String,
    pub version: String, // For migration if format changes

    /// Display category for the preset browser (e.g. "Bass", "Pad").
    #[serde(default = "default_category")]
    pub category: String,

    // Oscillator settings
    pub osc1_waveform: WaveformType,
    pub osc1_level: f32,
    pub osc1_phase: f32,

    pub osc2_waveform: WaveformType,
    pub osc2_level: f32,
    pub osc2_semitones: i8,
    pub osc2_cents: i8,
    pub osc2_phase: f32,

    pub osc3_waveform: WaveformType,
    pub osc3_level: f32,
    pub osc3_semitones: i8,
    pub osc3_cents: i8,
    pub osc3_phase: f32,

    // Oscillator 4 (FM/RM mod source; silent until level is raised)
    #[serde(default = "default_osc4_waveform")]
    pub osc4_waveform: WaveformType,
    #[serde(default)]
    pub osc4_level: f32,
    #[serde(default)]
    pub osc4_semitones: i8,
    #[serde(default)]
    pub osc4_cents: i8,
    #[serde(default)]
    pub osc4_phase: f32,

    // Pulse widths (Pulse waveform only; 0.5 keeps old presets unchanged)
    #[serde(default = "default_pw")]
    pub osc1_pw: f32,
    #[serde(default = "default_pw")]
    pub osc2_pw: f32,
    #[serde(default = "default_pw")]
    pub osc3_pw: f32,
    #[serde(default = "default_pw")]
    pub osc4_pw: f32,

    // Filter settings
    pub filter_cutoff: f32,
    pub filter_resonance: f32,

    // Second serial filter (wide-open SVF = transparent for old presets)
    #[serde(default = "default_filter2_cutoff")]
    pub filter2_cutoff: f32,
    #[serde(default)]
    pub filter2_resonance: f32,
    #[serde(default)]
    pub filter2_type: FilterType,

    // Amplitude envelope
    pub amp_attack: f32,
    pub amp_decay: f32,
    pub amp_sustain: f32,
    pub amp_release: f32,

    // Filter envelope
    pub filter_attack: f32,
    pub filter_decay: f32,
    pub filter_sustain: f32,
    pub filter_release: f32,
    pub filter_env_amount: f32,

    // LFO settings
    pub lfo_rate: f32,
    pub lfo_depth: f32,
    pub lfo_waveform: LfoWaveform,
    pub lfo_destination: LfoDestination,

    // Extra modulators
    #[serde(default)]
    pub lfo2_rate: f32,
    #[serde(default)]
    pub lfo2_depth: f32,
    #[serde(default)]
    pub lfo2_waveform: LfoWaveform,
    #[serde(default)]
    pub lfo3_rate: f32,
    #[serde(default)]
    pub lfo3_depth: f32,
    #[serde(default)]
    pub lfo3_waveform: LfoWaveform,
    #[serde(default)]
    pub env3_attack: f32,
    #[serde(default)]
    pub env3_decay: f32,
    #[serde(default)]
    pub env3_sustain: f32,
    #[serde(default)]
    pub env3_release: f32,
    /// Env4 ADSR: [attack, decay, sustain, release]
    #[serde(default = "default_env4")]
    pub env4: [f32; 4],

    // Modulation routing matrix (supersedes lfo_destination for new presets)
    #[serde(default)]
    pub mod_matrix: ModMatrix,

    // Tempo sync
    #[serde(default = "default_bpm")]
    pub tempo_bpm: f32,
    #[serde(default)]
    pub lfo1_sync: bool,
    #[serde(default = "default_beat")]
    pub lfo1_beats: f32,
    #[serde(default)]
    pub lfo2_sync: bool,
    #[serde(default = "default_beat")]
    pub lfo2_beats: f32,
    #[serde(default)]
    pub lfo3_sync: bool,
    #[serde(default = "default_beat")]
    pub lfo3_beats: f32,
    #[serde(default)]
    pub delay_sync: bool,
    #[serde(default = "default_beat")]
    pub delay_beats: f32,

    // Filter topology
    #[serde(default)]
    pub filter_type: FilterType,

    // Pitch bend
    pub pitch_bend_range: u8,

    // Portamento
    #[serde(default)]
    pub portamento_time: f32,

    // Stereo / Pan
    #[serde(default)]
    pub osc1_pan: f32,
    #[serde(default)]
    pub osc2_pan: f32,
    #[serde(default)]
    pub osc3_pan: f32,
    #[serde(default)]
    pub osc4_pan: f32,
    #[serde(default)]
    pub stereo_width: f32,

    // Polyphony mode
    #[serde(default)]
    pub polyphony_mode: PolyphonyMode,

    // Unison (monophonic mode): stack N voices on the held note with
    // symmetric detune + stereo spread for supersaw-style thickness.
    #[serde(default = "default_unison_voices")]
    pub unison_voices: u8,
    /// Max pitch deviation in cents at the unison edges (0 = no detune).
    #[serde(default = "default_unison_detune")]
    pub unison_detune: f32,
    /// Stereo spread 0..1 across unison voices.
    #[serde(default = "default_unison_spread")]
    pub unison_spread: f32,

    // Arpeggiator — a block-boundary note scheduler (no per-sample DSP cost).
    // Serde-defaulted off so all existing presets deserialize unchanged.
    #[serde(default)]
    pub arp_on: bool,
    #[serde(default)]
    pub arp_mode: ArpMode,
    /// Beats per arp step (0.25 = 1/16 note, 0.5 = 1/8, 1.0 = 1/4).
    #[serde(default = "default_arp_rate")]
    pub arp_rate: f32,
    /// Octave span of the arpeggiated sequence (1..=4).
    #[serde(default = "default_arp_oct")]
    pub arp_oct: u8,
    /// Latch mode: held notes keep arpeggiating after the keys release.
    #[serde(default)]
    pub arp_hold: bool,

    // Effects - Distortion
    #[serde(default)]
    pub distortion_enabled: bool,
    #[serde(default)]
    pub distortion_drive: f32,
    #[serde(default)]
    pub distortion_mix: f32,
    #[serde(default)]
    pub distortion_type: DistortionType,

    // Effects - Delay
    #[serde(default)]
    pub delay_enabled: bool,
    #[serde(default)]
    pub delay_time: f32,
    #[serde(default)]
    pub delay_feedback: f32,
    #[serde(default)]
    pub delay_mix: f32,

    // Effects - Reverb
    #[serde(default)]
    pub reverb_enabled: bool,
    #[serde(default)]
    pub reverb_room_size: f32,
    #[serde(default)]
    pub reverb_damping: f32,
    #[serde(default)]
    pub reverb_mix: f32,

    // Effects - Chorus
    #[serde(default)]
    pub chorus_enabled: bool,
    #[serde(default)]
    pub chorus_rate: f32,
    #[serde(default)]
    pub chorus_depth: f32,
    #[serde(default)]
    pub chorus_mix: f32,

    // Master
    pub master_volume: f32,
}

impl Default for Preset {
    fn default() -> Self {
        Self {
            name: "Init".to_string(),
            version: "1.0".to_string(),
            category: "Init".to_string(),

            // Default: Saw lead with slight detune
            osc1_waveform: WaveformType::Saw,
            osc1_level: 1.0,
            osc1_phase: 0.0,

            osc2_waveform: WaveformType::Saw,
            osc2_level: 0.8,
            osc2_semitones: 0,
            osc2_cents: 7,
            osc2_phase: 0.0,

            osc3_waveform: WaveformType::Square,
            osc3_level: 0.5,
            osc3_semitones: -12,
            osc3_cents: 0,
            osc3_phase: 0.0,

            osc4_waveform: WaveformType::Sine,
            osc4_level: 0.0,
            osc4_semitones: 0,
            osc4_cents: 0,
            osc4_phase: 0.0,

            // Pulse widths (50% = plain square)
            osc1_pw: 0.5,
            osc2_pw: 0.5,
            osc3_pw: 0.5,
            osc4_pw: 0.5,

            // Filter wide open
            filter_cutoff: 20000.0,
            filter_resonance: 0.0,
            // Filter 2 transparent (bypassed)
            filter2_cutoff: 20000.0,
            filter2_resonance: 0.0,
            filter2_type: FilterType::Svf,

            // Medium envelope
            amp_attack: 0.01,
            amp_decay: 0.1,
            amp_sustain: 0.7,
            amp_release: 0.2,

            // Filter envelope off
            filter_attack: 0.01,
            filter_decay: 0.3,
            filter_sustain: 0.0,
            filter_release: 0.3,
            filter_env_amount: 0.0,

            // LFO off
            lfo_rate: 6.0,
            lfo_depth: 0.0,
            lfo_waveform: LfoWaveform::Sine,
            lfo_destination: LfoDestination::Off,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,

            // Pitch bend
            pitch_bend_range: 12,

            // Portamento
            portamento_time: 0.001,

            // Stereo / Pan (default slight spread for fatness)
            osc1_pan: 0.0,
            osc2_pan: -0.3,
            osc3_pan: 0.3,
            osc4_pan: 0.0,
            stereo_width: 1.0,

            // Polyphony (default mono)
            polyphony_mode: PolyphonyMode::Mono,
            unison_voices: 1,
            unison_detune: 15.0,
            unison_spread: 0.7,

            // Effects - Delay (default off)
            distortion_enabled: false,
            distortion_drive: 0.0,
            distortion_mix: 0.5,
            distortion_type: DistortionType::Tube,
            delay_enabled: false,
            delay_time: 0.3,
            delay_feedback: 0.3,
            delay_mix: 0.3,

            // Effects - Reverb (default off)
            reverb_enabled: false,
            reverb_room_size: 0.5,
            reverb_damping: 0.5,
            reverb_mix: 0.3,

            // Effects - Chorus (default off)
            chorus_enabled: false,
            chorus_rate: 0.5,
            chorus_depth: 0.5,
            chorus_mix: 0.3,

            // Arpeggiator (default off)
            arp_on: false,
            arp_mode: ArpMode::Up,
            arp_rate: 0.25,
            arp_oct: 1,
            arp_hold: false,

            // Master
            master_volume: 0.5,
        }
    }
}

/// On-disk preset format version. v1 = bare `Preset` JSON; v2 = envelope.
pub const PRESET_FORMAT_VERSION: u32 = 2;

/// Versioned envelope written to disk since format v2.  The `preset` field
/// keeps the flat `Preset` shape so `serde(default)` inside it still gives
/// field-level forward/backward compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetFile {
    pub format_version: u32,
    pub preset: Preset,
}

impl PresetFile {
    pub fn new(preset: Preset) -> Self {
        Self {
            format_version: PRESET_FORMAT_VERSION,
            preset,
        }
    }
}

/// Parse a JSON string as a preset in either format:
/// v2 envelope (`{"format_version": 2, "preset": {...}}`) or v1 bare Preset.
fn parse_preset_json(json: &str) -> Result<Preset, PresetError> {
    // Peek at whether the document has a "preset" field (v2 envelope).
    // NB: kept `serde_json::Value`-free — its Deserialize impl emits a
    // `[2 x float]` const-pool entry the Xtensa LLVM backend cannot select
    // (rustc-LLVM ERROR: Cannot select XtensaISD::PCREL_WRAPPER).
    #[derive(Deserialize)]
    struct Probe {
        preset: Option<serde::de::IgnoredAny>,
    }
    let probe: Probe =
        serde_json::from_str(json).map_err(|e| PresetError::Deserialization(e.to_string()))?;
    if probe.preset.is_some() {
        let file: PresetFile = serde_json::from_str(json)
            .map_err(|e| PresetError::Deserialization(e.to_string()))?;
        Ok(file.preset)
    } else {
        // v1: bare Preset document — serde(default) fills fields added later.
        serde_json::from_str(json).map_err(|e| PresetError::Deserialization(e.to_string()))
    }
}

impl Preset {
    /// Create a new preset with the given name
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Save preset to a JSON file (format v2 envelope)
    #[cfg(feature = "fs")]
    pub fn save_to_file(&self, path: &Path) -> Result<(), PresetError> {
        let file = PresetFile::new(self.clone());
        let json = serde_json::to_string_pretty(&file)
            .map_err(|e| PresetError::Serialization(e.to_string()))?;
        fs::write(path, json).map_err(|e| PresetError::Io(e.to_string()))?;
        Ok(())
    }

    /// Load preset from a JSON file (accepts v1 bare and v2 envelope files)
    #[cfg(feature = "fs")]
    pub fn load_from_file(path: &Path) -> Result<Self, PresetError> {
        let json = fs::read_to_string(path).map_err(|e| PresetError::Io(e.to_string()))?;
        Self::from_json(&json)
    }

    /// Load preset from JSON string (accepts v1 bare and v2 envelope)
    pub fn from_json(json: &str) -> Result<Self, PresetError> {
        parse_preset_json(json)
    }

    /// Export preset to JSON string (format v2 envelope)
    pub fn to_json(&self) -> Result<String, PresetError> {
        serde_json::to_string_pretty(&PresetFile::new(self.clone()))
            .map_err(|e| PresetError::Serialization(e.to_string()))
    }
}

/// Error type for preset operations
#[derive(Debug, Clone, PartialEq)]
pub enum PresetError {
    Io(String),
    Serialization(String),
    Deserialization(String),
}

impl std::fmt::Display for PresetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PresetError::Io(msg) => write!(f, "IO error: {}", msg),
            PresetError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            PresetError::Deserialization(msg) => write!(f, "Deserialization error: {}", msg),
        }
    }
}

impl std::error::Error for PresetError {}

/// Get the default presets directory path
fn default_bpm() -> f32 {
    120.0
}

fn default_beat() -> f32 {
    1.0
}

fn default_env4() -> [f32; 4] {
    [0.01, 0.3, 0.0, 0.3]
}

fn default_unison_voices() -> u8 {
    1
}

fn default_unison_detune() -> f32 {
    15.0
}

fn default_unison_spread() -> f32 {
    0.7
}

fn default_arp_rate() -> f32 {
    0.25
}

fn default_arp_oct() -> u8 {
    1
}

fn default_osc4_waveform() -> WaveformType {
    WaveformType::Sine
}

fn default_filter2_cutoff() -> f32 {
    20000.0
}

fn default_pw() -> f32 {
    0.5
}

#[cfg(feature = "host-fs")]
pub fn default_presets_dir() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
        .join(".rustsynth")
        .join("presets")
}

/// Ensure the presets directory exists
#[cfg(feature = "host-fs")]
pub fn ensure_presets_dir() -> Result<std::path::PathBuf, PresetError> {
    let dir = default_presets_dir();
    fs::create_dir_all(&dir)
        .map_err(|e| PresetError::Io(format!("Failed to create presets dir: {}", e)))?;
    Ok(dir)
}

/// List all presets: factory presets first, then user presets from the
/// presets directory.  A user preset with the same name shadows the
/// factory one.
#[cfg(feature = "host-fs")]
pub fn list_presets() -> Result<Vec<String>, PresetError> {
    let dir = ensure_presets_dir()?;
    let mut presets = Vec::new();

    for entry in fs::read_dir(&dir)
        .map_err(|e| PresetError::Io(format!("Failed to read presets dir: {}", e)))?
    {
        let entry =
            entry.map_err(|e| PresetError::Io(format!("Failed to read directory entry: {}", e)))?;
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "json") {
            if let Some(stem) = path.file_stem() {
                presets.push(stem.to_string_lossy().to_string());
            }
        }
    }

    presets.sort();
    presets.dedup();
    Ok(presets)
}

/// Full library listing: factory names (marked with `is_factory`) merged
/// with user presets on disk.
#[cfg(feature = "host-fs")]
pub fn list_library() -> Vec<PresetEntry> {
    let factory = factory_presets();
    let factory_names: std::collections::HashSet<String> =
        factory.iter().map(|p| p.name.to_lowercase()).collect();
    let mut entries: Vec<PresetEntry> = factory
        .into_iter()
        .map(|p| PresetEntry {
            name: p.name,
            category: p.category,
            is_factory: true,
        })
        .collect();
    if let Ok(user) = list_presets() {
        for name in user {
            // On-disk copies of factory presets are listed once, as factory.
            if factory_names.contains(&name.to_lowercase()) {
                continue;
            }
            // Try to read the user file's category; fall back to "User".
            let category = default_presets_dir()
                .join(format!("{}.json", name))
                .exists()
                .then(|| {
                    Preset::load_from_file(&default_presets_dir().join(format!("{}.json", name)))
                })
                .and_then(|r| r.ok())
                .map(|p| p.category)
                .filter(|c| !c.is_empty() && c != "Uncategorized")
                .unwrap_or_else(|| "User".to_string());
            entries.push(PresetEntry {
                name,
                category,
                is_factory: false,
            });
        }
    }
    entries
}

/// One row in the preset library.
#[derive(Debug, Clone, PartialEq)]
pub struct PresetEntry {
    pub name: String,
    /// Category shown in the browser; empty string → "User"/"Uncategorized".
    pub category: String,
    pub is_factory: bool,
}

/// Fallback category for presets that don't specify one.
fn default_category() -> String {
    "Uncategorized".to_string()
}

/// Themed banks shown as shelves in the preset browser, in display order.
pub const PRESET_BANKS: [&str; 8] = ["Bass", "Lead", "Pad", "Keys", "Pluck", "Seq", "FX", "User"];

/// Load a preset by name.  Factory presets always resolve to the current
/// built-in version (so stale on-disk copies never shadow them); anything
/// else is read from the user presets directory.
#[cfg(feature = "host-fs")]
pub fn load_preset(name: &str) -> Result<Preset, PresetError> {
    if let Some(p) = factory_presets().into_iter().find(|p| p.name == name) {
        return Ok(p);
    }
    let path = default_presets_dir().join(format!("{}.json", name));
    if path.exists() {
        return Preset::load_from_file(&path);
    }
    Err(PresetError::Io(format!("Preset not found: {}", name)))
}

/// Save a preset by name (to default location)
#[cfg(feature = "host-fs")]
pub fn save_preset(preset: &Preset) -> Result<(), PresetError> {
    let dir = ensure_presets_dir()?;
    let path = dir.join(format!("{}.json", preset.name));
    preset.save_to_file(&path)
}

/// Build a `ModMatrix` from (source, dest, amount) triples — compact syntax
/// for factory preset literals. Disabled slots stay at the tail.
pub(crate) fn matrix(slots: &[(ModSource, ModDest, f32)]) -> ModMatrix {
    let mut m = ModMatrix::default();
    for (i, (source, dest, amount)) in slots.iter().enumerate().take(MOD_SLOTS) {
        m.slots[i] = ModSlot {
            source: *source,
            dest: *dest,
            amount: *amount,
            enabled: true,
        };
    }
    m
}

/// The full factory library: the original bank (refreshed for the newer
/// engine features) followed by the extended themed bank.
pub fn factory_presets() -> Vec<Preset> {
    let mut bank = original_factory_presets();
    super::factory_bank::renew(&mut bank);
    bank.extend(super::factory_bank::extended());
    // Leads and basses are monophonic instruments — voice them as such by
    // default regardless of source (original, renewed, or extended bank).
    // A preset can still opt out via the poly toggle after loading.
    for p in &mut bank {
        if p.category == "Bass" || p.category == "Lead" {
            p.polyphony_mode = PolyphonyMode::Mono;
        }
    }
    bank
}

/// The original factory bank, as first shipped.
fn original_factory_presets() -> Vec<Preset> {
    vec![
        // Classic sub bass - deep and powerful
        Preset {
            name: "Sub Bass".to_string(),
            category: "Bass".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Sine,
            osc1_level: 1.0,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Square,
            osc2_level: 0.3,
            osc2_semitones: 0,
            osc2_cents: 0,
            osc2_phase: 0.0,
            osc3_waveform: WaveformType::Sine,
            osc3_level: 0.8,
            osc3_semitones: -12,
            osc3_cents: 0,
            osc3_phase: 0.0,
            filter_cutoff: 800.0,
            filter_resonance: 0.2,
            amp_attack: 0.005,
            amp_decay: 0.2,
            amp_sustain: 0.9,
            amp_release: 0.15,
            filter_attack: 0.01,
            filter_decay: 0.3,
            filter_sustain: 0.3,
            filter_release: 0.2,
            filter_env_amount: 0.4,
            lfo_rate: 0.5,
            lfo_depth: 0.0,
            lfo_waveform: LfoWaveform::Sine,
            lfo_destination: LfoDestination::Off,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 2,
            portamento_time: 0.05,
            master_volume: 0.6,
            ..Default::default()
        },
        // Bright lead with portamento
        Preset {
            name: "Glide Lead".to_string(),
            category: "Lead".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Saw,
            osc1_level: 1.0,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Saw,
            osc2_level: 0.9,
            osc2_semitones: 0,
            osc2_cents: 12,
            osc2_phase: 0.25,
            osc3_waveform: WaveformType::Square,
            osc3_level: 0.4,
            osc3_semitones: 12,
            osc3_cents: 0,
            osc3_phase: 0.0,
            filter_cutoff: 3500.0,
            filter_resonance: 0.35,
            amp_attack: 0.01,
            amp_decay: 0.15,
            amp_sustain: 0.8,
            amp_release: 0.25,
            filter_attack: 0.01,
            filter_decay: 0.4,
            filter_sustain: 0.4,
            filter_release: 0.3,
            filter_env_amount: 0.6,
            lfo_rate: 5.5,
            lfo_depth: 0.15,
            lfo_waveform: LfoWaveform::Sine,
            lfo_destination: LfoDestination::Pitch,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 12,
            portamento_time: 0.12,
            master_volume: 0.5,
            ..Default::default()
        },
        // Lush pad with slow attack
        Preset {
            name: "Warm Pad".to_string(),
            category: "Pad".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Saw,
            osc1_level: 0.7,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Saw,
            osc2_level: 0.7,
            osc2_semitones: 0,
            osc2_cents: -8,
            osc2_phase: 0.33,
            osc3_waveform: WaveformType::Triangle,
            osc3_level: 0.5,
            osc3_semitones: -12,
            osc3_cents: 5,
            osc3_phase: 0.66,
            filter_cutoff: 2000.0,
            filter_resonance: 0.15,
            amp_attack: 0.8,
            amp_decay: 0.5,
            amp_sustain: 0.7,
            amp_release: 1.2,
            filter_attack: 1.0,
            filter_decay: 0.8,
            filter_sustain: 0.5,
            filter_release: 1.5,
            filter_env_amount: 0.3,
            lfo_rate: 0.8,
            lfo_depth: 0.1,
            lfo_waveform: LfoWaveform::Triangle,
            lfo_destination: LfoDestination::FilterCutoff,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 7,
            portamento_time: 0.0,
            master_volume: 0.45,
            ..Default::default()
        },
        // Punchy pluck
        Preset {
            name: "Pluck".to_string(),
            category: "Keys".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Saw,
            osc1_level: 1.0,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Square,
            osc2_level: 0.6,
            osc2_semitones: 0,
            osc2_cents: 3,
            osc2_phase: 0.0,
            osc3_waveform: WaveformType::Triangle,
            osc3_level: 0.3,
            osc3_semitones: 12,
            osc3_cents: 0,
            osc3_phase: 0.0,
            filter_cutoff: 5000.0,
            filter_resonance: 0.4,
            amp_attack: 0.001,
            amp_decay: 0.3,
            amp_sustain: 0.0,
            amp_release: 0.2,
            filter_attack: 0.001,
            filter_decay: 0.25,
            filter_sustain: 0.1,
            filter_release: 0.15,
            filter_env_amount: 0.8,
            lfo_rate: 6.0,
            lfo_depth: 0.0,
            lfo_waveform: LfoWaveform::Sine,
            lfo_destination: LfoDestination::Off,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 12,
            portamento_time: 0.0,
            master_volume: 0.5,
            ..Default::default()
        },
        // Noise sweep FX
        Preset {
            name: "Sweep FX".to_string(),
            category: "FX".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Noise,
            osc1_level: 0.8,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Saw,
            osc2_level: 0.4,
            osc2_semitones: 0,
            osc2_cents: 0,
            osc2_phase: 0.0,
            osc3_waveform: WaveformType::Square,
            osc3_level: 0.3,
            osc3_semitones: 7,
            osc3_cents: 0,
            osc3_phase: 0.0,
            filter_cutoff: 500.0,
            filter_resonance: 0.7,
            amp_attack: 0.5,
            amp_decay: 1.0,
            amp_sustain: 0.3,
            amp_release: 2.0,
            filter_attack: 0.8,
            filter_decay: 1.5,
            filter_sustain: 0.2,
            filter_release: 2.5,
            filter_env_amount: 0.9,
            lfo_rate: 0.3,
            lfo_depth: 0.5,
            lfo_waveform: LfoWaveform::Triangle,
            lfo_destination: LfoDestination::FilterCutoff,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 24,
            portamento_time: 0.0,
            master_volume: 0.4,
            ..Default::default()
        },
        // Classic Minimoog-style bass
        Preset {
            name: "Moog Bass".to_string(),
            category: "Bass".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Saw,
            osc1_level: 1.0,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Square,
            osc2_level: 0.7,
            osc2_semitones: 0,
            osc2_cents: -5,
            osc2_phase: 0.0,
            osc3_waveform: WaveformType::Saw,
            osc3_level: 0.6,
            osc3_semitones: -12,
            osc3_cents: 0,
            osc3_phase: 0.0,
            filter_cutoff: 600.0,
            filter_resonance: 0.5,
            amp_attack: 0.002,
            amp_decay: 0.4,
            amp_sustain: 0.6,
            amp_release: 0.1,
            filter_attack: 0.001,
            filter_decay: 0.35,
            filter_sustain: 0.2,
            filter_release: 0.1,
            filter_env_amount: 0.7,
            lfo_rate: 6.0,
            lfo_depth: 0.0,
            lfo_waveform: LfoWaveform::Sine,
            lfo_destination: LfoDestination::Off,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 2,
            portamento_time: 0.03,
            master_volume: 0.55,
            ..Default::default()
        },
        // Wobble bass (dubstep style)
        Preset {
            name: "Wobble".to_string(),
            category: "Bass".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Saw,
            osc1_level: 1.0,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Square,
            osc2_level: 0.8,
            osc2_semitones: 0,
            osc2_cents: 0,
            osc2_phase: 0.5,
            osc3_waveform: WaveformType::Saw,
            osc3_level: 0.7,
            osc3_semitones: -12,
            osc3_cents: 0,
            osc3_phase: 0.0,
            filter_cutoff: 400.0,
            filter_resonance: 0.6,
            amp_attack: 0.01,
            amp_decay: 0.1,
            amp_sustain: 0.9,
            amp_release: 0.15,
            filter_attack: 0.01,
            filter_decay: 0.1,
            filter_sustain: 0.8,
            filter_release: 0.1,
            filter_env_amount: 0.2,
            lfo_rate: 4.0,
            lfo_depth: 0.9,
            lfo_waveform: LfoWaveform::Square,
            lfo_destination: LfoDestination::FilterCutoff,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 2,
            portamento_time: 0.0,
            master_volume: 0.5,
            ..Default::default()
        },
        // Soft keys / electric piano style
        Preset {
            name: "Soft Keys".to_string(),
            category: "Keys".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Sine,
            osc1_level: 1.0,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Triangle,
            osc2_level: 0.5,
            osc2_semitones: 0,
            osc2_cents: 2,
            osc2_phase: 0.0,
            osc3_waveform: WaveformType::Sine,
            osc3_level: 0.3,
            osc3_semitones: 12,
            osc3_cents: 0,
            osc3_phase: 0.0,
            filter_cutoff: 4000.0,
            filter_resonance: 0.1,
            amp_attack: 0.005,
            amp_decay: 0.8,
            amp_sustain: 0.4,
            amp_release: 0.5,
            filter_attack: 0.001,
            filter_decay: 0.5,
            filter_sustain: 0.3,
            filter_release: 0.4,
            filter_env_amount: 0.4,
            lfo_rate: 5.0,
            lfo_depth: 0.05,
            lfo_waveform: LfoWaveform::Sine,
            lfo_destination: LfoDestination::Amplitude,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 2,
            portamento_time: 0.0,
            master_volume: 0.5,
            ..Default::default()
        },
        // Wide stereo pad - lush and spacious with full stereo spread
        Preset {
            name: "Wide Pad".to_string(),
            category: "Pad".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Saw,
            osc1_level: 0.8,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Square,
            osc2_level: 0.6,
            osc2_semitones: 7,
            osc2_cents: 5,
            osc2_phase: 0.5,
            osc3_waveform: WaveformType::Saw,
            osc3_level: 0.5,
            osc3_semitones: -5,
            osc3_cents: -3,
            osc3_phase: 0.0,
            filter_cutoff: 2500.0,
            filter_resonance: 0.4,
            amp_attack: 0.3,
            amp_decay: 0.5,
            amp_sustain: 0.75,
            amp_release: 1.2,
            filter_attack: 0.4,
            filter_decay: 0.6,
            filter_sustain: 0.6,
            filter_release: 1.0,
            filter_env_amount: 0.3,
            lfo_rate: 0.4,
            lfo_depth: 0.2,
            lfo_waveform: LfoWaveform::Triangle,
            lfo_destination: LfoDestination::FilterCutoff,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 12,
            portamento_time: 0.15,
            master_volume: 0.55,
            osc1_pan: -0.8,
            osc2_pan: 0.0,
            osc3_pan: 0.8,
            stereo_width: 1.8,
            polyphony_mode: PolyphonyMode::Poly,
            distortion_enabled: false,
            distortion_drive: 0.0,
            distortion_mix: 0.5,
            delay_enabled: true,
            delay_time: 0.375,
            delay_feedback: 0.35,
            delay_mix: 0.25,
            reverb_enabled: true,
            reverb_room_size: 0.75,
            reverb_damping: 0.4,
            reverb_mix: 0.4,
            chorus_enabled: true,
            chorus_rate: 0.6,
            chorus_depth: 0.5,
            chorus_mix: 0.35,
            ..Default::default()
        },
        // Polyphonic keys - electric piano style with chorus and delay
        Preset {
            name: "Digital Keys".to_string(),
            category: "Keys".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Sine,
            osc1_level: 1.0,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Triangle,
            osc2_level: 0.4,
            osc2_semitones: 0,
            osc2_cents: 7,
            osc2_phase: 0.0,
            osc3_waveform: WaveformType::Saw,
            osc3_level: 0.2,
            osc3_semitones: 12,
            osc3_cents: 0,
            osc3_phase: 0.5,
            filter_cutoff: 6000.0,
            filter_resonance: 0.15,
            amp_attack: 0.001,
            amp_decay: 1.2,
            amp_sustain: 0.2,
            amp_release: 0.8,
            filter_attack: 0.01,
            filter_decay: 0.4,
            filter_sustain: 0.0,
            filter_release: 0.3,
            filter_env_amount: 0.6,
            lfo_rate: 4.5,
            lfo_depth: 0.03,
            lfo_waveform: LfoWaveform::Sine,
            lfo_destination: LfoDestination::Pitch,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 2,
            portamento_time: 0.001,
            master_volume: 0.65,
            osc1_pan: 0.0,
            osc2_pan: -0.4,
            osc3_pan: 0.4,
            stereo_width: 1.2,
            polyphony_mode: PolyphonyMode::Poly,
            distortion_enabled: false,
            distortion_drive: 0.0,
            distortion_mix: 0.5,
            delay_enabled: true,
            delay_time: 0.25,
            delay_feedback: 0.3,
            delay_mix: 0.2,
            reverb_enabled: false,
            reverb_room_size: 0.5,
            reverb_damping: 0.5,
            reverb_mix: 0.0,
            chorus_enabled: true,
            chorus_rate: 1.2,
            chorus_depth: 0.4,
            chorus_mix: 0.45,
            ..Default::default()
        },
        // Space drone - ambient and evolving with heavy reverb
        Preset {
            name: "Space Drone".to_string(),
            category: "FX".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Square,
            osc1_level: 0.9,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Saw,
            osc2_level: 0.7,
            osc2_semitones: -7,
            osc2_cents: 11,
            osc2_phase: 0.25,
            osc3_waveform: WaveformType::Sine,
            osc3_level: 0.6,
            osc3_semitones: 7,
            osc3_cents: -7,
            osc3_phase: 0.5,
            filter_cutoff: 1200.0,
            filter_resonance: 0.55,
            amp_attack: 1.5,
            amp_decay: 2.0,
            amp_sustain: 1.0,
            amp_release: 3.0,
            filter_attack: 2.0,
            filter_decay: 1.5,
            filter_sustain: 0.8,
            filter_release: 2.5,
            filter_env_amount: 0.5,
            lfo_rate: 0.15,
            lfo_depth: 0.6,
            lfo_waveform: LfoWaveform::Sine,
            lfo_destination: LfoDestination::FilterCutoff,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 24,
            portamento_time: 0.8,
            master_volume: 0.5,
            osc1_pan: -0.6,
            osc2_pan: 0.0,
            osc3_pan: 0.6,
            stereo_width: 2.0,
            polyphony_mode: PolyphonyMode::Poly,
            distortion_enabled: false,
            distortion_drive: 0.0,
            distortion_mix: 0.5,
            delay_enabled: true,
            delay_time: 0.5,
            delay_feedback: 0.5,
            delay_mix: 0.3,
            reverb_enabled: true,
            reverb_room_size: 0.95,
            reverb_damping: 0.2,
            reverb_mix: 0.6,
            chorus_enabled: true,
            chorus_rate: 0.3,
            chorus_depth: 0.8,
            chorus_mix: 0.5,
            ..Default::default()
        },
        // Pumping bass - mono with chorus for thickness
        Preset {
            name: "Pumping Bass".to_string(),
            category: "Bass".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Saw,
            osc1_level: 1.0,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Square,
            osc2_level: 0.6,
            osc2_semitones: 0,
            osc2_cents: 8,
            osc2_phase: 0.0,
            osc3_waveform: WaveformType::Saw,
            osc3_level: 0.4,
            osc3_semitones: -12,
            osc3_cents: 0,
            osc3_phase: 0.5,
            filter_cutoff: 1800.0,
            filter_resonance: 0.3,
            amp_attack: 0.001,
            amp_decay: 0.25,
            amp_sustain: 0.4,
            amp_release: 0.15,
            filter_attack: 0.005,
            filter_decay: 0.2,
            filter_sustain: 0.2,
            filter_release: 0.15,
            filter_env_amount: 0.7,
            lfo_rate: 8.0,
            lfo_depth: 0.0,
            lfo_waveform: LfoWaveform::Sine,
            lfo_destination: LfoDestination::Off,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 2,
            portamento_time: 0.001,
            master_volume: 0.7,
            osc1_pan: 0.0,
            osc2_pan: -0.15,
            osc3_pan: 0.15,
            stereo_width: 0.6,
            polyphony_mode: PolyphonyMode::Mono,
            distortion_enabled: false,
            distortion_drive: 0.0,
            distortion_mix: 0.5,
            delay_enabled: false,
            delay_time: 0.3,
            delay_feedback: 0.2,
            delay_mix: 0.0,
            reverb_enabled: false,
            reverb_room_size: 0.5,
            reverb_damping: 0.5,
            reverb_mix: 0.0,
            chorus_enabled: true,
            chorus_rate: 2.5,
            chorus_depth: 0.3,
            chorus_mix: 0.25,
            ..Default::default()
        },
        // Ethereal bells - uses delay and stereo width for shimmer
        Preset {
            name: "Shimmer Bells".to_string(),
            category: "Keys".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Sine,
            osc1_level: 0.9,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Sine,
            osc2_level: 0.6,
            osc2_semitones: 7,
            osc2_cents: 0,
            osc2_phase: 0.33,
            osc3_waveform: WaveformType::Triangle,
            osc3_level: 0.4,
            osc3_semitones: 12,
            osc3_cents: 5,
            osc3_phase: 0.66,
            filter_cutoff: 8000.0,
            filter_resonance: 0.2,
            amp_attack: 0.002,
            amp_decay: 1.5,
            amp_sustain: 0.1,
            amp_release: 2.0,
            filter_attack: 0.01,
            filter_decay: 0.8,
            filter_sustain: 0.0,
            filter_release: 0.6,
            filter_env_amount: 0.8,
            lfo_rate: 6.5,
            lfo_depth: 0.1,
            lfo_waveform: LfoWaveform::Sine,
            lfo_destination: LfoDestination::Amplitude,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 7,
            portamento_time: 0.02,
            master_volume: 0.6,
            osc1_pan: -0.5,
            osc2_pan: 0.0,
            osc3_pan: 0.5,
            stereo_width: 1.5,
            polyphony_mode: PolyphonyMode::Poly,
            distortion_enabled: false,
            distortion_drive: 0.0,
            distortion_mix: 0.5,
            delay_enabled: true,
            delay_time: 0.187,
            delay_feedback: 0.45,
            delay_mix: 0.35,
            reverb_enabled: true,
            reverb_room_size: 0.8,
            reverb_damping: 0.5,
            reverb_mix: 0.35,
            chorus_enabled: false,
            chorus_rate: 0.5,
            chorus_depth: 0.5,
            chorus_mix: 0.0,
            ..Default::default()
        },
        // Acid line - classic TB-303 style with portamento
        Preset {
            name: "Acid Line".to_string(),
            category: "Bass".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Saw,
            osc1_level: 1.0,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Square,
            osc2_level: 0.0,
            osc2_semitones: 0,
            osc2_cents: 0,
            osc2_phase: 0.0,
            osc3_waveform: WaveformType::Square,
            osc3_level: 0.0,
            osc3_semitones: 0,
            osc3_cents: 0,
            osc3_phase: 0.0,
            filter_cutoff: 450.0,
            filter_resonance: 0.75,
            amp_attack: 0.001,
            amp_decay: 0.4,
            amp_sustain: 0.3,
            amp_release: 0.25,
            filter_attack: 0.005,
            filter_decay: 0.25,
            filter_sustain: 0.1,
            filter_release: 0.2,
            filter_env_amount: 0.85,
            lfo_rate: 5.0,
            lfo_depth: 0.0,
            lfo_waveform: LfoWaveform::Sine,
            lfo_destination: LfoDestination::Off,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 12,
            portamento_time: 0.08,
            master_volume: 0.7,
            osc1_pan: 0.0,
            osc2_pan: 0.0,
            osc3_pan: 0.0,
            stereo_width: 1.0,
            polyphony_mode: PolyphonyMode::Mono,
            distortion_enabled: false,
            distortion_drive: 0.0,
            distortion_mix: 0.5,
            delay_enabled: true,
            delay_time: 0.125,
            delay_feedback: 0.25,
            delay_mix: 0.15,
            reverb_enabled: false,
            reverb_room_size: 0.5,
            reverb_damping: 0.5,
            reverb_mix: 0.0,
            chorus_enabled: false,
            chorus_rate: 0.5,
            chorus_depth: 0.5,
            chorus_mix: 0.0,
            ..Default::default()
        },
        // Dream sequence - arpeggio-friendly with delay and chorus
        Preset {
            name: "Dream Seq".to_string(),
            category: "FX".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Square,
            osc1_level: 0.8,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Saw,
            osc2_level: 0.7,
            osc2_semitones: 12,
            osc2_cents: 0,
            osc2_phase: 0.5,
            osc3_waveform: WaveformType::Sine,
            osc3_level: 0.5,
            osc3_semitones: 24,
            osc3_cents: 0,
            osc3_phase: 0.0,
            filter_cutoff: 3500.0,
            filter_resonance: 0.45,
            amp_attack: 0.01,
            amp_decay: 0.35,
            amp_sustain: 0.3,
            amp_release: 0.4,
            filter_attack: 0.02,
            filter_decay: 0.3,
            filter_sustain: 0.2,
            filter_release: 0.35,
            filter_env_amount: 0.5,
            lfo_rate: 3.5,
            lfo_depth: 0.25,
            lfo_waveform: LfoWaveform::Triangle,
            lfo_destination: LfoDestination::Pitch,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: ModMatrix::default(),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 12,
            portamento_time: 0.04,
            master_volume: 0.6,
            osc1_pan: -0.3,
            osc2_pan: 0.0,
            osc3_pan: 0.3,
            stereo_width: 1.4,
            polyphony_mode: PolyphonyMode::Poly,
            distortion_enabled: false,
            distortion_drive: 0.0,
            distortion_mix: 0.5,
            delay_enabled: true,
            delay_time: 0.333,
            delay_feedback: 0.4,
            delay_mix: 0.3,
            reverb_enabled: true,
            reverb_room_size: 0.7,
            reverb_damping: 0.45,
            reverb_mix: 0.3,
            chorus_enabled: true,
            chorus_rate: 1.8,
            chorus_depth: 0.35,
            chorus_mix: 0.4,
            ..Default::default()
        },
        // ── Matrix showcase bank ────────────────────────────────────────────
        // These presets route LFO2/LFO3/Env3/mod-wheel through the modulation
        // matrix, demonstrating tempo sync, mod-on-mod, and the new filters.

        // Growl: tempo-sync'd LFO1 on the comb filter, Env3 pushes resonance
        // and drive makes it dirty.
        Preset {
            name: "Growl Bass".to_string(),
            category: "Bass".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Saw,
            osc1_level: 1.0,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Square,
            osc2_level: 0.7,
            osc2_semitones: -12,
            osc2_cents: 0,
            osc2_phase: 0.0,
            osc3_waveform: WaveformType::Sine,
            osc3_level: 0.6,
            osc3_semitones: -12,
            osc3_cents: 0,
            osc3_phase: 0.0,
            filter_cutoff: 900.0,
            filter_resonance: 0.4,
            amp_attack: 0.005,
            amp_decay: 0.25,
            amp_sustain: 0.75,
            amp_release: 0.12,
            filter_attack: 0.01,
            filter_decay: 0.2,
            filter_sustain: 0.3,
            filter_release: 0.15,
            filter_env_amount: 0.6,
            lfo_rate: 6.0,
            lfo_depth: 0.0,
            lfo_waveform: LfoWaveform::Saw,
            lfo_destination: LfoDestination::Off,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.4,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: matrix(&[
                (ModSource::Lfo1, ModDest::FilterCutoff, 0.55),
                (ModSource::Env3, ModDest::FilterResonance, 0.35),
                (ModSource::ModWheel, ModDest::Lfo1Rate, 0.8),
            ]),
            tempo_bpm: 120.0,
            lfo1_sync: true,
            lfo1_beats: 0.5,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Comb,
            pitch_bend_range: 12,
            portamento_time: 0.0,
            master_volume: 0.55,
            osc1_pan: 0.0,
            osc2_pan: -0.2,
            osc3_pan: 0.2,
            stereo_width: 1.2,
            polyphony_mode: PolyphonyMode::Mono,
            distortion_enabled: true,
            distortion_drive: 0.45,
            distortion_mix: 0.6,
            delay_enabled: false,
            delay_time: 0.3,
            delay_feedback: 0.3,
            delay_mix: 0.0,
            reverb_enabled: false,
            reverb_room_size: 0.4,
            reverb_damping: 0.5,
            reverb_mix: 0.0,
            chorus_enabled: false,
            chorus_rate: 1.0,
            chorus_depth: 0.3,
            chorus_mix: 0.0,
            ..Default::default()
        },
        // Velocity Lead: velocity opens the ladder filter and the mod wheel
        // adds LFO2 vibrato — expressive, playable matrix routing.
        Preset {
            name: "Velocity Lead".to_string(),
            category: "Lead".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Saw,
            osc1_level: 0.9,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Saw,
            osc2_level: 0.9,
            osc2_semitones: 0,
            osc2_cents: 5,
            osc2_phase: 0.0,
            osc3_waveform: WaveformType::Saw,
            osc3_level: 0.4,
            osc3_semitones: 12,
            osc3_cents: -4,
            osc3_phase: 0.0,
            filter_cutoff: 4500.0,
            filter_resonance: 0.25,
            amp_attack: 0.02,
            amp_decay: 0.3,
            amp_sustain: 0.7,
            amp_release: 0.25,
            filter_attack: 0.02,
            filter_decay: 0.35,
            filter_sustain: 0.4,
            filter_release: 0.3,
            filter_env_amount: 0.3,
            lfo_rate: 5.5,
            lfo_depth: 0.0,
            lfo_waveform: LfoWaveform::Sine,
            lfo_destination: LfoDestination::Off,
            lfo2_rate: 5.5,
            lfo2_depth: 1.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: matrix(&[
                (ModSource::Velocity, ModDest::FilterCutoff, 0.5),
                (ModSource::Lfo2, ModDest::AllPitch, 0.0),
                (ModSource::ModWheel, ModDest::Lfo2Depth, 1.0),
            ]),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Ladder,
            pitch_bend_range: 12,
            portamento_time: 0.03,
            master_volume: 0.55,
            osc1_pan: -0.1,
            osc2_pan: 0.1,
            osc3_pan: 0.0,
            stereo_width: 1.1,
            polyphony_mode: PolyphonyMode::Mono,
            distortion_enabled: true,
            distortion_drive: 0.2,
            distortion_mix: 0.35,
            delay_enabled: true,
            delay_time: 0.375,
            delay_feedback: 0.3,
            delay_mix: 0.2,
            reverb_enabled: true,
            reverb_room_size: 0.4,
            reverb_damping: 0.5,
            reverb_mix: 0.15,
            chorus_enabled: false,
            chorus_rate: 1.0,
            chorus_depth: 0.3,
            chorus_mix: 0.0,
            ..Default::default()
        },
        // Pulse Gate Pad: Env3 gates the amplitude while synced LFOs drift the
        // cutoff and widen the stereo field.
        Preset {
            name: "Pulse Gate Pad".to_string(),
            category: "Pad".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Saw,
            osc1_level: 0.7,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Saw,
            osc2_level: 0.7,
            osc2_semitones: 0,
            osc2_cents: 9,
            osc2_phase: 0.0,
            osc3_waveform: WaveformType::Triangle,
            osc3_level: 0.5,
            osc3_semitones: -12,
            osc3_cents: 0,
            osc3_phase: 0.0,
            filter_cutoff: 2200.0,
            filter_resonance: 0.35,
            amp_attack: 0.08,
            amp_decay: 0.4,
            amp_sustain: 0.8,
            amp_release: 0.6,
            filter_attack: 0.3,
            filter_decay: 0.6,
            filter_sustain: 0.5,
            filter_release: 0.5,
            filter_env_amount: 0.25,
            lfo_rate: 6.0,
            lfo_depth: 0.0,
            lfo_waveform: LfoWaveform::Sine,
            lfo_destination: LfoDestination::Off,
            lfo2_rate: 6.0,
            lfo2_depth: 1.0,
            lfo2_waveform: LfoWaveform::Triangle,
            lfo3_rate: 6.0,
            lfo3_depth: 1.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.005,
            env3_decay: 0.25,
            env3_sustain: 0.0,
            env3_release: 0.1,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: matrix(&[
                (ModSource::Env3, ModDest::Amplitude, 0.6),
                (ModSource::Lfo2, ModDest::FilterCutoff, 0.35),
                (ModSource::Lfo3, ModDest::StereoWidth, 0.7),
            ]),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: true,
            lfo2_beats: 1.0,
            lfo3_sync: true,
            lfo3_beats: 4.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 12,
            portamento_time: 0.0,
            master_volume: 0.5,
            osc1_pan: -0.4,
            osc2_pan: 0.4,
            osc3_pan: 0.0,
            stereo_width: 1.6,
            polyphony_mode: PolyphonyMode::Poly,
            distortion_enabled: false,
            distortion_drive: 0.0,
            distortion_mix: 0.5,
            delay_enabled: true,
            delay_time: 0.5,
            delay_feedback: 0.35,
            delay_mix: 0.2,
            reverb_enabled: true,
            reverb_room_size: 0.75,
            reverb_damping: 0.4,
            reverb_mix: 0.35,
            chorus_enabled: true,
            chorus_rate: 0.8,
            chorus_depth: 0.4,
            chorus_mix: 0.3,
            ..Default::default()
        },
        // Riser: LFO3 ramps pitch up while mod-on-mod speeds LFO2's cutoff
        // sweep — a classic build-up transition.
        Preset {
            name: "Riser".to_string(),
            category: "FX".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Saw,
            osc1_level: 0.8,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Saw,
            osc2_level: 0.8,
            osc2_semitones: 7,
            osc2_cents: 0,
            osc2_phase: 0.0,
            osc3_waveform: WaveformType::Sine,
            osc3_level: 0.4,
            osc3_semitones: 12,
            osc3_cents: 0,
            osc3_phase: 0.0,
            filter_cutoff: 1200.0,
            filter_resonance: 0.5,
            amp_attack: 0.02,
            amp_decay: 0.5,
            amp_sustain: 1.0,
            amp_release: 0.4,
            filter_attack: 1.5,
            filter_decay: 0.8,
            filter_sustain: 0.9,
            filter_release: 0.4,
            filter_env_amount: 0.7,
            lfo_rate: 6.0,
            lfo_depth: 0.0,
            lfo_waveform: LfoWaveform::Sine,
            lfo_destination: LfoDestination::Off,
            lfo2_rate: 6.0,
            lfo2_depth: 1.0,
            lfo2_waveform: LfoWaveform::Saw,
            lfo3_rate: 6.0,
            lfo3_depth: 1.0,
            lfo3_waveform: LfoWaveform::Saw,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: matrix(&[
                (ModSource::Lfo2, ModDest::FilterCutoff, 0.6),
                (ModSource::Lfo3, ModDest::AllPitch, 0.25),
                (ModSource::Lfo3, ModDest::Lfo2Rate, 0.7),
            ]),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: true,
            lfo2_beats: 2.0,
            lfo3_sync: true,
            lfo3_beats: 8.0,
            delay_sync: true,
            delay_beats: 0.5,
            filter_type: FilterType::Svf,
            pitch_bend_range: 12,
            portamento_time: 0.0,
            master_volume: 0.5,
            osc1_pan: -0.3,
            osc2_pan: 0.3,
            osc3_pan: 0.0,
            stereo_width: 1.5,
            polyphony_mode: PolyphonyMode::Poly,
            distortion_enabled: false,
            distortion_drive: 0.0,
            distortion_mix: 0.5,
            delay_enabled: true,
            delay_time: 0.25,
            delay_feedback: 0.45,
            delay_mix: 0.3,
            reverb_enabled: true,
            reverb_room_size: 0.85,
            reverb_damping: 0.35,
            reverb_mix: 0.45,
            chorus_enabled: false,
            chorus_rate: 1.0,
            chorus_depth: 0.3,
            chorus_mix: 0.0,
            ..Default::default()
        },
        // Aftertouch Keys: soft keys that open up with aftertouch — a per-voice
        // destination that only the matrix can express.
        Preset {
            name: "Aftertouch Keys".to_string(),
            category: "Keys".to_string(),
            version: "1.0".to_string(),
            osc1_waveform: WaveformType::Triangle,
            osc1_level: 0.9,
            osc1_phase: 0.0,
            osc2_waveform: WaveformType::Sine,
            osc2_level: 0.6,
            osc2_semitones: 12,
            osc2_cents: 0,
            osc2_phase: 0.0,
            osc3_waveform: WaveformType::Sine,
            osc3_level: 0.3,
            osc3_semitones: 24,
            osc3_cents: 0,
            osc3_phase: 0.0,
            filter_cutoff: 2800.0,
            filter_resonance: 0.15,
            amp_attack: 0.03,
            amp_decay: 0.5,
            amp_sustain: 0.5,
            amp_release: 0.35,
            filter_attack: 0.05,
            filter_decay: 0.4,
            filter_sustain: 0.4,
            filter_release: 0.3,
            filter_env_amount: 0.2,
            lfo_rate: 4.0,
            lfo_depth: 0.0,
            lfo_waveform: LfoWaveform::Sine,
            lfo_destination: LfoDestination::Off,
            lfo2_rate: 6.0,
            lfo2_depth: 0.0,
            lfo2_waveform: LfoWaveform::Sine,
            lfo3_rate: 6.0,
            lfo3_depth: 0.0,
            lfo3_waveform: LfoWaveform::Sine,
            env3_attack: 0.01,
            env3_decay: 0.3,
            env3_sustain: 0.0,
            env3_release: 0.3,
            env4: [0.01, 0.3, 0.0, 0.3],
            mod_matrix: matrix(&[
                (ModSource::Aftertouch, ModDest::FilterCutoff, 0.45),
                (ModSource::Aftertouch, ModDest::Osc3Level, 0.5),
                (ModSource::Velocity, ModDest::Amplitude, 0.35),
            ]),
            tempo_bpm: 120.0,
            lfo1_sync: false,
            lfo1_beats: 1.0,
            lfo2_sync: false,
            lfo2_beats: 1.0,
            lfo3_sync: false,
            lfo3_beats: 1.0,
            delay_sync: false,
            delay_beats: 1.0,
            filter_type: FilterType::Svf,
            pitch_bend_range: 12,
            portamento_time: 0.0,
            master_volume: 0.55,
            osc1_pan: 0.0,
            osc2_pan: -0.15,
            osc3_pan: 0.15,
            stereo_width: 1.2,
            polyphony_mode: PolyphonyMode::Poly,
            distortion_enabled: false,
            distortion_drive: 0.0,
            distortion_mix: 0.5,
            delay_enabled: true,
            delay_time: 0.4,
            delay_feedback: 0.25,
            delay_mix: 0.15,
            reverb_enabled: true,
            reverb_room_size: 0.6,
            reverb_damping: 0.5,
            reverb_mix: 0.25,
            chorus_enabled: true,
            chorus_rate: 1.2,
            chorus_depth: 0.35,
            chorus_mix: 0.3,
            ..Default::default()
        },
    ]
}

/// Write the factory bank to the presets directory, installing missing
/// files and renewing on-disk copies that are out of date. Returns how many
/// files were written.
#[cfg(feature = "host-fs")]
pub fn install_factory_presets() -> Result<usize, PresetError> {
    let dir = ensure_presets_dir()?;
    let mut written = 0;

    for preset in factory_presets() {
        let path = dir.join(format!("{}.json", preset.name));
        let up_to_date =
            path.exists() && Preset::load_from_file(&path).map_or(false, |p| p == preset);
        if !up_to_date {
            preset.save_to_file(&path)?;
            written += 1;
        }
    }

    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "fs")]
    use tempfile::NamedTempFile;

    #[test]
    fn test_preset_default() {
        let preset = Preset::default();
        assert_eq!(preset.name, "Init");
        assert_eq!(preset.osc1_waveform, WaveformType::Saw);
        assert_eq!(preset.amp_attack, 0.01);
    }

    #[test]
    fn test_preset_serialize_deserialize() {
        let preset = Preset::new("Test Patch");
        let json = preset.to_json().unwrap();
        let loaded = Preset::from_json(&json).unwrap();
        assert_eq!(preset, loaded);
    }

    #[test]
    #[cfg(feature = "fs")]
    fn test_preset_save_load_file() {
        let preset = Preset::new("File Test");

        // Create a temp file
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        // Save
        preset.save_to_file(&path).unwrap();

        // Load
        let loaded = Preset::load_from_file(&path).unwrap();
        assert_eq!(preset, loaded);
    }

    /// Regression: presets saved before Osc4/Filter2/DistortionType existed
    /// must load with safe defaults (silent osc4, transparent filter2, Tube).
    #[test]
    fn test_old_preset_json_defaults_new_fields() {
        let mut v = serde_json::to_value(Preset::default()).unwrap();
        let obj = v.as_object_mut().unwrap();
        for key in [
            "osc4_waveform",
            "osc4_level",
            "osc4_semitones",
            "osc4_cents",
            "osc4_phase",
            "osc4_pan",
            "filter2_cutoff",
            "filter2_resonance",
            "filter2_type",
            "distortion_type",
        ] {
            obj.remove(key);
        }
        let p: Preset = serde_json::from_value(v).unwrap();
        assert_eq!(p.osc4_waveform, WaveformType::Sine);
        assert_eq!(p.osc4_level, 0.0);
        assert_eq!(p.osc4_pan, 0.0);
        assert_eq!(p.filter2_cutoff, 20000.0);
        assert_eq!(p.filter2_resonance, 0.0);
        assert_eq!(p.filter2_type, FilterType::Svf);
        assert_eq!(p.distortion_type, DistortionType::Tube);
    }

    #[test]
    fn test_preset_version() {
        let preset = Preset::default();
        assert_eq!(preset.version, "1.0");
    }

    #[test]
    fn test_v2_roundtrip() {
        let preset = Preset::default();
        let json = preset.to_json().unwrap();
        // Saved as an envelope
        assert!(json.contains("\"format_version\""));
        assert!(json.contains("\"preset\""));
        let loaded = Preset::from_json(&json).unwrap();
        assert_eq!(loaded.name, preset.name);
        assert_eq!(loaded.filter_cutoff, preset.filter_cutoff);
    }

    #[test]
    fn test_v1_bare_preset_still_loads() {
        // A v1 document is just the bare Preset struct, possibly missing
        // newer fields entirely — serde(default) fills them.
        let json = r#"{
            "name": "Old Pad",
            "version": "1.0",
            "osc1_waveform": "Sine",
            "osc1_level": 1.0,
            "osc1_phase": 0.0,
            "osc2_waveform": "Saw",
            "osc2_level": 0.0,
            "osc2_semitones": 0,
            "osc2_cents": 0,
            "osc2_phase": 0.0,
            "osc3_waveform": "Sine",
            "osc3_level": 0.0,
            "osc3_semitones": 0,
            "osc3_cents": 0,
            "osc3_phase": 0.0,
            "filter_cutoff": 5000.0,
            "filter_resonance": 0.3,
            "amp_attack": 0.01,
            "amp_decay": 0.1,
            "amp_sustain": 0.7,
            "amp_release": 0.2,
            "filter_attack": 0.01,
            "filter_decay": 0.3,
            "filter_sustain": 0.0,
            "filter_release": 0.3,
            "filter_env_amount": 0.0,
            "lfo_rate": 6.0,
            "lfo_depth": 0.0,
            "lfo_waveform": "Sine",
            "lfo_destination": "Off",
            "pitch_bend_range": 12,
            "master_volume": 0.5,
            "delay_enabled": false,
            "delay_time": 0.3,
            "delay_feedback": 0.3,
            "delay_mix": 0.3,
            "reverb_enabled": false,
            "reverb_room_size": 0.5,
            "reverb_damping": 0.5,
            "reverb_mix": 0.3,
            "chorus_enabled": false,
            "chorus_rate": 0.5,
            "chorus_depth": 0.5,
            "chorus_mix": 0.3
        }"#;
        let p = Preset::from_json(json).unwrap();
        assert_eq!(p.name, "Old Pad");
        assert_eq!(p.tempo_bpm, 120.0); // defaulted
        assert!(!p.lfo1_sync);
        assert_eq!(p.delay_beats, 1.0);
    }

    #[test]
    #[cfg(feature = "host-fs")]
    fn test_factory_preset_loadable_by_name() {
        // Factory presets must be loadable even without a file on disk.
        let p = load_preset("Sub Bass").unwrap();
        assert_eq!(p.name, "Sub Bass");
        assert!(load_preset("Does Not Exist 42").is_err());
    }
}
