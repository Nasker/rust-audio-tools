//! Tempo-synced arpeggiator — a block-boundary note scheduler.
//!
//! This is deliberately **not DSP**: it tracks held notes and emits
//! [`NoteEvent`]s on a schedule derived from `bpm × rate_beats`, the same
//! mechanism a demo riff uses. Per-sample cost is zero — the host calls
//! [`Arpeggiator::tick`] once per audio block and forwards the emitted
//! events into the voice manager.
//!
//! Held-note tracking means NoteOn/NoteOff semantics "just work": keys
//! pressed while the arp is on are latched into the pattern, keys released
//! leave it, and the sequence re-derives on the next step. Works in poly
//! or mono (mono + arp = classic bass lines — the engine's legato
//! retrigger handles the off→on ordering within a block).
//!
//! Held notes are stored sorted by pitch (insertion-sorted, ≤16 entries —
//! deterministic, no alloc, RT-safe).

use rinsynth_model::arp::ArpMode;
use rinsynth_model::event::NoteEvent;
use rinsynth_model::presets::Preset;
use rinsynth_model::types::MidiNote;

/// Max simultaneously held notes tracked by the arp (same order as the
/// voice pool — nobody holds 16 keys on a mono synth UI).
pub const MAX_HELD: usize = 16;

/// Note gate length as a fraction of the step (0.8 = slight separation
/// between hits — classic arp feel; 1.0 would run notes into each other).
const GATE: f32 = 0.8;

/// Chord-capture window for HOLD mode, in seconds. With hold engaged,
/// notes arriving within this span of the previous NoteOn merge into one
/// chord; a NoteOn after the gap (or after every key was released)
/// *replaces* the latched chord instead of piling onto it — the arp
/// always walks the last chord played, never an endless accumulation.
const CHORD_WINDOW_S: f32 = 1.5;

pub struct Arpeggiator {
    enabled: bool,
    mode: ArpMode,
    /// Beats per step (0.25 = 1/16, 0.5 = 1/8, 1.0 = 1/4).
    rate_beats: f32,
    /// Octave span applied upward over the held chord (1..=4).
    octaves: u8,
    /// Latch mode: held notes are never released by NoteOff — the chord
    /// keeps arpeggiating after the keys are let go. Releasing hold drops
    /// the latched chord (and silences the sounding step note).
    hold: bool,
    /// Held (note, velocity) pairs sorted ascending by note.
    held: [(MidiNote, f32); MAX_HELD],
    held_len: usize,
    /// Physically pressed keys right now (independent of the latch).
    /// Reaching 0 commits the chord — the next NoteOn starts a new one.
    keys_down: u8,
    /// Synth-rate samples elapsed since the last NoteOn (chord window).
    chord_age: f32,
    /// Sample rate from the last [`tick`](Self::tick) — converts
    /// `CHORD_WINDOW_S` to samples in `note_on` (which has no clock).
    sample_rate: f32,
    /// Expanded sequence length = held_len × octaves.
    seq_len: usize,
    /// Current position in the expanded sequence.
    step_idx: usize,
    /// UpDown bounce direction.
    updown_fwd: bool,
    /// xorshift32 state for Random mode.
    rng: u32,
    /// Samples elapsed within the current step.
    phase: f32,
    /// Note currently sounding under the arp (for the gate NoteOff).
    playing: Option<MidiNote>,
    /// NoteOff that must be emitted on the next tick (arp disabled or
    /// last held key released while a note was sounding).
    off_pending: Option<MidiNote>,
    /// Set when the first key lands so step 0 fires on the next tick.
    trigger_now: bool,
}

impl Default for Arpeggiator {
    fn default() -> Self {
        Self::new()
    }
}

impl Arpeggiator {
    pub fn new() -> Self {
        Self {
            enabled: false,
            mode: ArpMode::Up,
            rate_beats: 0.25,
            octaves: 1,
            hold: false,
            held: [(0, 0.0); MAX_HELD],
            held_len: 0,
            keys_down: 0,
            chord_age: f32::MAX,
            sample_rate: 44_100.0,
            seq_len: 0,
            step_idx: 0,
            updown_fwd: true,
            rng: 0x9E3779B9,
            phase: 0.0,
            playing: None,
            off_pending: None,
            trigger_now: false,
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn mode(&self) -> ArpMode {
        self.mode
    }

    pub fn rate_beats(&self) -> f32 {
        self.rate_beats
    }

    pub fn octaves(&self) -> u8 {
        self.octaves
    }

    /// Apply the arp fields of a preset (called right after
    /// `EmbeddedEngine::apply_preset` — keeps arp state in the patch).
    ///
    /// **Preset-agnostic by design**: only a preset that explicitly ships
    /// `arp_on = true` applies arp settings. Presets can arm/configure the
    /// arp, but never force it off or stomp the player's live settings —
    /// the arp survives preset browsing.
    pub fn apply_preset(&mut self, p: &Preset) {
        if !p.arp_on {
            return;
        }
        self.set_mode(p.arp_mode);
        self.set_rate_beats(p.arp_rate);
        self.set_octaves(p.arp_oct);
        self.set_hold(p.arp_hold);
        self.set_enabled(true);
    }

    pub fn hold_enabled(&self) -> bool {
        self.hold
    }

    /// Latch mode. Engaging keeps the current held chord; releasing drops
    /// every latched note (silencing the sounding step note) so the next
    /// keypress starts a fresh chord — classic arp-hold semantics.
    pub fn set_hold(&mut self, hold: bool) {
        self.hold = hold;
        if !hold {
            self.held_len = 0;
            self.seq_len = 0;
            self.step_idx = 0;
            self.trigger_now = false;
            if let Some(n) = self.playing.take() {
                self.off_pending = Some(n);
            }
        }
    }

    pub fn set_enabled(&mut self, on: bool) {
        if self.enabled && !on {
            // Turning off: stop the sounding arp note, drop the chord.
            if let Some(n) = self.playing.take() {
                self.off_pending = Some(n);
            }
            self.held_len = 0;
            self.seq_len = 0;
            self.step_idx = 0;
            self.trigger_now = false;
            self.keys_down = 0;
            self.chord_age = f32::MAX;
        }
        self.enabled = on;
    }

    pub fn set_mode(&mut self, mode: ArpMode) {
        self.mode = mode;
        self.step_idx = 0;
        self.updown_fwd = true;
    }

    pub fn set_rate_beats(&mut self, beats: f32) {
        self.rate_beats = beats.clamp(0.0625, 4.0);
    }

    pub fn set_octaves(&mut self, oct: u8) {
        self.octaves = oct.clamp(1, 4);
        self.seq_len = self.held_len * self.octaves as usize;
        if self.step_idx >= self.seq_len {
            self.step_idx = 0;
        }
    }

    /// Feed a NoteOn. Called only when enabled (the host passes events
    /// straight to the engine when the arp is off).
    pub fn note_on(&mut self, note: MidiNote, velocity: f32) {
        self.keys_down = self.keys_down.saturating_add(1);
        let chord_expired = self.chord_age > CHORD_WINDOW_S * self.sample_rate;
        if self.hold && self.held_len > 0 && (self.keys_down == 1 || chord_expired) {
            // Last-chord semantics: all keys were released (or the capture
            // window lapsed) since the latched chord — start a fresh chord
            // instead of accumulating. Silence the sounding step note so
            // its voice can't leak when `playing` is overwritten below.
            self.held_len = 0;
            self.seq_len = 0;
            self.step_idx = 0;
            self.updown_fwd = true;
            if let Some(n) = self.playing.take() {
                self.off_pending = Some(n);
            }
        }
        self.chord_age = 0.0;
        if self.held_len >= MAX_HELD {
            return;
        }
        // Replace velocity if the note is already held (retrigger).
        for i in 0..self.held_len {
            if self.held[i].0 == note {
                self.held[i].1 = velocity;
                return;
            }
        }
        // Insertion sort by pitch — classic arps walk the held chord in
        // note order regardless of the order keys were struck.
        let mut i = self.held_len;
        self.held[i] = (note, velocity);
        self.held_len += 1;
        while i > 0 && self.held[i].0 < self.held[i - 1].0 {
            self.held.swap(i, i - 1);
            i -= 1;
        }
        self.seq_len = self.held_len * self.octaves as usize;
        if self.held_len == 1 {
            // First key of a new chord: fire step 0 on the next tick.
            self.step_idx = 0;
            self.updown_fwd = true;
            self.trigger_now = true;
        }
    }

    /// Feed a NoteOff. Returns `Some(NoteEvent)` to forward to the engine
    /// when the note was **not** part of the held chord — e.g. a note
    /// struck before the arp was enabled must still release its voice.
    pub fn note_off(&mut self, note: MidiNote) -> Option<NoteEvent> {
        self.keys_down = self.keys_down.saturating_sub(1);
        for i in 0..self.held_len {
            if self.held[i].0 == note {
                if self.hold {
                    // Latch: the key physically released but the note stays
                    // in the pattern until hold is switched off.
                    return None;
                }
                self.held.copy_within(i + 1..self.held_len, i);
                self.held_len -= 1;
                self.seq_len = self.held_len * self.octaves as usize;
                if self.step_idx >= self.seq_len {
                    self.step_idx = 0;
                }
                if self.held_len == 0 {
                    // Chord released: silence the sounding arp note.
                    if let Some(p) = self.playing.take() {
                        self.off_pending = Some(p);
                    }
                    self.trigger_now = false;
                }
                return None;
            }
        }
        // Not our note — pass it through to the voice manager.
        Some(NoteEvent::note_off(note))
    }

    /// Advance the scheduler by `n_samples` of the *synth-rate* clock and
    /// emit any `NoteEvent`s whose boundaries fall inside this block.
    /// `bpm` is the live tempo (may change via MIDI clock or UI).
    pub fn tick(
        &mut self,
        n_samples: u32,
        bpm: f32,
        sample_rate: f32,
        emit: &mut impl FnMut(NoteEvent),
    ) {
        self.sample_rate = sample_rate;
        self.chord_age += n_samples as f32;
        if let Some(n) = self.off_pending.take() {
            emit(NoteEvent::note_off(n));
        }
        if !self.enabled || self.held_len == 0 {
            return;
        }

        let step_samples =
            ((60.0 / bpm.max(20.0)) * self.rate_beats * sample_rate).max(32.0);
        let gate_samples = step_samples * GATE;

        if self.trigger_now {
            self.trigger_now = false;
            self.phase = 0.0;
            self.emit_step(emit);
        }

        self.phase += n_samples as f32;

        // Crossed one (or pathologically more) step boundaries: release the
        // old note then fire the next step. Gate-offs also land here when a
        // whole step elapsed — the emitted order stays off→on either way.
        while self.phase >= step_samples {
            self.phase -= step_samples;
            if let Some(n) = self.playing.take() {
                emit(NoteEvent::note_off(n));
            }
            self.advance();
            self.emit_step(emit);
        }

        // Gate boundary crossed inside the step: end the note early so
        // hits stay separated.
        if let Some(n) = self.playing {
            if self.phase >= gate_samples {
                self.playing = None;
                emit(NoteEvent::note_off(n));
            }
        }
    }

    /// Note number for the current step: held-chord index + octave offset.
    fn step_note(&self) -> (MidiNote, f32) {
        let base = self.step_idx % self.held_len;
        let oct = self.step_idx / self.held_len;
        let (note, vel) = self.held[base];
        (note.saturating_add(12 * oct as u8).min(127), vel)
    }

    fn emit_step(&mut self, emit: &mut impl FnMut(NoteEvent)) {
        if self.seq_len == 0 {
            return;
        }
        if self.step_idx >= self.seq_len {
            self.step_idx = 0;
        }
        let (note, vel) = self.step_note();
        emit(NoteEvent::note_on(note, vel));
        self.playing = Some(note);
    }

    /// Move `step_idx` to the next position per the current mode.
    fn advance(&mut self) {
        if self.seq_len == 0 {
            return;
        }
        match self.mode {
            ArpMode::Up => self.step_idx = (self.step_idx + 1) % self.seq_len,
            ArpMode::Down => {
                self.step_idx = if self.step_idx == 0 {
                    self.seq_len - 1
                } else {
                    self.step_idx - 1
                };
            }
            ArpMode::UpDown => {
                if self.seq_len <= 1 {
                    self.step_idx = 0;
                } else if self.updown_fwd {
                    self.step_idx += 1;
                    if self.step_idx >= self.seq_len {
                        self.step_idx = self.seq_len - 2;
                        self.updown_fwd = false;
                    }
                } else if self.step_idx == 0 {
                    self.step_idx = 1;
                    self.updown_fwd = true;
                } else {
                    self.step_idx -= 1;
                }
            }
            ArpMode::Random => {
                // xorshift32 — deterministic, no seed table needed.
                self.rng ^= self.rng << 13;
                self.rng ^= self.rng >> 17;
                self.rng ^= self.rng << 5;
                let idx = (self.rng as usize) % self.seq_len;
                // Avoid an immediate repeat when there's more than one
                // choice — it reads as a stuck note, not randomness.
                self.step_idx = if self.seq_len > 1 && idx == self.step_idx {
                    (idx + 1) % self.seq_len
                } else {
                    idx
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 44_100.0;

    fn collect(arp: &mut Arpeggiator, n: u32, bpm: f32) -> Vec<NoteEvent> {
        let mut out = Vec::new();
        arp.tick(n, bpm, SR, &mut |e| out.push(e));
        out
    }

    fn ons(evs: &[NoteEvent]) -> Vec<u8> {
        evs.iter()
            .filter(|e| e.kind == rinsynth_model::event::SynthEventKind::NoteOn)
            .map(|e| e.note)
            .collect()
    }

    #[test]
    fn up_mode_cycles_held_chord() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.set_rate_beats(0.25); // 1/16 @120bpm = 5512 samples
        a.note_on(48, 1.0);
        a.note_on(60, 1.0);
        a.note_on(64, 1.0);

        // First tick fires step 0 immediately.
        let mut notes = ons(&collect(&mut a, 256, 120.0));
        for _ in 0..40 {
            notes.extend(ons(&collect(&mut a, 2205, 120.0)));
        }
        // Expect the repeating pattern 48,60,64,48,60,64...
        assert!(notes.len() >= 6, "got {:?}", notes);
        for (i, &n) in notes.iter().take(6).enumerate() {
            assert_eq!(n, [48, 60, 64][i % 3]);
        }
    }

    #[test]
    fn down_mode_reverses() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.set_mode(ArpMode::Down);
        a.set_rate_beats(0.25);
        a.note_on(48, 1.0);
        a.note_on(60, 1.0);
        let mut notes = ons(&collect(&mut a, 100, 120.0));
        for _ in 0..10 {
            notes.extend(ons(&collect(&mut a, 5512, 120.0)));
        }
        // First hit is index 0 (48), then walks down and wraps.
        assert_eq!(notes[..4], [48, 60, 48, 60]);
    }

    #[test]
    fn updown_bounces_without_repeating_ends() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.set_mode(ArpMode::UpDown);
        a.set_rate_beats(0.25);
        a.note_on(48, 1.0);
        a.note_on(60, 1.0);
        a.note_on(64, 1.0);
        let mut notes = ons(&collect(&mut a, 100, 120.0));
        for _ in 0..10 {
            notes.extend(ons(&collect(&mut a, 5512, 120.0)));
        }
        assert_eq!(notes[..6], [48, 60, 64, 60, 48, 60]);
    }

    #[test]
    fn octaves_expand_sequence_upward() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.set_octaves(2);
        a.set_rate_beats(0.25);
        a.note_on(48, 1.0);
        a.note_on(60, 1.0);
        let mut notes = ons(&collect(&mut a, 100, 120.0));
        for _ in 0..10 {
            notes.extend(ons(&collect(&mut a, 5512, 120.0)));
        }
        // Expanded seq = chord + chord+12: [48,60, 48+12=60, 60+12=72] —
        // 48+12 legitimately lands on the held 60 (no dedup, like real arps).
        assert_eq!(notes[..5], [48, 60, 60, 72, 48]);
    }

    #[test]
    fn gate_emits_note_off_before_next_step() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.set_rate_beats(0.25);
        a.note_on(60, 1.0);
        // 1/16 @120bpm ≈ 5513 samples; gate off at ~4410.
        let evs = collect(&mut a, 5513, 120.0);
        let offs = evs
            .iter()
            .filter(|e| e.kind == rinsynth_model::event::SynthEventKind::NoteOff)
            .count();
        assert!(offs >= 1);
    }

    #[test]
    fn disabling_releases_playing_note() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.note_on(60, 1.0);
        let _ = collect(&mut a, 100, 120.0);
        a.set_enabled(false);
        let evs = collect(&mut a, 100, 120.0);
        assert!(evs
            .iter()
            .any(|e| e.kind == rinsynth_model::event::SynthEventKind::NoteOff && e.note == 60));
    }

    #[test]
    fn releasing_all_keys_stops_sequence() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.note_on(60, 1.0);
        let _ = collect(&mut a, 100, 120.0);
        assert!(a.note_off(60).is_none()); // was held -> not forwarded
        let evs = collect(&mut a, 6000, 120.0);
        assert!(evs
            .iter()
            .all(|e| e.kind == rinsynth_model::event::SynthEventKind::NoteOff));
        // And stays silent afterwards.
        assert!(collect(&mut a, 6000, 120.0).is_empty());
    }

    #[test]
    fn note_off_of_untracked_note_passes_through() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.note_on(60, 1.0);
        // Note 72 was struck before the arp engaged — its release must be
        // forwarded or the engine voice would hang forever.
        assert!(a.note_off(72).is_some());
    }

    #[test]
    fn hold_latches_chord_after_release() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.set_hold(true);
        a.set_rate_beats(0.25);
        a.note_on(48, 1.0);
        a.note_on(64, 1.0);
        let _ = collect(&mut a, 100, 120.0);
        // Release every key — the latched chord keeps arpeggiating.
        assert!(a.note_off(48).is_none());
        assert!(a.note_off(64).is_none());
        let mut notes = ons(&collect(&mut a, 256, 120.0));
        for _ in 0..10 {
            notes.extend(ons(&collect(&mut a, 5513, 120.0)));
        }
        assert!(notes.iter().all(|&n| n == 48 || n == 64));
        assert!(notes.len() > 4, "latch should keep stepping");
    }

    #[test]
    fn hold_new_chord_replaces_latched() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.set_hold(true);
        a.set_rate_beats(0.25);
        a.note_on(48, 1.0);
        a.note_on(64, 1.0);
        let _ = collect(&mut a, 100, 120.0);
        // Release every key — the next press starts a NEW chord.
        a.note_off(48);
        a.note_off(64);
        a.note_on(50, 1.0);
        let mut notes = ons(&collect(&mut a, 100, 120.0));
        for _ in 0..6 {
            notes.extend(ons(&collect(&mut a, 5513, 120.0)));
        }
        assert!(notes.iter().all(|&n| n == 50), "got {:?}", notes);
        assert!(notes.len() >= 3);
    }

    #[test]
    fn hold_chord_window_expiry_replaces() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.set_hold(true);
        a.set_rate_beats(0.25);
        a.note_on(48, 1.0);
        let _ = collect(&mut a, 100, 120.0);
        // Key 48 is still physically held, but >1.5 s passes — the next
        // NoteOn is a fresh chord (legato overlap can't glue in stale notes).
        let _ = collect(&mut a, (SR * 2.0) as u32, 120.0);
        a.note_on(60, 1.0);
        let mut notes = ons(&collect(&mut a, 100, 120.0));
        for _ in 0..6 {
            notes.extend(ons(&collect(&mut a, 5513, 120.0)));
        }
        assert!(notes.iter().all(|&n| n == 60), "got {:?}", notes);
    }

    #[test]
    fn hold_notes_within_window_merge_into_chord() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.set_hold(true);
        a.set_rate_beats(0.25);
        a.note_on(48, 1.0);
        a.note_on(64, 1.0);
        let _ = collect(&mut a, 100, 120.0);
        a.note_off(48); // key 64 still held — chord not committed yet
        // Press a third note while one key is still down: same chord.
        a.note_on(67, 1.0);
        let mut notes = ons(&collect(&mut a, 100, 120.0));
        for _ in 0..9 {
            notes.extend(ons(&collect(&mut a, 5513, 120.0)));
        }
        assert!(notes.iter().all(|&n| n == 48 || n == 64 || n == 67));
        assert!(notes.iter().any(|&n| n == 67));
    }

    #[test]
    fn replacing_chord_releases_old_sounding_note() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.set_hold(true);
        a.set_rate_beats(0.25);
        a.note_on(48, 1.0);
        let _ = collect(&mut a, 100, 120.0); // 48 is sounding
        a.note_off(48); // latched
        a.note_on(50, 1.0); // replaces the chord
        let evs = collect(&mut a, 100, 120.0);
        assert!(evs
            .iter()
            .any(|e| e.kind == rinsynth_model::event::SynthEventKind::NoteOff && e.note == 48));
        assert!(evs
            .iter()
            .any(|e| e.kind == rinsynth_model::event::SynthEventKind::NoteOn && e.note == 50));
    }

    #[test]
    fn releasing_hold_drops_latched_chord() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.set_hold(true);
        a.note_on(60, 1.0);
        let _ = collect(&mut a, 100, 120.0);
        a.note_off(60); // latched
        a.set_hold(false);
        let evs = collect(&mut a, 6000, 120.0);
        // The sounding step note is released, then the arp stays silent.
        assert!(evs
            .iter()
            .any(|e| e.kind == rinsynth_model::event::SynthEventKind::NoteOff && e.note == 60));
        assert!(collect(&mut a, 6000, 120.0).is_empty());
    }

    #[test]
    fn preset_application_is_opt_in() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.set_mode(ArpMode::Down);
        a.set_rate_beats(0.5);
        // A preset that doesn't ship arp config must not touch live state.
        let mut p = Preset::default();
        p.arp_on = false;
        a.apply_preset(&p);
        assert!(a.enabled());
        assert_eq!(a.mode(), ArpMode::Down);
        assert_eq!(a.rate_beats(), 0.5);
        // A preset that opts in takes over completely.
        p.arp_on = true;
        p.arp_mode = ArpMode::UpDown;
        p.arp_rate = 0.125;
        p.arp_oct = 2;
        p.arp_hold = true;
        a.apply_preset(&p);
        assert_eq!(a.mode(), ArpMode::UpDown);
        assert_eq!(a.rate_beats(), 0.125);
        assert_eq!(a.octaves(), 2);
        assert!(a.hold_enabled());
    }

    #[test]
    fn random_stays_within_sequence() {
        let mut a = Arpeggiator::new();
        a.set_enabled(true);
        a.set_mode(ArpMode::Random);
        a.set_rate_beats(0.125);
        a.note_on(48, 1.0);
        a.note_on(60, 1.0);
        let mut notes = ons(&collect(&mut a, 100, 120.0));
        for _ in 0..30 {
            notes.extend(ons(&collect(&mut a, 2756, 120.0)));
        }
        assert!(notes.iter().all(|&n| n == 48 || n == 60));
        assert!(notes.len() > 10);
    }
}
