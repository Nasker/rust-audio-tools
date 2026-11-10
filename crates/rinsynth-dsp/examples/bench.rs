//! Riff-accurate render benchmark: brassy stab with Chorus+Delay+Reverb,
//! LFO->Cutoff, filter env — 3-note chords with overlapping release tails.
use rinsynth_dsp::embedded::EmbeddedEngine;
use rinsynth_model::{NoteEvent, Preset, SynthParam};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let mut e = EmbeddedEngine::new(6, 44100);
    let mut p = Preset::default();
    p.filter_cutoff = 1800.0;
    p.filter_resonance = 0.50;
    p.filter_env_amount = 0.40;
    p.lfo_destination = rinsynth_model::LfoDestination::FilterCutoff;
    p.lfo_rate = 0.4;
    p.lfo_depth = 0.20;
    p.chorus_enabled = true;
    p.delay_enabled = true;
    p.reverb_enabled = true;
    e.apply_preset(&p);
    e.effects.set_delay_enabled(true);
    e.effects.set_reverb_enabled(true);
    e.effects.set_chorus_enabled(true);
    e.set_param(SynthParam::MasterVolume, 0.8);
    // 3-note chord + retrigger pattern leaves release tails like the riff.
    for _ in 0..44100 {
        black_box(e.next_sample_stereo());
    }
    let n = 4_410_000usize;
    let t = Instant::now();
    let mut acc = 0.0f32;
    for b in 0..n / 2205 {
        if b % 10 == 0 {
            for &note in &[67u8, 71, 74] {
                e.receive_event(NoteEvent::note_on_midi(note, 110));
            }
        }
        for _ in 0..2205 {
            let s = e.next_sample_stereo();
            acc += s.left;
        }
    }
    let el = t.elapsed();
    black_box(acc);
    println!(
        "{} samples in {:?} -> {:.2} ns/sample ({:.1}% of 44.1kHz realtime)",
        n,
        el,
        el.as_nanos() as f64 / n as f64,
        (el.as_secs_f64() / (n as f64 / 44100.0)) * 100.0
    );
}
