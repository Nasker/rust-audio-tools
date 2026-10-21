//! Per-voice render cost micro-benchmark.
//! `cargo run -p rinsynth-dsp --release --example perf -- [n_voices]`

use std::time::Instant;

use rinsynth_dsp::voice::VoiceManager;
use rinsynth_model::event::{NoteEvent, SynthEventReceiver};
use rinsynth_model::presets::Preset;

const SR: u32 = 44_100;
const FRAMES: usize = 256;

fn main() {
    let n_voices: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let mut vm = VoiceManager::polyphonic(SR);
    let mut preset = Preset::default();
    preset.polyphony_mode = rinsynth_model::voice::PolyphonyMode::Poly;
    vm.apply_preset(&preset);
    vm.set_master_volume(0.7);

    for i in 0..n_voices {
        vm.receive_event(NoteEvent::note_on_midi(48 + (i * 4) as u8, 104));
    }

    // Warmup
    let mut acc = 0.0f32;
    for _ in 0..200 {
        for _ in 0..FRAMES {
            let s = vm.next_sample_stereo();
            acc += s.left;
        }
    }

    // Measure blocks
    let mut total = std::time::Duration::ZERO;
    let mut max = std::time::Duration::ZERO;
    let blocks = 2000;
    for _ in 0..blocks {
        let t0 = Instant::now();
        for _ in 0..FRAMES {
            let s = vm.next_sample_stereo();
            acc += s.left;
        }
        let dt = t0.elapsed();
        total += dt;
        if dt > max {
            max = dt;
        }
    }

    let per_block_us = total.as_secs_f64() * 1e6 / blocks as f64;
    println!(
        "voices={n_voices}: avg {:.0} us/block, max {:.0} us/block ({:.1}% of 5805 us budget)",
        per_block_us,
        max.as_secs_f64() * 1e6,
        per_block_us / 5805.0 * 100.0
    );
    eprintln!("(sink {acc})");
}
