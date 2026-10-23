//! rinsynth-dsp — the portable DSP core of RustInSynth.
//!
//! Voice/oscillator/filter/envelope/effects implementations shared between
//! the desktop host/plugin and the ESP32-S3 firmware port. Depends only on
//! `rinsynth-model` for schema/param types and carries **no** serde derives —
//! this is the audio hot path and must build fully optimized (see the
//! Xtensa LLVM-21 serde-f32 codegen issue, esp-rs/rust#277).

pub mod effects;
pub mod embedded;
pub mod envelope;
pub mod filter;
pub mod oscillator;
pub mod voice;

pub use effects::{Chorus, Delay, Distortion, Effect, EffectsChain, Reverb};
pub use embedded::EmbeddedEngine;
pub use envelope::{ADSREnvelope, AREnvelope, Envelope, EnvelopeState};
pub use filter::{cc_to_cutoff, cc_to_resonance, Filter, FilterMode, SVFilter};
pub use oscillator::{
    create_oscillator, NoiseOscillator, Oscillator, OscillatorBank, OscillatorUnit, SawOscillator,
    SineOscillator, SquareOscillator, TriangleOscillator,
};
pub use voice::{LfoSettings, ModState, Voice, VoiceManager};

/// Per-stage cycle profiler for the voice render path. The host installs a
/// cycle-counter hook (e.g. Xtensa `xthal_get_ccount`); with no hook set the
/// instrumentation is a single predictable branch per sample.
pub mod profile {
    use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

    /// Stage indices: envelopes, control tick, oscillators+mix, filters,
    /// FX chain, sample count.
    pub const ENV: usize = 0;
    pub const CONTROL: usize = 1;
    pub const OSC: usize = 2;
    pub const FILTER: usize = 3;
    pub const FX: usize = 4;
    pub const SAMPLES: usize = 5;
    const N: usize = 6;

    static HOOK: AtomicUsize = AtomicUsize::new(0);
    static COUNTERS: [AtomicU32; N] = [
        AtomicU32::new(0),
        AtomicU32::new(0),
        AtomicU32::new(0),
        AtomicU32::new(0),
        AtomicU32::new(0),
        AtomicU32::new(0),
    ];

    pub fn set_hook(f: fn() -> u64) {
        HOOK.store(f as usize, Ordering::Release);
    }

    #[inline]
    pub fn enabled() -> bool {
        HOOK.load(Ordering::Relaxed) != 0
    }

    #[inline]
    pub fn now() -> u64 {
        let h = HOOK.load(Ordering::Relaxed);
        if h == 0 {
            0
        } else {
            unsafe { core::mem::transmute::<usize, fn() -> u64>(h)() }
        }
    }

    #[inline]
    pub fn add(stage: usize, cycles: u64) {
        COUNTERS[stage].fetch_add(cycles as u32, Ordering::Relaxed);
    }

    pub fn snapshot() -> [u64; N] {
        core::array::from_fn(|i| COUNTERS[i].load(Ordering::Relaxed) as u64)
    }

    pub fn reset() {
        for c in &COUNTERS {
            c.store(0, Ordering::Relaxed);
        }
    }
}
