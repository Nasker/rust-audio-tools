//! Single source of truth for every continuous/discrete synth parameter.
//!
//! One `ParamSpec` per `SynthParam` defines its range, default, curve and
//! unit. Everything else derives from this table:
//!   - the standalone `ParamBank` defaults (`gui::ParamBank`)
//!   - the plugin `FloatParam`/`IntParam` definitions (`plugin::RustInSynthParams`)
//!   - the shared GUI slider widget (`gui::widgets::param_slider`)
//!   - MIDI CC → plain value conversion (`cc_to_plain`)
//!
//! If a knob feels different between the standalone, the plugin GUI, a MIDI
//! controller or DAW automation, THIS table is the place that got out of sync.

use crate::params::SynthParam;

/// How a normalized `[0, 1]` position maps to the plain parameter value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamCurve {
    /// `v = min + t * (max - min)`
    Linear,
    /// `v = min * (max / min) ^ t` — requires `min > 0`.
    Logarithmic,
    /// `t = 0` maps to exactly `0.0` (feature off); any `t > 0` maps
    /// logarithmically between `log_min` and `max`. Used for parameters like
    /// portamento where 0 disables the effect but the rest of the range is
    /// perceptual (time).
    LogarithmicWithZero { log_min: f32 },
}

/// Static description of one synth parameter.
pub struct ParamSpec {
    pub param: SynthParam,
    pub min: f32,
    pub max: f32,
    pub default: f32,
    pub curve: ParamCurve,
    /// Display unit suffix (e.g. `" s"`, `" Hz"`); empty for unitless.
    pub unit: &'static str,
    /// True when the parameter only takes whole-number values
    /// (waveform selectors, semitones, cents, pitch bend range, ...).
    pub stepped: bool,
}

impl ParamSpec {
    /// Map a plain value to normalized `[0, 1]` according to the curve.
    pub fn normalize(&self, value: f32) -> f32 {
        let value = value.clamp(self.min, self.max);
        match self.curve {
            ParamCurve::Linear => (value - self.min) / (self.max - self.min),
            ParamCurve::Logarithmic => (value / self.min).ln() / (self.max / self.min).ln(),
            ParamCurve::LogarithmicWithZero { log_min } => {
                if value <= 0.0 {
                    0.0
                } else {
                    let t = (value / log_min).ln() / (self.max / log_min).ln();
                    // Reserve a small bottom segment for the "off" position so
                    // the log segment never quite reaches 0.
                    0.02 + 0.98 * t.clamp(0.0, 1.0)
                }
            }
        }
    }

    /// Map a normalized `[0, 1]` position back to the plain value.
    pub fn denormalize(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        let plain = match self.curve {
            ParamCurve::Linear => self.min + t * (self.max - self.min),
            ParamCurve::Logarithmic => self.min * (self.max / self.min).powf(t),
            ParamCurve::LogarithmicWithZero { log_min } => {
                if t < 0.02 {
                    0.0
                } else {
                    let lt = (t - 0.02) / 0.98;
                    log_min * (self.max / log_min).powf(lt)
                }
            }
        };
        let plain = plain.clamp(self.min, self.max);
        if self.stepped {
            plain.round().clamp(self.min, self.max)
        } else {
            plain
        }
    }

    /// Format a plain value for display (matches the shared slider widget).
    pub fn format_value(&self, value: f32) -> String {
        let v = if self.stepped { value.round() } else { value };
        let body = if v.abs() >= 1000.0 {
            format!("{:.2}k", v / 1000.0)
        } else {
            format!("{:.2}", v)
        };
        format!("{}{}", body, self.unit)
    }

    /// Skew factor for `nih_plug::params::FloatRange::Skewed` that approximates
    /// this spec's curve, so host-drawn controls and automation lanes feel the
    /// same as the GUI. nih's skew maps `t = x^factor` where `x` is the linear
    /// fraction; we solve the factor so the normalized midpoint lands on the
    /// curve's midpoint value.
    pub fn nih_skew_factor(&self) -> f32 {
        match self.curve {
            ParamCurve::Linear => 1.0,
            ParamCurve::Logarithmic => skew_factor_for_log(self.min, self.max),
            ParamCurve::LogarithmicWithZero { log_min } => skew_factor_for_log(log_min, self.max),
        }
    }
}

/// Solve `factor` so that `FloatRange::Skewed { min, max, factor }` passes
/// through the geometric midpoint of `[min, max]` at `t = 0.5`.
pub fn skew_factor_for_log(min: f32, max: f32) -> f32 {
    let x_mid = ((min * max).sqrt() - min) / (max - min);
    0.5f32.ln() / x_mid.ln()
}

// ============================================================================
// The table
// ============================================================================

const S: &'static str = " s";
const HZ: &'static str = " Hz";
const NONE: &'static str = "";

const LIN: ParamCurve = ParamCurve::Linear;
const LOG: ParamCurve = ParamCurve::Logarithmic;
const LOG_ZERO_PORTAMENTO: ParamCurve = ParamCurve::LogarithmicWithZero { log_min: 0.005 };

/// All parameter specs. Index in this array is irrelevant; lookup goes through
/// [`spec`] / [`index`].
pub static PARAM_SPECS: [ParamSpec; 49] = [
    // ── Amp envelope ────────────────────────────────────────────────────
    ParamSpec {
        param: SynthParam::Attack,
        min: 0.001,
        max: 5.0,
        default: 0.01,
        curve: LOG,
        unit: S,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Decay,
        min: 0.001,
        max: 5.0,
        default: 0.1,
        curve: LOG,
        unit: S,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Sustain,
        min: 0.0,
        max: 1.0,
        default: 0.7,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Release,
        min: 0.001,
        max: 5.0,
        default: 0.2,
        curve: LOG,
        unit: S,
        stepped: false,
    },
    // ── Filter ──────────────────────────────────────────────────────────
    ParamSpec {
        param: SynthParam::FilterCutoff,
        min: 20.0,
        max: 20000.0,
        default: 20000.0,
        curve: LOG,
        unit: HZ,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::FilterResonance,
        min: 0.0,
        max: 1.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    // ── Filter 2 (serial after Filter 1; wide open + SVF = bypassed) ────
    ParamSpec {
        param: SynthParam::Filter2Cutoff,
        min: 20.0,
        max: 20000.0,
        default: 20000.0,
        curve: LOG,
        unit: HZ,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Filter2Resonance,
        min: 0.0,
        max: 1.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Filter2Type,
        min: 0.0,
        max: 2.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: true,
    },
    // ── Filter envelope ─────────────────────────────────────────────────
    ParamSpec {
        param: SynthParam::FilterAttack,
        min: 0.001,
        max: 5.0,
        default: 0.01,
        curve: LOG,
        unit: S,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::FilterDecay,
        min: 0.001,
        max: 5.0,
        default: 0.3,
        curve: LOG,
        unit: S,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::FilterSustain,
        min: 0.0,
        max: 1.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::FilterRelease,
        min: 0.001,
        max: 5.0,
        default: 0.3,
        curve: LOG,
        unit: S,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::FilterEnvAmount,
        min: 0.0,
        max: 1.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    // ── LFO ─────────────────────────────────────────────────────────────
    ParamSpec {
        param: SynthParam::LfoRate,
        min: 0.1,
        max: 20.0,
        default: 6.0,
        curve: LOG,
        unit: HZ,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::LfoDepth,
        min: 0.0,
        max: 1.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::LfoWaveform,
        min: 0.0,
        max: 4.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: true,
    },
    ParamSpec {
        param: SynthParam::LfoDestination,
        min: 0.0,
        max: 3.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: true,
    },
    // ── Pitch ───────────────────────────────────────────────────────────
    ParamSpec {
        param: SynthParam::PitchBendRange,
        min: 1.0,
        max: 24.0,
        default: 12.0,
        curve: LIN,
        unit: NONE,
        stepped: true,
    },
    ParamSpec {
        param: SynthParam::PortamentoTime,
        min: 0.0,
        max: 3.0,
        default: 0.0,
        curve: LOG_ZERO_PORTAMENTO,
        unit: S,
        stepped: false,
    },
    // ── Oscillator 1 ────────────────────────────────────────────────────
    ParamSpec {
        param: SynthParam::Osc1Waveform,
        min: 0.0,
        max: 5.0,
        default: 2.0,
        curve: LIN,
        unit: NONE,
        stepped: true,
    },
    ParamSpec {
        param: SynthParam::Osc1Level,
        min: 0.0,
        max: 1.0,
        default: 1.0,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Osc1Phase,
        min: 0.0,
        max: 1.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Osc1Pw,
        min: 0.0,
        max: 1.0,
        default: 0.5,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    // ── Oscillator 2 ────────────────────────────────────────────────────
    ParamSpec {
        param: SynthParam::Osc2Waveform,
        min: 0.0,
        max: 5.0,
        default: 2.0,
        curve: LIN,
        unit: NONE,
        stepped: true,
    },
    ParamSpec {
        param: SynthParam::Osc2Level,
        min: 0.0,
        max: 1.0,
        default: 0.8,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Osc2Semitones,
        min: -24.0,
        max: 24.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: true,
    },
    ParamSpec {
        param: SynthParam::Osc2Cents,
        min: -100.0,
        max: 100.0,
        default: 7.0,
        curve: LIN,
        unit: NONE,
        stepped: true,
    },
    ParamSpec {
        param: SynthParam::Osc2Phase,
        min: 0.0,
        max: 1.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Osc2Pw,
        min: 0.0,
        max: 1.0,
        default: 0.5,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    // ── Oscillator 3 ────────────────────────────────────────────────────
    ParamSpec {
        param: SynthParam::Osc3Waveform,
        min: 0.0,
        max: 5.0,
        default: 1.0,
        curve: LIN,
        unit: NONE,
        stepped: true,
    },
    ParamSpec {
        param: SynthParam::Osc3Level,
        min: 0.0,
        max: 1.0,
        default: 0.5,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Osc3Semitones,
        min: -24.0,
        max: 24.0,
        default: -12.0,
        curve: LIN,
        unit: NONE,
        stepped: true,
    },
    ParamSpec {
        param: SynthParam::Osc3Cents,
        min: -100.0,
        max: 100.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: true,
    },
    ParamSpec {
        param: SynthParam::Osc3Phase,
        min: 0.0,
        max: 1.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Osc3Pw,
        min: 0.0,
        max: 1.0,
        default: 0.5,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    // ── Oscillator 4 (FM/RM source; silent until raised) ────────────────
    ParamSpec {
        param: SynthParam::Osc4Waveform,
        min: 0.0,
        max: 5.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: true,
    },
    ParamSpec {
        param: SynthParam::Osc4Level,
        min: 0.0,
        max: 1.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Osc4Semitones,
        min: -24.0,
        max: 24.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: true,
    },
    ParamSpec {
        param: SynthParam::Osc4Cents,
        min: -100.0,
        max: 100.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: true,
    },
    ParamSpec {
        param: SynthParam::Osc4Phase,
        min: 0.0,
        max: 1.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Osc4Pan,
        min: -1.0,
        max: 1.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Osc4Pw,
        min: 0.0,
        max: 1.0,
        default: 0.5,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    // ── Stereo ──────────────────────────────────────────────────────────
    ParamSpec {
        param: SynthParam::Osc1Pan,
        min: -1.0,
        max: 1.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Osc2Pan,
        min: -1.0,
        max: 1.0,
        default: -0.3,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::Osc3Pan,
        min: -1.0,
        max: 1.0,
        default: 0.3,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    ParamSpec {
        param: SynthParam::StereoWidth,
        min: 0.0,
        max: 2.0,
        default: 1.0,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    // ── Master ──────────────────────────────────────────────────────────
    ParamSpec {
        param: SynthParam::MasterVolume,
        min: 0.0,
        max: 1.0,
        default: 0.5,
        curve: LIN,
        unit: NONE,
        stepped: false,
    },
    // ── Effects ─────────────────────────────────────────────────────────
    ParamSpec {
        param: SynthParam::DistortionType,
        min: 0.0,
        max: 3.0,
        default: 0.0,
        curve: LIN,
        unit: NONE,
        stepped: true,
    },
];

/// Stable index of a parameter (ParamBank layout / preset serialization).
pub fn index(param: SynthParam) -> usize {
    match param {
        SynthParam::Attack => 0,
        SynthParam::Decay => 1,
        SynthParam::Sustain => 2,
        SynthParam::Release => 3,
        SynthParam::FilterCutoff => 4,
        SynthParam::FilterResonance => 5,
        SynthParam::Filter2Cutoff => 35,
        SynthParam::Filter2Resonance => 36,
        SynthParam::Filter2Type => 37,
        SynthParam::FilterAttack => 6,
        SynthParam::FilterDecay => 7,
        SynthParam::FilterSustain => 8,
        SynthParam::FilterRelease => 9,
        SynthParam::FilterEnvAmount => 10,
        SynthParam::LfoRate => 11,
        SynthParam::LfoDepth => 12,
        SynthParam::LfoWaveform => 13,
        SynthParam::LfoDestination => 14,
        SynthParam::PitchBendRange => 15,
        SynthParam::Osc1Waveform => 16,
        SynthParam::Osc1Level => 17,
        SynthParam::Osc1Phase => 18,
        SynthParam::Osc1Pw => 45,
        SynthParam::Osc2Waveform => 19,
        SynthParam::Osc2Level => 20,
        SynthParam::Osc2Semitones => 21,
        SynthParam::Osc2Cents => 22,
        SynthParam::Osc2Phase => 23,
        SynthParam::Osc2Pw => 46,
        SynthParam::Osc3Waveform => 24,
        SynthParam::Osc3Level => 25,
        SynthParam::Osc3Semitones => 26,
        SynthParam::Osc3Cents => 27,
        SynthParam::Osc3Phase => 28,
        SynthParam::Osc3Pw => 47,
        SynthParam::Osc4Waveform => 38,
        SynthParam::Osc4Level => 39,
        SynthParam::Osc4Semitones => 40,
        SynthParam::Osc4Cents => 41,
        SynthParam::Osc4Phase => 42,
        SynthParam::Osc4Pan => 43,
        SynthParam::Osc4Pw => 48,
        SynthParam::Osc1Pan => 29,
        SynthParam::Osc2Pan => 30,
        SynthParam::Osc3Pan => 31,
        SynthParam::StereoWidth => 32,
        SynthParam::MasterVolume => 33,
        SynthParam::PortamentoTime => 34,
        SynthParam::DistortionType => 44,
    }
}

/// Inverse of [`index`].
pub fn param_at_index(i: usize) -> Option<SynthParam> {
    SynthParam::all().iter().copied().find(|&p| index(p) == i)
}

/// Look up the spec for a parameter.
pub fn spec(param: SynthParam) -> &'static ParamSpec {
    PARAM_SPECS
        .iter()
        .find(|s| s.param == param)
        .expect("every SynthParam must have a ParamSpec")
}

/// Convert a MIDI CC value (0–127) to a plain parameter value using the
/// spec's range and curve. This is THE CC scaling function — both the
/// standalone backend and the DSP-side CC handler must use it.
pub fn cc_to_plain(param: SynthParam, value: u8) -> f32 {
    let s = spec(param);
    let t = value as f32 / 127.0;
    s.denormalize(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_param_has_spec() {
        for &p in SynthParam::all() {
            let s = spec(p);
            assert_eq!(s.param, p);
        }
        assert_eq!(PARAM_SPECS.len(), SynthParam::all().len());
    }

    #[test]
    fn defaults_within_range() {
        for s in PARAM_SPECS.iter() {
            assert!(
                s.default >= s.min && s.default <= s.max,
                "{:?} default {} outside [{}, {}]",
                s.param,
                s.default,
                s.min,
                s.max
            );
        }
    }

    #[test]
    fn curve_round_trip() {
        for s in PARAM_SPECS.iter() {
            if s.stepped {
                // Quantized params: every integer step must round-trip exactly.
                let mut v = s.min;
                while v <= s.max {
                    let t = s.normalize(v);
                    let back = s.denormalize(t);
                    assert!(
                        (back - v).abs() < 1e-6,
                        "{:?} stepped round trip failed: v={} t={} back={}",
                        s.param,
                        v,
                        t,
                        back
                    );
                    v += 1.0;
                }
            } else {
                for i in 0..=20 {
                    let t = i as f32 / 20.0;
                    let plain = s.denormalize(t);
                    let back = s.normalize(plain);
                    assert!(
                        (back - t).abs() < 0.03,
                        "{:?} round trip failed: t={} plain={} back={}",
                        s.param,
                        t,
                        plain,
                        back
                    );
                }
            }
        }
    }

    #[test]
    fn endpoints_map_exactly() {
        for s in PARAM_SPECS.iter() {
            let lo = s.denormalize(0.0);
            let hi = s.denormalize(1.0);
            assert!(
                (hi - s.max).abs() < 1e-3,
                "{:?} t=1 -> {} (max {})",
                s.param,
                hi,
                s.max
            );
            // For LogWithZero, t=0 means "off" (0.0), not min.
            let expected_lo = if matches!(s.curve, ParamCurve::LogarithmicWithZero { .. }) {
                0.0
            } else {
                s.min
            };
            assert!(
                (lo - expected_lo).abs() < 1e-3,
                "{:?} t=0 -> {} (expected {})",
                s.param,
                lo,
                expected_lo
            );
        }
    }

    #[test]
    fn cc_endpoints() {
        for &p in SynthParam::all() {
            let s = spec(p);
            let lo = cc_to_plain(p, 0);
            let hi = cc_to_plain(p, 127);
            assert!(hi >= lo, "{:?} CC mapping not monotonic at endpoints", p);
            assert!(
                (hi - s.max).abs() < 0.51_f32.max(s.max * 0.01),
                "{:?} CC 127 -> {} (max {})",
                p,
                hi,
                s.max
            );
        }
    }

    #[test]
    fn stepped_params_snap_to_integers() {
        for s in PARAM_SPECS.iter().filter(|s| s.stepped) {
            for i in 0..=50 {
                let t = i as f32 / 50.0;
                let v = s.denormalize(t);
                assert!(
                    (v - v.round()).abs() < 1e-6,
                    "{:?} stepped denormalize({}) = {} not integer",
                    s.param,
                    t,
                    v
                );
            }
        }
    }

    #[test]
    fn log_midpoint_is_geometric_mean() {
        let s = spec(SynthParam::Attack);
        let mid = s.denormalize(0.5);
        let expected = (s.min * s.max).sqrt();
        assert!(
            (mid - expected).abs() / expected < 0.01,
            "log midpoint {} != geometric mean {}",
            mid,
            expected
        );
    }

    #[test]
    fn nih_skew_matches_log_midpoint() {
        // Verify the skew factor approximation: unnormalizing t=0.5 through
        // nih's Skewed formula should land near the geometric midpoint.
        for p in [
            SynthParam::Attack,
            SynthParam::FilterCutoff,
            SynthParam::LfoRate,
            SynthParam::PortamentoTime,
        ] {
            let s = spec(p);
            let factor = s.nih_skew_factor();
            let plain = 0.5f32.powf(factor.recip()) * (s.max - s.min) + s.min;
            let log_min = match s.curve {
                ParamCurve::LogarithmicWithZero { log_min } => log_min,
                _ => s.min,
            };
            let expected = (log_min * s.max).sqrt();
            let rel_err = (plain - expected).abs() / expected;
            assert!(
                rel_err < 0.35,
                "{:?} nih skew midpoint {} vs log midpoint {} (rel err {:.2})",
                p,
                plain,
                expected,
                rel_err
            );
        }
    }

    #[test]
    fn param_at_index_round_trip() {
        for &p in SynthParam::all() {
            assert_eq!(param_at_index(index(p)), Some(p));
        }
    }
}
