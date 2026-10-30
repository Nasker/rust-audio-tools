//! Audio effects: Delay, Reverb, Chorus


use rinsynth_model::types::{Sample, SampleRate, StereoSample};

// Schema type lives in the portable model crate; re-exported so
// `crate::effects::DistortionType` keeps resolving.
pub use rinsynth_model::effects::DistortionType;

/// Trait for audio effects (mono)
pub trait Effect: Send {
    /// Process a single sample
    fn process(&mut self, input: Sample) -> Sample;

    /// Set the sample rate (call when sample rate changes)
    fn set_sample_rate(&mut self, sample_rate: SampleRate);

    /// Reset internal state (clear buffers)
    fn reset(&mut self);
}

/// Trait for stereo audio effects
pub trait StereoEffect: Send {
    /// Process a stereo sample
    fn process_stereo(&mut self, input: StereoSample) -> StereoSample;

    /// Set the sample rate
    fn set_sample_rate(&mut self, sample_rate: SampleRate);

    /// Reset internal state
    fn reset(&mut self);
}

// ============================================================================
// DELAY
// ============================================================================

/// Simple power-of-two delay line buffer for branchless bitmask indexing
struct DelayLine {
    buffer: Vec<Sample>,
    mask: usize,
    write_pos: usize,
}

impl DelayLine {
    fn new(max_delay_seconds: f32, sample_rate: SampleRate) -> Self {
        let min_size = (max_delay_seconds * sample_rate as f32) as usize + 2;
        let size = min_size.next_power_of_two();
        Self {
            buffer: vec![0.0; size],
            mask: size - 1,
            write_pos: 0,
        }
    }

    #[inline(always)]
    fn write(&mut self, sample: Sample) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) & self.mask;
    }

    #[inline(always)]
    fn read(&self, delay_samples: usize) -> Sample {
        self.buffer[self.write_pos.wrapping_sub(delay_samples) & self.mask]
    }

    #[inline(always)]
    fn read_interpolated(&self, delay_samples: f32) -> Sample {
        let delay_int = delay_samples as usize;
        let frac = delay_samples - delay_int as f32;
        let i0 = self.write_pos.wrapping_sub(delay_int) & self.mask;
        let i1 = i0.wrapping_sub(1) & self.mask;
        let s1 = self.buffer[i0];
        let s2 = self.buffer[i1];
        s1 + frac * (s2 - s1)
    }

    fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }
}

/// Stereo ping-pong delay effect
pub struct Delay {
    delay_line_l: DelayLine,
    delay_line_r: DelayLine,
    delay_time: f32, // Delay time in seconds (0.0 - 2.0)
    feedback: f32,   // Feedback amount (0.0 - 0.95)
    mix: f32,        // Dry/wet mix (0.0 = dry, 1.0 = wet)
    ping_pong: bool, // Enable ping-pong mode
    // Tempo sync: when on, delay_time = beats * 60/bpm
    tempo_sync: bool,
    sync_beats: f32,
    bpm: f32,
    sample_rate: SampleRate,
    // Modulation offsets from the routing matrix (added on top of the
    // base values each sample; not user-facing state)
    time_mod: f32,
    feedback_mod: f32,
    mix_mod: f32,
}

impl Delay {
    pub fn new(sample_rate: SampleRate) -> Self {
        Self {
            delay_line_l: DelayLine::new(2.0, sample_rate),
            delay_line_r: DelayLine::new(2.0, sample_rate),
            delay_time: 0.3,
            feedback: 0.4,
            mix: 0.3,
            ping_pong: true, // Ping-pong on by default for stereo goodness
            tempo_sync: false,
            sync_beats: 1.0,
            bpm: 120.0,
            sample_rate,
            time_mod: 0.0,
            feedback_mod: 0.0,
            mix_mod: 0.0,
        }
    }

    /// Set modulation offsets applied on top of the base parameters.
    /// `time` is additive seconds; `feedback`/`mix` are additive 0..1.
    pub fn set_mods(&mut self, time: f32, feedback: f32, mix: f32) {
        self.time_mod = time;
        self.feedback_mod = feedback;
        self.mix_mod = mix;
    }

    pub fn set_delay_time(&mut self, time: f32) {
        self.delay_time = time.clamp(0.001, 2.0);
    }

    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.clamp(0.0, 0.95);
    }

    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
    }

    pub fn set_ping_pong(&mut self, enabled: bool) {
        self.ping_pong = enabled;
    }

    /// Set the tempo in BPM (for tempo-synced delay times).
    pub fn set_bpm(&mut self, bpm: f32) {
        self.bpm = bpm.clamp(20.0, 300.0);
        self.update_synced_time();
    }

    /// Enable/disable tempo sync and set the delay length in beats.
    pub fn set_tempo_sync(&mut self, enabled: bool, beats: f32) {
        self.tempo_sync = enabled;
        self.sync_beats = beats.clamp(0.05, 16.0);
        self.update_synced_time();
    }

    fn update_synced_time(&mut self) {
        if self.tempo_sync {
            self.delay_time = (self.sync_beats * 60.0 / self.bpm).clamp(0.001, 2.0);
        }
    }

    pub fn tempo_sync(&self) -> bool {
        self.tempo_sync
    }
    pub fn sync_beats(&self) -> f32 {
        self.sync_beats
    }

    pub fn delay_time(&self) -> f32 {
        self.delay_time
    }
    pub fn feedback(&self) -> f32 {
        self.feedback
    }
    pub fn mix(&self) -> f32 {
        self.mix
    }
    pub fn ping_pong(&self) -> bool {
        self.ping_pong
    }

    /// Process stereo with ping-pong
    #[link_section = ".iram1.text"]
    pub fn process_stereo(&mut self, input: StereoSample) -> StereoSample {
        let eff_time = (self.delay_time + self.time_mod).clamp(0.001, 2.0);
        let eff_feedback = (self.feedback + self.feedback_mod).clamp(0.0, 0.95);
        let eff_mix = (self.mix + self.mix_mod).clamp(0.0, 1.0);

        let delay_samples = (eff_time * self.sample_rate as f32) as usize;
        let delay_samples = delay_samples.max(1);

        let delayed_l = self.delay_line_l.read(delay_samples);
        let delayed_r = self.delay_line_r.read(delay_samples);

        if self.ping_pong {
            // Ping-pong: L feeds into R, R feeds into L
            self.delay_line_l
                .write(input.left + delayed_r * eff_feedback);
            self.delay_line_r
                .write(input.right + delayed_l * eff_feedback);
        } else {
            // Standard stereo delay
            self.delay_line_l
                .write(input.left + delayed_l * eff_feedback);
            self.delay_line_r
                .write(input.right + delayed_r * eff_feedback);
        }

        StereoSample::new(
            input.left * (1.0 - eff_mix) + delayed_l * eff_mix,
            input.right * (1.0 - eff_mix) + delayed_r * eff_mix,
        )
    }
}

impl Effect for Delay {
    fn process(&mut self, input: Sample) -> Sample {
        let stereo = self.process_stereo(StereoSample::from_mono(input));
        (stereo.left + stereo.right) * 0.5
    }

    fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
        self.delay_line_l = DelayLine::new(2.0, sample_rate);
        self.delay_line_r = DelayLine::new(2.0, sample_rate);
    }

    fn reset(&mut self) {
        self.delay_line_l.clear();
        self.delay_line_r.clear();
    }
}

// ============================================================================
// REVERB (Schroeder-style with allpass + comb filters)
// ============================================================================

/// Simple comb filter for reverb
struct CombFilter {
    buffer: Vec<Sample>,
    write_pos: usize,
    feedback: f32,
    damp: f32,
    damp_state: Sample,
}

impl CombFilter {
    fn new(delay_samples: usize, feedback: f32, damp: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_samples],
            write_pos: 0,
            feedback,
            damp,
            damp_state: 0.0,
        }
    }

    #[inline(always)]
    fn process(&mut self, input: Sample) -> Sample {
        let output = self.buffer[self.write_pos];

        // Low-pass filter on feedback (damping)
        self.damp_state = output * (1.0 - self.damp) + self.damp_state * self.damp;

        self.buffer[self.write_pos] = input + self.damp_state * self.feedback;
        self.write_pos += 1;
        if self.write_pos >= self.buffer.len() {
            self.write_pos = 0;
        }

        output
    }

    fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.damp_state = 0.0;
    }
}

/// Allpass filter for reverb diffusion
struct AllpassFilter {
    buffer: Vec<Sample>,
    write_pos: usize,
    feedback: f32,
}

impl AllpassFilter {
    fn new(delay_samples: usize, feedback: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_samples],
            write_pos: 0,
            feedback,
        }
    }

    #[inline(always)]
    fn process(&mut self, input: Sample) -> Sample {
        let delayed = self.buffer[self.write_pos];
        let output = -input + delayed;

        self.buffer[self.write_pos] = input + delayed * self.feedback;
        self.write_pos += 1;
        if self.write_pos >= self.buffer.len() {
            self.write_pos = 0;
        }

        output
    }

    fn clear(&mut self) {
        self.buffer.fill(0.0);
    }
}

/// Stereo Schroeder reverb with different L/R comb filter delays
pub struct Reverb {
    combs_l: Vec<CombFilter>,
    combs_r: Vec<CombFilter>,
    allpasses_l: Vec<AllpassFilter>,
    allpasses_r: Vec<AllpassFilter>,
    room_size: f32, // 0.0 - 1.0
    damping: f32,   // 0.0 - 1.0
    mix: f32,       // Dry/wet mix
    sample_rate: SampleRate,
    mix_mod: f32, // Matrix modulation offset on mix
    half_step: bool,
    wet_l: Sample,
    wet_r: Sample,
}

impl Reverb {
    pub fn new(sample_rate: SampleRate) -> Self {
        let mut reverb = Self {
            combs_l: Vec::new(),
            combs_r: Vec::new(),
            allpasses_l: Vec::new(),
            allpasses_r: Vec::new(),
            room_size: 0.5,
            damping: 0.5,
            mix: 0.3,
            sample_rate,
            mix_mod: 0.0,
            half_step: false,
            wet_l: 0.0,
            wet_r: 0.0,
        };
        reverb.rebuild_filters();
        reverb
    }

    fn rebuild_filters(&mut self) {
        // Comb/allpass tanks run at half rate (22.05 kHz at 44.1 kHz): halving
        // buffer lengths preserves exact delay times and RT60 decay while
        // keeping the total working set (~24 KB) inside ESP32-S3's 32 KB D-cache.
        let scale = (self.sample_rate as f32 / 44100.0) * 0.5;
        let comb_delays_l = [1116, 1188, 1277, 1356];
        let comb_delays_r = [1139, 1211, 1300, 1379]; // Slightly different for stereo
        let allpass_delays_l = [556, 441];
        let allpass_delays_r = [579, 464]; // Slightly different for stereo

        let feedback = 0.84 + self.room_size * 0.12; // 0.84 - 0.96

        self.combs_l = comb_delays_l
            .iter()
            .map(|&d| CombFilter::new(((d as f32 * scale) as usize).max(1), feedback, self.damping))
            .collect();
        self.combs_r = comb_delays_r
            .iter()
            .map(|&d| CombFilter::new(((d as f32 * scale) as usize).max(1), feedback, self.damping))
            .collect();

        self.allpasses_l = allpass_delays_l
            .iter()
            .map(|&d| AllpassFilter::new(((d as f32 * scale) as usize).max(1), 0.5))
            .collect();
        self.allpasses_r = allpass_delays_r
            .iter()
            .map(|&d| AllpassFilter::new(((d as f32 * scale) as usize).max(1), 0.5))
            .collect();
    }

    pub fn set_room_size(&mut self, size: f32) {
        self.room_size = size.clamp(0.0, 1.0);
        // Update comb filter feedback
        let feedback = 0.84 + self.room_size * 0.12;
        for comb in &mut self.combs_l {
            comb.feedback = feedback;
        }
        for comb in &mut self.combs_r {
            comb.feedback = feedback;
        }
    }

    pub fn set_damping(&mut self, damping: f32) {
        self.damping = damping.clamp(0.0, 1.0);
        for comb in &mut self.combs_l {
            comb.damp = self.damping;
        }
        for comb in &mut self.combs_r {
            comb.damp = self.damping;
        }
    }

    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
    }

    /// Set the matrix modulation offset on the dry/wet mix.
    pub fn set_mix_mod(&mut self, offset: f32) {
        self.mix_mod = offset;
    }

    pub fn room_size(&self) -> f32 {
        self.room_size
    }
    pub fn damping(&self) -> f32 {
        self.damping
    }
    pub fn mix(&self) -> f32 {
        self.mix
    }

    /// Process stereo reverb
    #[link_section = ".iram1.text"]
    pub fn process_stereo(&mut self, input: StereoSample) -> StereoSample {
        self.half_step = !self.half_step;
        if self.half_step {
            // Sum of parallel comb filters (separate L/R)
            let mut comb_sum_l: Sample = 0.0;
            let mut comb_sum_r: Sample = 0.0;
            for comb in &mut self.combs_l {
                comb_sum_l += comb.process(input.left);
            }
            for comb in &mut self.combs_r {
                comb_sum_r += comb.process(input.right);
            }
            comb_sum_l *= 0.25;
            comb_sum_r *= 0.25;

            // Series allpass filters for diffusion
            let mut output_l = comb_sum_l;
            let mut output_r = comb_sum_r;
            for allpass in &mut self.allpasses_l {
                output_l = allpass.process(output_l);
            }
            for allpass in &mut self.allpasses_r {
                output_r = allpass.process(output_r);
            }
            self.wet_l = output_l;
            self.wet_r = output_r;
        }

        // Mix dry and wet
        let eff_mix = (self.mix + self.mix_mod).clamp(0.0, 1.0);
        StereoSample::new(
            input.left * (1.0 - eff_mix) + self.wet_l * eff_mix,
            input.right * (1.0 - eff_mix) + self.wet_r * eff_mix,
        )
    }
}

impl Effect for Reverb {
    fn process(&mut self, input: Sample) -> Sample {
        let stereo = self.process_stereo(StereoSample::from_mono(input));
        (stereo.left + stereo.right) * 0.5
    }

    fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
        self.rebuild_filters();
    }

    fn reset(&mut self) {
        for comb in &mut self.combs_l {
            comb.clear();
        }
        for comb in &mut self.combs_r {
            comb.clear();
        }
        for allpass in &mut self.allpasses_l {
            allpass.clear();
        }
        for allpass in &mut self.allpasses_r {
            allpass.clear();
        }
    }
}

// ============================================================================
// CHORUS
// ============================================================================

/// Stereo chorus effect with LFO phase offset between L/R
pub struct Chorus {
    delay_line_l: DelayLine,
    delay_line_r: DelayLine,
    lfo_phase: f32,
    rate: f32,          // LFO rate in Hz (0.1 - 5.0)
    depth: f32,         // Modulation depth in ms (0.0 - 10.0)
    mix: f32,           // Dry/wet mix
    stereo_spread: f32, // LFO phase offset for stereo (0.0 - 0.5)
    sample_rate: SampleRate,
    // Matrix modulation offsets: rate_mod is an octave exponent
    // (rate * 4^v), depth_mod is scaled to ±5 ms, mix_mod is additive.
    rate_mod: f32,
    depth_mod: f32,
    mix_mod: f32,
}

impl Chorus {
    pub fn new(sample_rate: SampleRate) -> Self {
        Self {
            delay_line_l: DelayLine::new(0.05, sample_rate), // 50ms max
            delay_line_r: DelayLine::new(0.05, sample_rate),
            lfo_phase: 0.0,
            rate: 1.5,
            depth: 3.0,
            mix: 0.5,
            stereo_spread: 0.25, // 90 degrees phase offset
            sample_rate,
            rate_mod: 0.0,
            depth_mod: 0.0,
            mix_mod: 0.0,
        }
    }

    /// Set modulation offsets: `rate` is exponential (±2 octaves at ±1.0),
    /// `depth` maps ±1.0 to ±5 ms, `mix` is additive 0..1.
    pub fn set_mods(&mut self, rate: f32, depth: f32, mix: f32) {
        self.rate_mod = rate;
        self.depth_mod = depth;
        self.mix_mod = mix;
    }

    pub fn set_rate(&mut self, rate: f32) {
        self.rate = rate.clamp(0.1, 5.0);
    }

    pub fn set_depth(&mut self, depth: f32) {
        self.depth = depth.clamp(0.0, 10.0);
    }

    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
    }

    pub fn rate(&self) -> f32 {
        self.rate
    }
    pub fn depth(&self) -> f32 {
        self.depth
    }
    pub fn mix(&self) -> f32 {
        self.mix
    }

    /// Process stereo chorus
    #[link_section = ".iram1.text"]
    pub fn process_stereo(&mut self, input: StereoSample) -> StereoSample {
        // Fast parabolic sine on normalized phase [0, 1): 4 muls, no LUT/OnceLock
        #[inline(always)]
        fn fast_sin01(p: f32) -> f32 {
            let x = p - 0.5;
            4.0 * x * (1.0 - 2.0 * x.abs())
        }
        let lfo_l = fast_sin01(self.lfo_phase);
        let mut pr = self.lfo_phase + self.stereo_spread;
        if pr >= 1.0 {
            pr -= 1.0;
        }
        let lfo_r = fast_sin01(pr);

        let eff_rate = if self.rate_mod == 0.0 {
            self.rate
        } else {
            (self.rate * 4.0f32.powf(self.rate_mod)).clamp(0.05, 20.0)
        };
        self.lfo_phase += eff_rate / self.sample_rate as f32;
        if self.lfo_phase >= 1.0 {
            self.lfo_phase -= 1.0;
        }

        // Modulated delay times
        let eff_depth = (self.depth + self.depth_mod * 5.0).clamp(0.0, 15.0);
        let eff_mix = (self.mix + self.mix_mod).clamp(0.0, 1.0);
        let center_delay_ms = 7.0;
        let delay_ms_l = center_delay_ms + lfo_l * eff_depth * 0.5;
        let delay_ms_r = center_delay_ms + lfo_r * eff_depth * 0.5;
        let delay_samples_l = delay_ms_l * 0.001 * self.sample_rate as f32;
        let delay_samples_r = delay_ms_r * 0.001 * self.sample_rate as f32;

        // Write to delay lines
        self.delay_line_l.write(input.left);
        self.delay_line_r.write(input.right);

        // Read with interpolation
        let delayed_l = self.delay_line_l.read_interpolated(delay_samples_l);
        let delayed_r = self.delay_line_r.read_interpolated(delay_samples_r);

        // Mix dry and wet
        StereoSample::new(
            input.left * (1.0 - eff_mix) + delayed_l * eff_mix,
            input.right * (1.0 - eff_mix) + delayed_r * eff_mix,
        )
    }
}

impl Effect for Chorus {
    fn process(&mut self, input: Sample) -> Sample {
        let stereo = self.process_stereo(StereoSample::from_mono(input));
        (stereo.left + stereo.right) * 0.5
    }

    fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
        self.delay_line_l = DelayLine::new(0.05, sample_rate);
        self.delay_line_r = DelayLine::new(0.05, sample_rate);
    }

    fn reset(&mut self) {
        self.delay_line_l.clear();
        self.delay_line_r.clear();
        self.lfo_phase = 0.0;
    }
}

// ============================================================================
// DISTORTION
// ============================================================================

/// Waveshaping distortion with selectable topology.
/// `drive` is 0..1 mapped to type-specific intensity; `mix` is dry/wet.
/// Output is level-compensated so cranking drive doesn't blow the mix out.
pub struct Distortion {
    drive: f32,
    mix: f32,
    dtype: DistortionType,
    // Precomputed level compensation (sqrt denominators are soft-float on
    // Xtensa — cache them in set_drive instead of computing per sample).
    tube_comp: f32,
    fuzz_comp: f32,
    // Bitcrush: samples held so far and held value, per channel [L, R].
    bc_count: [u32; 2],
    bc_held: [Sample; 2],
}

impl Distortion {
    pub fn new() -> Self {
        Self {
            drive: 0.0,
            mix: 0.5,
            dtype: DistortionType::Tube,
            tube_comp: 1.0,
            fuzz_comp: 1.25,
            bc_count: [0; 2],
            bc_held: [0.0; 2],
        }
    }

    #[inline]
    fn shape(&mut self, x: f32, ch: usize) -> f32 {
        if self.drive <= 0.0 {
            return x;
        }
        match self.dtype {
            DistortionType::Tube => {
                let gain = 1.0 + 29.0 * self.drive;
                // tanh clip via Pade [3/3] (<=0.02 error for |x|<3; saturates
                // past it) — libm tanh is soft-float and costly on Xtensa.
                let y = x * gain;
                let y2 = y * y;
                let t = if y.abs() > 3.0 {
                    y.signum()
                } else {
                    y * (27.0 + y2) / (27.0 + 9.0 * y2)
                };
                t * self.tube_comp
            }
            DistortionType::Fuzz => {
                let gain = 1.0 + 49.0 * self.drive;
                // Bias for asymmetry, then hard clip; keep level roughly even.
                let y = x * gain + 0.3 * self.drive;
                y.clamp(-0.8, 0.8) * self.fuzz_comp
            }
            DistortionType::Bitcrush => {
                // Drive → hold length 1..48 samples and bit depth 12..3 bits.
                let hold_len = 1 + (self.drive * 47.0) as u32;
                let bits = (12.0 - self.drive * 9.0).clamp(3.0, 12.0);
                if self.bc_count[ch] >= hold_len {
                    let levels = 2.0f32.powi(bits as i32);
                    self.bc_held[ch] = (x * levels).round() / levels;
                    self.bc_count[ch] = 0;
                }
                self.bc_count[ch] += 1;
                self.bc_held[ch]
            }
            DistortionType::Fold => {
                let gain = 1.0 + 29.0 * self.drive;
                // Sine fold: bounded, no DC offset, turns saws into clusters.
                // sine_lut only wraps ±1 turn, so wrap the (large) argument
                // first — avoids libm sinf range reduction on Xtensa.
                const INV_TAU: f32 = 1.0 / (2.0 * core::f32::consts::PI);
                const TAU: f32 = 2.0 * core::f32::consts::PI;
                let p = x * gain * 0.5;
                let wrapped = p - (p * INV_TAU).floor() * TAU;
                crate::oscillator::sine_lut(wrapped) * 0.9
            }
        }
    }

    pub fn process_stereo(&mut self, input: StereoSample) -> StereoSample {
        let l = self.shape(input.left, 0);
        let r = self.shape(input.right, 1);
        StereoSample::new(
            input.left * (1.0 - self.mix) + l * self.mix,
            input.right * (1.0 - self.mix) + r * self.mix,
        )
    }

    pub fn set_drive(&mut self, drive: f32) {
        self.drive = drive.clamp(0.0, 1.0);
        // Tube: 1/sqrt(1+29d); Fuzz: 1.25/sqrt(1+1.5d)
        self.tube_comp = 1.0 / (1.0 + 29.0 * self.drive).sqrt();
        self.fuzz_comp = 1.25 / (1.0 + 1.5 * self.drive).sqrt();
    }
    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
    }
    pub fn set_type(&mut self, dtype: DistortionType) {
        if dtype != self.dtype {
            self.dtype = dtype;
            self.bc_count = [0; 2];
            self.bc_held = [0.0; 2];
        }
    }
    pub fn drive(&self) -> f32 {
        self.drive
    }
    pub fn mix(&self) -> f32 {
        self.mix
    }
    pub fn dtype(&self) -> DistortionType {
        self.dtype
    }
}

// ============================================================================
// EFFECTS CHAIN
// ============================================================================

/// Chain of effects processed in series
pub struct EffectsChain {
    pub distortion: Distortion,
    pub delay: Delay,
    pub reverb: Reverb,
    pub chorus: Chorus,
    pub distortion_enabled: bool,
    pub delay_enabled: bool,
    pub reverb_enabled: bool,
    pub chorus_enabled: bool,
}

impl EffectsChain {
    pub fn new(sample_rate: SampleRate) -> Self {
        Self {
            distortion: Distortion::new(),
            delay: Delay::new(sample_rate),
            reverb: Reverb::new(sample_rate),
            chorus: Chorus::new(sample_rate),
            distortion_enabled: false,
            delay_enabled: false,
            reverb_enabled: false,
            chorus_enabled: false,
        }
    }

    /// Process mono (for backwards compatibility)
    pub fn process(&mut self, input: Sample) -> Sample {
        let stereo = self.process_stereo(StereoSample::from_mono(input));
        (stereo.left + stereo.right) * 0.5
    }

    /// Apply global FX modulation offsets resolved by the VoiceManager from
    /// the routing matrix. Call once per sample before `process_stereo`.
    pub fn set_fx_mods(&mut self, mods: &rinsynth_model::mod_matrix::FxMods) {
        self.delay
            .set_mods(mods.delay_time, mods.delay_feedback, mods.delay_mix);
        self.reverb.set_mix_mod(mods.reverb_mix);
        self.chorus
            .set_mods(mods.chorus_rate, mods.chorus_depth, mods.chorus_mix);
    }

    /// Process stereo signal through effects chain
    #[link_section = ".iram1.text"]
    pub fn process_stereo(&mut self, input: StereoSample) -> StereoSample {
        if !self.distortion_enabled
            && !self.chorus_enabled
            && !self.delay_enabled
            && !self.reverb_enabled
        {
            return input;
        }
        // Inject a tiny anti-denormal DC bias (-380 dBFS) when any effect is
        // active so feedback states in chorus/delay/reverb never decay into
        // subnormal software traps on Xtensa.
        let mut output = StereoSample::new(input.left + 1e-19, input.right - 1e-19);

        // Process in order: Distortion → Chorus → Delay → Reverb
        if self.distortion_enabled {
            output = self.distortion.process_stereo(output);
        }
        if self.chorus_enabled {
            output = self.chorus.process_stereo(output);
        }
        if self.delay_enabled {
            output = self.delay.process_stereo(output);
        }
        if self.reverb_enabled {
            output = self.reverb.process_stereo(output);
        }

        output
    }

    pub fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.delay.set_sample_rate(sample_rate);
        self.reverb.set_sample_rate(sample_rate);
        self.chorus.set_sample_rate(sample_rate);
    }

    pub fn reset(&mut self) {
        self.delay.reset();
        self.reverb.reset();
        self.chorus.reset();
    }

    // ── Distortion wrappers ──────────────────────────────────────────────────

    pub fn set_distortion_enabled(&mut self, enabled: bool) {
        self.distortion_enabled = enabled;
    }

    pub fn set_distortion_drive(&mut self, drive: f32) {
        self.distortion.set_drive(drive);
    }

    pub fn set_distortion_mix(&mut self, mix: f32) {
        self.distortion.set_mix(mix);
    }

    pub fn set_distortion_type(&mut self, dtype: DistortionType) {
        self.distortion.set_type(dtype);
    }

    // ── Delay wrappers ───────────────────────────────────────────────────────

    pub fn set_delay_enabled(&mut self, enabled: bool) {
        self.delay_enabled = enabled;
    }

    pub fn set_delay_time(&mut self, time: f32) {
        self.delay.set_delay_time(time);
    }

    pub fn set_delay_feedback(&mut self, feedback: f32) {
        self.delay.set_feedback(feedback);
    }

    pub fn set_delay_mix(&mut self, mix: f32) {
        self.delay.set_mix(mix);
    }

    /// Set tempo for all tempo-synced effects.
    pub fn set_bpm(&mut self, bpm: f32) {
        self.delay.set_bpm(bpm);
    }

    /// Enable/disable tempo sync on the delay and set its beat length.
    pub fn set_delay_tempo_sync(&mut self, enabled: bool, beats: f32) {
        self.delay.set_tempo_sync(enabled, beats);
    }

    // ── Reverb wrappers ──────────────────────────────────────────────────────

    pub fn set_reverb_enabled(&mut self, enabled: bool) {
        self.reverb_enabled = enabled;
    }

    pub fn set_reverb_room_size(&mut self, size: f32) {
        self.reverb.set_room_size(size);
    }

    pub fn set_reverb_damping(&mut self, damping: f32) {
        self.reverb.set_damping(damping);
    }

    pub fn set_reverb_mix(&mut self, mix: f32) {
        self.reverb.set_mix(mix);
    }

    // ── Chorus wrappers ──────────────────────────────────────────────────────

    pub fn set_chorus_enabled(&mut self, enabled: bool) {
        self.chorus_enabled = enabled;
    }

    pub fn set_chorus_rate(&mut self, rate: f32) {
        self.chorus.set_rate(rate);
    }

    pub fn set_chorus_depth(&mut self, depth: f32) {
        self.chorus.set_depth(depth);
    }

    pub fn set_chorus_mix(&mut self, mix: f32) {
        self.chorus.set_mix(mix);
    }

    // ── Getters ──────────────────────────────────────────────────────────────

    pub fn distortion_enabled(&self) -> bool {
        self.distortion_enabled
    }
    pub fn distortion_drive(&self) -> f32 {
        self.distortion.drive()
    }
    pub fn distortion_mix(&self) -> f32 {
        self.distortion.mix()
    }
    pub fn distortion_type(&self) -> DistortionType {
        self.distortion.dtype()
    }

    pub fn delay_enabled(&self) -> bool {
        self.delay_enabled
    }
    pub fn delay_time(&self) -> f32 {
        self.delay.delay_time()
    }
    pub fn delay_feedback(&self) -> f32 {
        self.delay.feedback()
    }
    pub fn delay_mix(&self) -> f32 {
        self.delay.mix()
    }
    pub fn delay_tempo_sync(&self) -> bool {
        self.delay.tempo_sync()
    }
    pub fn delay_sync_beats(&self) -> f32 {
        self.delay.sync_beats()
    }

    pub fn reverb_enabled(&self) -> bool {
        self.reverb_enabled
    }
    pub fn reverb_room_size(&self) -> f32 {
        self.reverb.room_size()
    }
    pub fn reverb_damping(&self) -> f32 {
        self.reverb.damping()
    }
    pub fn reverb_mix(&self) -> f32 {
        self.reverb.mix()
    }

    pub fn chorus_enabled(&self) -> bool {
        self.chorus_enabled
    }
    pub fn chorus_rate(&self) -> f32 {
        self.chorus.rate()
    }
    pub fn chorus_depth(&self) -> f32 {
        self.chorus.depth()
    }
    pub fn chorus_mix(&self) -> f32 {
        self.chorus.mix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Render a 440 Hz sine through the chain with only the distortion stage
    /// enabled at the given type.
    fn render(dtype: DistortionType) -> Vec<f32> {
        let mut fx = EffectsChain::new(44100);
        fx.set_distortion_enabled(true);
        fx.set_distortion_type(dtype);
        fx.set_distortion_drive(0.7);
        fx.set_distortion_mix(1.0);
        let mut out = Vec::with_capacity(2048);
        let mut phase = 0.0f32;
        for _ in 0..2048 {
            phase += 440.0 / 44100.0;
            let s = (phase * std::f32::consts::TAU).sin() * 0.6;
            out.push(fx.process_stereo(StereoSample::from_mono(s)).left);
        }
        out
    }

    /// Regression: each waveshaper topology must produce a distinct output.
    #[test]
    fn test_distortion_types_differ() {
        let outs: Vec<Vec<f32>> = DistortionType::ALL.iter().map(|t| render(*t)).collect();
        for i in 0..outs.len() {
            for j in (i + 1)..outs.len() {
                let diff: f32 = outs[i]
                    .iter()
                    .zip(outs[j].iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                assert!(
                    diff > 0.5,
                    "{:?} and {:?} produce identical output",
                    DistortionType::ALL[i],
                    DistortionType::ALL[j]
                );
            }
        }
    }

    /// Regression: every type round-trips through its integer index (used by
    /// the SynthParam, preset and host-parameter plumbing).
    #[test]
    fn test_distortion_type_index_roundtrip() {
        for t in DistortionType::ALL {
            assert_eq!(DistortionType::from_index(t.index()), t);
        }
        // Out-of-range indices fall back to Tube.
        assert_eq!(DistortionType::from_index(99), DistortionType::Tube);
    }

    /// Regression: the wet output must actually be shaped — full mix and
    /// drive must differ from the dry signal.
    #[test]
    fn test_distortion_shapes_signal() {
        let dry = render_to_mix(0.0);
        let wet = render_to_mix(1.0);
        let diff: f32 = dry.iter().zip(wet.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1.0, "wet distortion should differ from dry signal");
    }

    fn render_to_mix(mix: f32) -> Vec<f32> {
        let mut fx = EffectsChain::new(44100);
        fx.set_distortion_enabled(true);
        fx.set_distortion_type(DistortionType::Fuzz);
        fx.set_distortion_drive(0.9);
        fx.set_distortion_mix(mix);
        let mut phase = 0.0f32;
        (0..1024)
            .map(|_| {
                phase += 440.0 / 44100.0;
                let s = (phase * std::f32::consts::TAU).sin() * 0.6;
                fx.process_stereo(StereoSample::from_mono(s)).left
            })
            .collect()
    }
}
