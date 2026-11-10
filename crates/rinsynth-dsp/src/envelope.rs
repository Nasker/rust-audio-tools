use rinsynth_model::types::{Amplitude, SampleRate};

/// Current state of an envelope
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnvelopeState {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

/// Trait for all envelope types.
/// Envelopes shape the amplitude of a sound over time.
pub trait Envelope: Send + Sync {
    /// Get the next amplitude value
    #[link_section = ".iram1.text"]

    fn next_amplitude(&mut self) -> Amplitude;

    /// Trigger the envelope (note on)
    fn trigger(&mut self);

    /// Release the envelope (note off)
    fn release(&mut self);

    /// Check if the envelope has finished (returned to idle)
    fn is_finished(&self) -> bool;

    /// Get the current state
    fn state(&self) -> EnvelopeState;

    /// Reset the envelope to initial state
    fn reset(&mut self);

    /// Set the sample rate
    fn set_sample_rate(&mut self, sample_rate: SampleRate);

    /// Set attack time in seconds
    fn set_attack(&mut self, _attack_time: f32) {}

    /// Set decay time in seconds
    fn set_decay(&mut self, _decay_time: f32) {}

    /// Set sustain level (0.0 to 1.0)
    fn set_sustain(&mut self, _sustain_level: f32) {}

    /// Set release time in seconds
    fn set_release(&mut self, _release_time: f32) {}

    /// Get attack time in seconds
    fn attack(&self) -> f32 {
        0.0
    }

    /// Get decay time in seconds
    fn decay(&self) -> f32 {
        0.0
    }

    /// Get sustain level
    fn sustain(&self) -> f32 {
        1.0
    }

    /// Get release time in seconds
    fn release_time(&self) -> f32 {
        0.0
    }

    /// Current output amplitude (0.0-1.0) — used by the CPU governor to shed
    /// the quietest voice, whose fade-out is least audible. Default 1.0 so
    /// unimplemented envelopes never look like the cheapest shed candidate.
    fn level(&self) -> f32 {
        1.0
    }
}

/// Simple Attack-Release envelope
/// Attack: time to reach full amplitude
/// Release: time to fade to zero after note off
pub struct AREnvelope {
    attack_time: f32,
    release_time: f32,
    sample_rate: SampleRate,
    state: EnvelopeState,
    current_amplitude: Amplitude,
    attack_increment: f32,
    release_decrement: f32,
}

impl AREnvelope {
    pub fn new(attack_time: f32, release_time: f32, sample_rate: SampleRate) -> Self {
        let mut env = Self {
            attack_time,
            release_time,
            sample_rate,
            state: EnvelopeState::Idle,
            current_amplitude: 0.0,
            attack_increment: 0.0,
            release_decrement: 0.0,
        };
        env.update_increments();
        env
    }

    pub fn set_attack(&mut self, attack_time: f32) {
        self.attack_time = attack_time.max(0.001);
        self.update_increments();
    }

    pub fn set_release(&mut self, release_time: f32) {
        self.release_time = release_time.max(0.001);
        self.update_increments();
    }

    fn update_increments(&mut self) {
        let samples_per_second = self.sample_rate as f32;
        self.attack_increment = 1.0 / (self.attack_time * samples_per_second);
        self.release_decrement = 1.0 / (self.release_time * samples_per_second);
    }
}

impl Envelope for AREnvelope {
    #[link_section = ".iram1.text"]

    fn next_amplitude(&mut self) -> Amplitude {
        match self.state {
            EnvelopeState::Idle => {
                self.current_amplitude = 0.0;
            }
            EnvelopeState::Attack => {
                self.current_amplitude += self.attack_increment;
                if self.current_amplitude >= 1.0 {
                    self.current_amplitude = 1.0;
                    self.state = EnvelopeState::Sustain;
                }
            }
            EnvelopeState::Sustain => {
                self.current_amplitude = 1.0;
            }
            EnvelopeState::Release => {
                self.current_amplitude -= self.release_decrement;
                if self.current_amplitude <= 0.0 {
                    self.current_amplitude = 0.0;
                    self.state = EnvelopeState::Idle;
                }
            }
            EnvelopeState::Decay => {
                // AR envelope doesn't use decay, but included for trait completeness
                self.state = EnvelopeState::Sustain;
            }
        }
        self.current_amplitude
    }

    fn trigger(&mut self) {
        self.state = EnvelopeState::Attack;
    }

    fn release(&mut self) {
        if self.state != EnvelopeState::Idle {
            self.state = EnvelopeState::Release;
        }
    }

    fn is_finished(&self) -> bool {
        self.state == EnvelopeState::Idle
    }

    fn state(&self) -> EnvelopeState {
        self.state
    }

    fn reset(&mut self) {
        self.state = EnvelopeState::Idle;
        self.current_amplitude = 0.0;
    }

    fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
        self.update_increments();
    }

    fn set_attack(&mut self, attack_time: f32) {
        self.attack_time = attack_time.max(0.001);
        self.update_increments();
    }

    fn set_release(&mut self, release_time: f32) {
        self.release_time = release_time.max(0.001);
        self.update_increments();
    }

    fn attack(&self) -> f32 {
        self.attack_time
    }

    fn release_time(&self) -> f32 {
        self.release_time
    }

    fn level(&self) -> f32 {
        self.current_amplitude
    }
}

/// Full ADSR envelope for future use
pub struct ADSREnvelope {
    attack_time: f32,
    decay_time: f32,
    sustain_level: Amplitude,
    release_time: f32,
    sample_rate: SampleRate,
    state: EnvelopeState,
    current_amplitude: Amplitude,
    attack_increment: f32,
    decay_decrement: f32,
    release_decrement: f32,
}

impl ADSREnvelope {
    pub fn new(
        attack_time: f32,
        decay_time: f32,
        sustain_level: Amplitude,
        release_time: f32,
        sample_rate: SampleRate,
    ) -> Self {
        let mut env = Self {
            attack_time: attack_time.max(0.001),
            decay_time: decay_time.max(0.001),
            sustain_level: sustain_level.clamp(0.0, 1.0),
            release_time: release_time.max(0.001),
            sample_rate,
            state: EnvelopeState::Idle,
            current_amplitude: 0.0,
            attack_increment: 0.0,
            decay_decrement: 0.0,
            release_decrement: 0.0,
        };
        env.update_increments();
        env
    }

    /// Create with default parameters (quick attack, medium decay, 70% sustain, medium release)
    pub fn default_adsr(sample_rate: SampleRate) -> Self {
        Self::new(0.01, 0.1, 0.7, 0.2, sample_rate)
    }

    fn update_increments(&mut self) {
        let samples_per_second = self.sample_rate as f32;
        self.attack_increment = 1.0 / (self.attack_time * samples_per_second);
        self.decay_decrement = (1.0 - self.sustain_level) / (self.decay_time * samples_per_second);
        // Release decrement is recalculated in release() based on current amplitude
        self.release_decrement = self.sustain_level / (self.release_time * samples_per_second);
    }
}

impl Envelope for ADSREnvelope {
    #[link_section = ".iram1.text"]

    fn next_amplitude(&mut self) -> Amplitude {
        match self.state {
            EnvelopeState::Idle => {
                self.current_amplitude = 0.0;
            }
            EnvelopeState::Attack => {
                self.current_amplitude += self.attack_increment;
                if self.current_amplitude >= 1.0 {
                    self.current_amplitude = 1.0;
                    self.state = EnvelopeState::Decay;
                }
            }
            EnvelopeState::Decay => {
                self.current_amplitude -= self.decay_decrement;
                if self.current_amplitude <= self.sustain_level {
                    self.current_amplitude = self.sustain_level;
                    self.state = EnvelopeState::Sustain;
                }
            }
            EnvelopeState::Sustain => {
                self.current_amplitude = self.sustain_level;
            }
            EnvelopeState::Release => {
                self.current_amplitude -= self.release_decrement;
                if self.current_amplitude <= 0.0 {
                    self.current_amplitude = 0.0;
                    self.state = EnvelopeState::Idle;
                }
            }
        }
        self.current_amplitude
    }

    fn trigger(&mut self) {
        self.state = EnvelopeState::Attack;
    }

    fn release(&mut self) {
        if self.state != EnvelopeState::Idle {
            // Recalculate release decrement based on current amplitude
            let samples_per_second = self.sample_rate as f32;
            self.release_decrement =
                self.current_amplitude / (self.release_time.max(0.001) * samples_per_second);
            self.state = EnvelopeState::Release;
        }
    }

    fn is_finished(&self) -> bool {
        self.state == EnvelopeState::Idle
    }

    fn state(&self) -> EnvelopeState {
        self.state
    }

    fn reset(&mut self) {
        self.state = EnvelopeState::Idle;
        self.current_amplitude = 0.0;
    }

    fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
        self.update_increments();
    }

    fn set_attack(&mut self, attack_time: f32) {
        self.attack_time = attack_time.max(0.001);
        self.update_increments();
    }

    fn set_decay(&mut self, decay_time: f32) {
        self.decay_time = decay_time.max(0.001);
        self.update_increments();
    }

    fn set_sustain(&mut self, sustain_level: f32) {
        self.sustain_level = sustain_level.clamp(0.0, 1.0);
        self.update_increments();
    }

    fn set_release(&mut self, release_time: f32) {
        self.release_time = release_time.max(0.001);
        self.update_increments();
    }

    fn attack(&self) -> f32 {
        self.attack_time
    }

    fn decay(&self) -> f32 {
        self.decay_time
    }

    fn sustain(&self) -> f32 {
        self.sustain_level
    }

    fn release_time(&self) -> f32 {
        self.release_time
    }

    fn level(&self) -> f32 {
        self.current_amplitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Record envelope samples and state transitions for analysis
    struct EnvelopeRecording {
        samples: Vec<f32>,
        states: Vec<EnvelopeState>,
        events: Vec<(usize, String)>, // (sample_index, event_description)
        sample_rate: SampleRate,
    }

    impl EnvelopeRecording {
        fn new(sample_rate: SampleRate) -> Self {
            Self {
                samples: Vec::new(),
                states: Vec::new(),
                events: Vec::new(),
                sample_rate,
            }
        }

        fn record_sample(&mut self, amplitude: f32, state: EnvelopeState) {
            self.samples.push(amplitude);
            self.states.push(state);
        }

        fn add_event(&mut self, sample_index: usize, description: &str) {
            self.events.push((sample_index, description.to_string()));
        }

        fn duration_seconds(&self) -> f32 {
            self.samples.len() as f32 / self.sample_rate as f32
        }

        /// Check for discontinuities (jumps) in the envelope
        fn find_discontinuities(&self, threshold: f32) -> Vec<(usize, f32, f32, f32)> {
            let mut discontinuities = Vec::new();
            for i in 1..self.samples.len() {
                let prev = self.samples[i - 1];
                let curr = self.samples[i];
                let delta = (curr - prev).abs();
                // Expected maximum change per sample for a smooth envelope
                // At 44100 Hz, a 1ms attack = 44.1 samples to go 0->1, so max delta ~0.023
                let expected_max_delta = 0.05; // Generous threshold
                if delta > expected_max_delta && delta > threshold {
                    discontinuities.push((i, prev, curr, delta));
                }
            }
            discontinuities
        }

        /// Check for state transition discontinuities
        fn find_state_transition_issues(&self) -> Vec<(usize, EnvelopeState, f32, f32)> {
            let mut issues = Vec::new();
            for i in 1..self.samples.len() {
                let prev_state = self.states[i - 1];
                let curr_state = self.states[i];
                let prev_amp = self.samples[i - 1];
                let curr_amp = self.samples[i];

                // Check for suspicious state transitions
                match (prev_state, curr_state) {
                    // Attack to Release should not cause a jump
                    (EnvelopeState::Attack, EnvelopeState::Release) => {
                        if (curr_amp - prev_amp).abs() > 0.1 {
                            issues.push((i, curr_state, prev_amp, curr_amp));
                        }
                    }
                    // Any transition to Idle should end at 0
                    (_, EnvelopeState::Idle) => {
                        if prev_amp > 0.01 && curr_amp == 0.0 {
                            // Hard cut to zero - potential click
                            issues.push((i, curr_state, prev_amp, curr_amp));
                        }
                    }
                    _ => {}
                }
            }
            issues
        }
    }

    fn record_envelope_sequence<E: Envelope>(
        envelope: &mut E,
        events: &[(f32, bool)], // (time_seconds, is_note_on)
        duration_seconds: f32,
        sample_rate: SampleRate,
    ) -> EnvelopeRecording {
        let total_samples = (duration_seconds * sample_rate as f32) as usize;
        let mut recording = EnvelopeRecording::new(sample_rate);

        let mut event_index = 0;
        for sample in 0..total_samples {
            // Check for events at this sample
            while event_index < events.len() {
                let (event_time, is_note_on) = events[event_index];
                let event_sample = (event_time * sample_rate as f32) as usize;
                if event_sample == sample {
                    if is_note_on {
                        envelope.trigger();
                        recording.add_event(sample, "NOTE ON");
                    } else {
                        envelope.release();
                        recording.add_event(sample, "NOTE OFF");
                    }
                    event_index += 1;
                } else {
                    break;
                }
            }

            let amp = envelope.next_amplitude();
            recording.record_sample(amp, envelope.state());
        }

        recording
    }

    fn plot_envelope(recording: &EnvelopeRecording, filename: &str, title: &str) -> PathBuf {
        use plotters::prelude::*;

        let path = PathBuf::from(filename);
        {
            let root = BitMapBackend::new(&path, (1200, 600)).into_drawing_area();
            root.fill(&WHITE).unwrap();

            let duration = recording.duration_seconds();
            let min_amp = recording
                .samples
                .iter()
                .fold(f32::INFINITY, |a, &b| a.min(b));
            let max_amp = recording.samples.iter().fold(0.0_f32, |a, &b| a.max(b));

            let mut chart = ChartBuilder::on(&root)
                .caption(title, ("sans-serif", 30))
                .margin(10)
                .x_label_area_size(40)
                .y_label_area_size(40)
                .build_cartesian_2d(0.0f32..duration, -0.1f32..1.1f32)
                .unwrap();

            chart
                .configure_mesh()
                .x_desc("Time (seconds)")
                .y_desc("Amplitude")
                .draw()
                .unwrap();

            // Plot envelope curve
            let points: Vec<(f32, f32)> = recording
                .samples
                .iter()
                .enumerate()
                .map(|(i, &amp)| (i as f32 / recording.sample_rate as f32, amp))
                .collect();

            chart
                .draw_series(LineSeries::new(points, &BLUE))
                .unwrap()
                .label("Envelope")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

            // Draw state transition points
            let mut last_state = EnvelopeState::Idle;
            let state_colors = [
                (EnvelopeState::Idle, RGBColor(200, 200, 200)),
                (EnvelopeState::Attack, RGBColor(0, 255, 0)),
                (EnvelopeState::Decay, RGBColor(255, 255, 0)),
                (EnvelopeState::Sustain, RGBColor(0, 0, 255)),
                (EnvelopeState::Release, RGBColor(255, 0, 0)),
            ];

            for (i, &state) in recording.states.iter().enumerate() {
                if state != last_state {
                    let x = i as f32 / recording.sample_rate as f32;
                    let color = state_colors
                        .iter()
                        .find(|(s, _)| *s == state)
                        .map(|(_, c)| *c)
                        .unwrap_or(BLACK);
                    chart
                        .draw_series(std::iter::once(Circle::new(
                            (x, recording.samples[i]),
                            3,
                            color.filled(),
                        )))
                        .unwrap();
                    last_state = state;
                }
            }

            // Draw event markers
            for (sample_idx, event) in &recording.events {
                let x = *sample_idx as f32 / recording.sample_rate as f32;
                let color = if event.contains("ON") { GREEN } else { RED };
                chart
                    .draw_series(std::iter::once(TriangleMarker::new(
                        (x, 1.05),
                        10,
                        color.filled(),
                    )))
                    .unwrap();
            }

            // Draw legend for events
            chart
                .configure_series_labels()
                .background_style(WHITE.mix(0.8))
                .border_style(BLACK)
                .draw()
                .unwrap();

            // Add text annotation
            let text_style = ("sans-serif", 15).into_font().color(&BLACK);
            root.draw_text(
                &format!(
                    "Min: {:.4}, Max: {:.4}, Samples: {}",
                    min_amp,
                    max_amp,
                    recording.samples.len()
                ),
                &text_style,
                (50, 50),
            )
            .unwrap();

            // Report discontinuities
            let discontinuities = recording.find_discontinuities(0.05);
            if !discontinuities.is_empty() {
                let disc_text = format!("⚠ {} discontinuities detected!", discontinuities.len());
                let warn_style = ("sans-serif", 15).into_font().color(&RED);
                root.draw_text(&disc_text, &warn_style, (50, 70)).unwrap();
            }

            root.present().unwrap();
        } // drop root and release borrow on path
        path
    }

    #[test]
    fn test_ar_envelope_lifecycle() {
        let mut env = AREnvelope::new(0.01, 0.01, 44100);

        assert_eq!(env.state(), EnvelopeState::Idle);

        env.trigger();
        assert_eq!(env.state(), EnvelopeState::Attack);

        // Run through attack
        for _ in 0..500 {
            env.next_amplitude();
        }

        env.release();
        assert_eq!(env.state(), EnvelopeState::Release);

        // Run through release
        for _ in 0..500 {
            env.next_amplitude();
        }

        assert!(env.is_finished());
    }

    #[test]
    fn test_adsr_envelope_lifecycle() {
        let mut env = ADSREnvelope::new(0.01, 0.05, 0.5, 0.01, 44100);

        assert_eq!(env.state(), EnvelopeState::Idle);
        assert_eq!(env.next_amplitude(), 0.0);

        // Trigger attack
        env.trigger();
        assert_eq!(env.state(), EnvelopeState::Attack);

        // Run through attack phase
        for _ in 0..500 {
            env.next_amplitude();
        }
        assert_eq!(env.state(), EnvelopeState::Decay);

        // Run through decay phase
        for _ in 0..3000 {
            env.next_amplitude();
        }
        assert_eq!(env.state(), EnvelopeState::Sustain);

        // Sustain level should be ~0.5
        let amp = env.next_amplitude();
        assert!((amp - 0.5).abs() < 0.01);

        // Release
        env.release();
        assert_eq!(env.state(), EnvelopeState::Release);

        // Run through release
        for _ in 0..500 {
            env.next_amplitude();
        }

        assert!(env.is_finished());
    }

    #[test]
    fn test_adsr_setters() {
        let mut env = ADSREnvelope::default_adsr(44100);

        env.set_attack(0.5);
        assert_eq!(env.attack(), 0.5);

        env.set_decay(0.3);
        assert_eq!(env.decay(), 0.3);

        env.set_sustain(0.6);
        assert_eq!(env.sustain(), 0.6);

        env.set_release(1.0);
        assert_eq!(env.release_time(), 1.0);
    }

    // ============================================================================
    // ENVELOPE VISUALIZATION TESTS - Generate plots for debugging clicks
    // ============================================================================

    /// Test: Plot simple ADSR envelope with note on/off
    /// Run with: cargo test plot_adsr_simple -- --nocapture
    #[test]
    fn plot_adsr_simple() {
        let sample_rate = 44100;
        let mut env = ADSREnvelope::new(0.1, 0.2, 0.7, 0.3, sample_rate);

        // Events: (time_seconds, is_note_on)
        let events = vec![
            (0.0, true),  // Note on at start
            (1.0, false), // Note off after 1 second
        ];

        let recording = record_envelope_sequence(&mut env, &events, 2.0, sample_rate);
        let path = plot_envelope(
            &recording,
            "test_adsr_simple.png",
            "ADSR Envelope - Simple Note On/Off",
        );

        println!("Plot saved to: {}", path.display());

        // Assert no discontinuities
        let discontinuities = recording.find_discontinuities(0.05);
        assert!(
            discontinuities.is_empty(),
            "Found {} discontinuities in envelope! This could cause clicks.\n{:?}",
            discontinuities.len(),
            discontinuities
        );
    }

    /// Test: Plot AR envelope with quick note on/off
    #[test]
    fn plot_ar_envelope() {
        let sample_rate = 44100;
        let mut env = AREnvelope::new(0.05, 0.2, sample_rate);

        let events = vec![
            (0.0, true),  // Note on
            (0.5, false), // Note off during sustain
        ];

        let recording = record_envelope_sequence(&mut env, &events, 1.0, sample_rate);
        let path = plot_envelope(
            &recording,
            "test_ar_envelope.png",
            "AR Envelope - Attack/Release",
        );

        println!("Plot saved to: {}", path.display());

        let discontinuities = recording.find_discontinuities(0.05);
        assert!(
            discontinuities.is_empty(),
            "Found {} discontinuities!",
            discontinuities.len()
        );
    }

    /// Test: Multiple note on/off cycles to check for continuity issues
    #[test]
    fn plot_adsr_multiple_cycles() {
        let sample_rate = 44100;
        let mut env = ADSREnvelope::new(0.05, 0.1, 0.6, 0.15, sample_rate);

        // Multiple quick note cycles
        let events = vec![
            (0.0, true),
            (0.3, false),
            (0.5, true), // Re-trigger during release
            (0.8, false),
            (1.0, true),
            (1.5, false),
        ];

        let recording = record_envelope_sequence(&mut env, &events, 2.0, sample_rate);
        let path = plot_envelope(
            &recording,
            "test_adsr_multiple_cycles.png",
            "ADSR - Multiple Note Cycles",
        );

        println!("Plot saved to: {}", path.display());

        // Check for discontinuities at re-trigger points
        let discontinuities = recording.find_discontinuities(0.05);
        assert!(
            discontinuities.is_empty(),
            "Found {} discontinuities at re-trigger points!",
            discontinuities.len()
        );
    }

    /// Test: Very short attack time (potential click source)
    #[test]
    fn plot_adsr_fast_attack() {
        let sample_rate = 44100;
        let mut env = ADSREnvelope::new(0.001, 0.1, 0.5, 0.1, sample_rate);

        let events = vec![(0.0, true), (0.5, false)];

        let recording = record_envelope_sequence(&mut env, &events, 1.0, sample_rate);
        let path = plot_envelope(
            &recording,
            "test_adsr_fast_attack.png",
            "ADSR - Fast Attack (1ms)",
        );

        println!("Plot saved to: {}", path.display());

        // Fast attack is expected, but should still be smooth
        let discontinuities = recording.find_discontinuities(0.1);
        assert!(
            discontinuities.is_empty(),
            "Fast attack caused {} discontinuities!",
            discontinuities.len()
        );
    }

    /// Test: Very short release time (common click source)
    #[test]
    fn plot_adsr_fast_release() {
        let sample_rate = 44100;
        let mut env = ADSREnvelope::new(0.05, 0.1, 0.8, 0.001, sample_rate);

        let events = vec![
            (0.0, true),
            (0.5, false), // Fast release
        ];

        let recording = record_envelope_sequence(&mut env, &events, 1.0, sample_rate);
        let path = plot_envelope(
            &recording,
            "test_adsr_fast_release.png",
            "ADSR - Fast Release (1ms)",
        );

        println!("Plot saved to: {}", path.display());

        // Check transition to idle
        let issues = recording.find_state_transition_issues();
        assert!(
            issues.is_empty(),
            "Found {} state transition issues! Possible click sources.\n{:?}",
            issues.len(),
            issues
        );
    }

    /// Test: Release from different sustain levels
    #[test]
    fn plot_adsr_release_from_various_levels() {
        let sample_rate = 44100;

        // Test with different sustain levels
        for (i, sustain) in [0.2, 0.5, 0.8].iter().enumerate() {
            let mut env = ADSREnvelope::new(0.02, 0.05, *sustain, 0.2, sample_rate);

            let events = vec![(0.0, true), (0.2, false)];

            let recording = record_envelope_sequence(&mut env, &events, 0.6, sample_rate);
            let filename = format!("test_adsr_release_sustain_{}.png", i);
            let title = format!("ADSR Release from {} sustain", sustain);
            let path = plot_envelope(&recording, &filename, &title);

            println!("Plot saved to: {}", path.display());
        }
    }

    /// Test: Compare AR vs ADSR envelopes side by side
    #[test]
    fn plot_envelope_comparison() {
        let sample_rate = 44100;

        // AR envelope
        let mut ar_env = AREnvelope::new(0.1, 0.3, sample_rate);
        let ar_events = vec![(0.0, true), (0.5, false)];
        let ar_recording = record_envelope_sequence(&mut ar_env, &ar_events, 1.0, sample_rate);
        let ar_path = plot_envelope(&ar_recording, "test_ar_comparison.png", "AR Envelope");
        println!("AR plot saved to: {}", ar_path.display());

        // ADSR envelope
        let mut adsr_env = ADSREnvelope::new(0.1, 0.2, 0.7, 0.3, sample_rate);
        let adsr_recording = record_envelope_sequence(&mut adsr_env, &ar_events, 1.0, sample_rate);
        let adsr_path = plot_envelope(&adsr_recording, "test_adsr_comparison.png", "ADSR Envelope");
        println!("ADSR plot saved to: {}", adsr_path.display());
    }

    /// Test: Analyze envelope for potential click sources
    #[test]
    fn analyze_envelope_for_clicks() {
        let sample_rate = 44100;
        let mut env = ADSREnvelope::new(0.05, 0.1, 0.5, 0.15, sample_rate);

        // Various edge cases
        let events = vec![
            (0.00, true),
            (0.05, false), // Very short note (during attack)
            (0.20, true),  // Re-trigger
            (0.30, false), // Short note (during decay)
            (0.50, true),
            (1.00, false), // Normal length note
        ];

        let recording = record_envelope_sequence(&mut env, &events, 1.5, sample_rate);
        let path = plot_envelope(
            &recording,
            "test_envelope_click_analysis.png",
            "Envelope Click Analysis",
        );

        println!("Analysis plot saved to: {}", path.display());
        println!("\n=== CLICK ANALYSIS ===");

        // Check for discontinuities
        let discontinuities = recording.find_discontinuities(0.05);
        if discontinuities.is_empty() {
            println!("✓ No discontinuities detected");
        } else {
            println!("⚠ Found {} discontinuities:", discontinuities.len());
            for (idx, prev, curr, delta) in &discontinuities {
                let time = *idx as f32 / sample_rate as f32;
                println!("  At {:.4}s: {} -> {} (jump: {})", time, prev, curr, delta);
            }
        }

        // Check state transitions
        let transition_issues = recording.find_state_transition_issues();
        if transition_issues.is_empty() {
            println!("✓ No state transition issues");
        } else {
            println!(
                "⚠ Found {} state transition issues:",
                transition_issues.len()
            );
            for (idx, state, prev_amp, curr_amp) in &transition_issues {
                let time = *idx as f32 / sample_rate as f32;
                println!(
                    "  At {:.4}s to {:?}: {} -> {} (potential click!)",
                    time, state, prev_amp, curr_amp
                );
            }
        }

        assert!(
            discontinuities.is_empty() && transition_issues.is_empty(),
            "Envelope has potential click sources!"
        );
    }
}
