//! Raw µV samples → per-channel EMG envelope, the input both control schemes
//! decode (gesture.rs, for muscle pairs and the digit array).
//!
//! Per sample: 20 Hz high-pass on every channel → common-average reference
//! (subtract the mean of the channels that have skin contact; this cancels the
//! mains hum and noise a shared REF electrode puts on every channel alike) →
//! wide double notches at mains and its 2nd harmonic → rectify → 5 Hz low-pass.

use super::cyton::{Sample, NUM_CHANNELS, SAMPLE_RATE, UV_PER_COUNT};
use super::filter::{Biquad, BUTTER2_Q, BUTTER4_Q};

const HP_HZ: f32 = 20.0;
/// Envelope smoothing: higher reacts faster to short pulses, lower ripples less.
const ENVELOPE_HZ: f32 = 5.0;
/// Applied twice: still >80 dB deep across the board's ~0.1 Hz clock offset
/// while preserving much more EMG passband around 50–70 Hz and 110–125 Hz.
const NOTCH_Q: f64 = 10.0;
/// Instantaneous post-HP amplitude (µV) above which a channel is excluded from
/// that sample's common-average reference so a loose/spiking electrode cannot
/// smear noise across healthy channels.
const CAR_SPIKE_UV: f64 = 2000.0;
/// Inputs beyond 95% of the ADC range are railed: the electrode is off or has
/// no skin contact.
const RAIL_UV: f32 = 0.95 * 8_388_607.0 * UV_PER_COUNT;
/// Once railed, a channel only counts as back after staying below this for
/// RAIL_HOLD samples, so an electrode hovering at the rail can't flap on and off.
const RECOVER_UV: f32 = 0.90 * 8_388_607.0 * UV_PER_COUNT;
const RAIL_HOLD: u32 = (2.0 * SAMPLE_RATE) as u32;
/// With fewer channels than this the common average would mostly cancel the
/// signal itself, so it's skipped.
const CAR_MIN_CHANNELS: usize = 3;
/// Channel meters before a calibration knows better: rest and full (µV of envelope).
const DEFAULT_REST: f32 = 3.0;
const DEFAULT_MAX: f32 = 80.0;

#[cfg(test)]
fn samples(seconds: f32) -> u32 {
    (seconds * SAMPLE_RATE).round() as u32
}

struct Chain {
    hp: [Biquad; 2],
    notch: [Biquad; 4],
    lp: Biquad,
    primed: bool,
    rail_hold: u32,
}

impl Chain {
    fn new(mains_hz: f32) -> Self {
        let notch = |k: f32| Biquad::notch(SAMPLE_RATE, mains_hz * k, NOTCH_Q);
        Self {
            hp: BUTTER4_Q.map(|q| Biquad::highpass(SAMPLE_RATE, HP_HZ, q)),
            // fundamental + 2nd harmonic (120 Hz is still below Nyquist), each twice
            notch: [notch(1.0), notch(1.0), notch(2.0), notch(2.0)],
            lp: Biquad::lowpass(SAMPLE_RATE, ENVELOPE_HZ, BUTTER2_Q),
            primed: false,
            rail_hold: 0,
        }
    }

    fn contact(&self) -> bool {
        self.rail_hold == 0
    }

    /// Contact check and DC removal.
    fn highpass(&mut self, x: f32) -> f64 {
        let limit = if self.contact() { RAIL_UV } else { RECOVER_UV };
        self.rail_hold = if x.abs() >= limit { RAIL_HOLD } else { self.rail_hold.saturating_sub(1) };
        let mut y = x as f64;
        if !self.primed {
            // Cyton channels can sit at a large DC offset; start the HP in steady state.
            self.hp[0].prime(y);
            self.primed = true;
        }
        for f in &mut self.hp {
            y = f.process(y);
        }
        y
    }

    /// Applies the mains notches and envelope detector; returns `(filtered_uv, envelope_uv)`.
    fn notch_and_envelope(&mut self, x: f64) -> (f32, f32) {
        let mut y = x;
        for f in &mut self.notch {
            y = f.process(y);
        }
        let env = self.lp.process(y.abs()).max(0.0) as f32;
        (y as f32, env)
    }
}

pub struct EmgPipeline {
    chains: [Chain; NUM_CHANNELS],
    mains_hz: f32,
    /// Common-average reference on/off (on for single-row wiring with a shared REF).
    pub car: bool,
    /// Channels flagged bad (e.g. outlier resting noise): excluded from CAR.
    excluded: [bool; NUM_CHANNELS],
    filtered: [f32; NUM_CHANNELS],
    envelope: [f32; NUM_CHANNELS],
}

impl EmgPipeline {
    pub fn new() -> Self {
        let mains_hz = 60.0;
        Self {
            chains: std::array::from_fn(|_| Chain::new(mains_hz)),
            mains_hz,
            car: true,
            excluded: [false; NUM_CHANNELS],
            filtered: [0.0; NUM_CHANNELS],
            envelope: [0.0; NUM_CHANNELS],
        }
    }

    /// Reset filter state (keeps CAR and bad-channel exclusion mask).
    pub fn reset_filters(&mut self) {
        self.chains = std::array::from_fn(|_| Chain::new(self.mains_hz));
        self.filtered = [0.0; NUM_CHANNELS];
        self.envelope = [0.0; NUM_CHANNELS];
    }

    pub fn set_mains(&mut self, hz: f32) {
        self.mains_hz = if hz < 55.0 { 50.0 } else { 60.0 };
        self.reset_filters();
    }

    /// Channels excluded from the common-average reference (e.g. outlier resting noise).
    pub fn set_excluded(&mut self, excluded: [bool; NUM_CHANNELS]) {
        self.excluded = excluded;
    }

    pub fn excluded(&self) -> [bool; NUM_CHANNELS] {
        self.excluded
    }

    pub fn process(&mut self, sample: &Sample) {
        let mut hp = [0.0f64; NUM_CHANNELS];
        for (ch, &x) in sample.iter().enumerate() {
            hp[ch] = self.chains[ch].highpass(x);
        }
        if self.car {
            let mut sum = 0.0;
            let mut count = 0usize;
            for ch in 0..NUM_CHANNELS {
                if self.chains[ch].contact() && !self.excluded[ch] && hp[ch].abs() <= CAR_SPIKE_UV {
                    sum += hp[ch];
                    count += 1;
                }
            }
            if count >= CAR_MIN_CHANNELS {
                let mean = sum / count as f64;
                hp.iter_mut().for_each(|y| *y -= mean);
            }
        }
        for (ch, &y) in hp.iter().enumerate() {
            let (filt, env) = self.chains[ch].notch_and_envelope(y);
            if self.chains[ch].contact() {
                self.filtered[ch] = filt;
                self.envelope[ch] = env;
            } else {
                self.filtered[ch] = 0.0;
                self.envelope[ch] = 0.0;
            }
        }
    }

    /// Per channel: false while the input is railed (electrode off / no contact).
    pub fn contact(&self) -> [bool; NUM_CHANNELS] {
        std::array::from_fn(|ch| self.chains[ch].contact())
    }

    /// Band-limited, notched EMG in µV (before rectification): feeds covariance features.
    pub fn filtered(&self) -> [f32; NUM_CHANNELS] {
        self.filtered
    }

    pub fn envelopes(&self) -> [f32; NUM_CHANNELS] {
        self.envelope
    }

    /// Channel meters on default levels (0 = rest, 1 = a strong contraction).
    pub fn activations(&self) -> [f32; NUM_CHANNELS] {
        self.envelope.map(|e| ((e - DEFAULT_REST) / (DEFAULT_MAX - DEFAULT_REST)).clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-noise in -1..1.
    fn noise(state: &mut u32) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        (*state as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    struct Sim {
        rng: u32,
        n: usize,
        /// Common-mode noise amplitude (µV), like a shared REF electrode picks up.
        common: f32,
        railed: Option<usize>,
    }

    impl Sim {
        fn new(seed: u32) -> Self {
            Self { rng: seed, n: 0, common: 0.0, railed: None }
        }

        fn sample(&mut self, burst: Option<usize>) -> Sample {
            let t = self.n as f32 / SAMPLE_RATE;
            self.n += 1;
            // mains slightly off nominal, as the Cyton's clock sees it, identical on every channel
            let hum = 300.0 * (2.0 * std::f32::consts::PI * 59.9 * t).sin();
            let common = self.common * noise(&mut self.rng);
            std::array::from_fn(|ch| {
                if Some(ch) == self.railed {
                    return -RAIL_UV - 1000.0;
                }
                let d = if burst == Some(ch) { 1.0 } else { 0.0 };
                let emg = d * 150.0 * noise(&mut self.rng);
                -4000.0 - 1000.0 * ch as f32 + hum + common + 2.0 * noise(&mut self.rng) + emg
            })
        }

        fn run(&mut self, p: &mut EmgPipeline, seconds: f32, burst: Option<usize>) {
            for _ in 0..samples(seconds) {
                let s = self.sample(burst);
                p.process(&s);
            }
        }
    }

    #[test]
    fn hum_and_dc_are_rejected_and_a_burst_shows() {
        let mut p = EmgPipeline::new();
        let mut sim = Sim::new(12345);
        sim.run(&mut p, 2.0, None);
        assert!(p.envelopes().iter().all(|&e| e < 5.0), "rest envelope {:?}", p.envelopes());

        sim.run(&mut p, 1.0, Some(4));
        assert!(p.envelopes()[4] > 40.0, "burst envelope {:?}", p.envelopes());
    }

    #[test]
    fn common_average_cancels_shared_reference_noise() {
        let rest_level = |car: bool| {
            let mut p = EmgPipeline::new();
            p.car = car;
            let mut sim = Sim::new(99);
            sim.common = 60.0;
            sim.run(&mut p, 3.0, None);
            p.envelopes().iter().sum::<f32>() / NUM_CHANNELS as f32
        };
        let (without, with) = (rest_level(false), rest_level(true));
        assert!(without > 15.0, "common noise should dominate without CAR: {without}");
        assert!(with < 3.0, "CAR should remove it: {with}");
    }

    #[test]
    fn railed_channel_is_flagged_and_excluded() {
        let mut p = EmgPipeline::new();
        let mut sim = Sim::new(5);
        sim.common = 60.0;
        sim.railed = Some(1);
        sim.run(&mut p, 3.0, None);
        assert!(!p.contact()[1]);
        assert!(p.contact().iter().enumerate().all(|(ch, &ok)| ok || ch == 1));
        assert_eq!(p.envelopes()[1], 0.0);
        // the railed channel must not leak its huge offset into the average
        assert!(p.envelopes().iter().all(|&e| e < 3.0), "{:?}", p.envelopes());
    }

    #[test]
    fn channel_hovering_at_the_rail_stays_off() {
        let mut p = EmgPipeline::new();
        let mut sim = Sim::new(11);
        for i in 0..samples(4.0) {
            let mut s = sim.sample(None);
            // like the recorded brachioradialis: dips in and out of the rail every ~0.3 s
            s[1] = if (i / 75) % 2 == 0 { -RAIL_UV - 500.0 } else { -RAIL_UV + 2000.0 + 500.0 * noise(&mut sim.rng) };
            p.process(&s);
            if i > samples(0.1) {
                assert!(!p.contact()[1], "flapped back on at sample {i}");
            }
        }
    }

    #[test]
    fn excluded_and_spiking_channels_do_not_pollute_car() {
        let mut p = EmgPipeline::new();
        let mut sim = Sim::new(77);
        p.set_excluded([false, false, true, false, false, false, false, false]);
        for _ in 0..samples(2.0) {
            let mut s = sim.sample(None);
            // CH3 has huge non-railed broadband noise (loose contact)
            s[2] += 800.0 * noise(&mut sim.rng);
            p.process(&s);
        }
        for ch in (0..NUM_CHANNELS).filter(|&c| c != 2) {
            assert!(p.envelopes()[ch] < 4.0, "CH{ch} polluted by excluded CH2: {:?}", p.envelopes());
        }
    }
}
