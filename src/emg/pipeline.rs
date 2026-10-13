//! Raw µV samples → per-channel EMG envelope, the input both control schemes
//! decode (gesture.rs for muscle pairs, finger.rs for the finger grid).
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
/// Wide and applied twice: the board's clock puts mains a fraction of a hertz
/// off nominal, where a narrow notch barely attenuates it.
const NOTCH_Q: f64 = 5.0;
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

    fn envelope(&mut self, x: f64) -> f32 {
        let mut y = x;
        for f in &mut self.notch {
            y = f.process(y);
        }
        self.lp.process(y.abs()).max(0.0) as f32
    }
}

pub struct EmgPipeline {
    chains: [Chain; NUM_CHANNELS],
    mains_hz: f32,
    /// Common-average reference on/off (on for single-row wiring with a shared REF).
    pub car: bool,
    envelope: [f32; NUM_CHANNELS],
}

impl EmgPipeline {
    pub fn new() -> Self {
        let mains_hz = 60.0;
        Self { chains: std::array::from_fn(|_| Chain::new(mains_hz)), mains_hz, car: true, envelope: [0.0; NUM_CHANNELS] }
    }

    /// Reset filter state.
    pub fn reset_filters(&mut self) {
        self.chains = std::array::from_fn(|_| Chain::new(self.mains_hz));
        self.envelope = [0.0; NUM_CHANNELS];
    }

    pub fn set_mains(&mut self, hz: f32) {
        self.mains_hz = if hz < 55.0 { 50.0 } else { 60.0 };
        self.reset_filters();
    }

    pub fn process(&mut self, sample: &Sample) {
        let mut hp = [0.0f64; NUM_CHANNELS];
        for (ch, &x) in sample.iter().enumerate() {
            hp[ch] = self.chains[ch].highpass(x);
        }
        if self.car {
            let good: Vec<usize> = (0..NUM_CHANNELS).filter(|&ch| self.chains[ch].contact()).collect();
            if good.len() >= CAR_MIN_CHANNELS {
                let mean = good.iter().map(|&ch| hp[ch]).sum::<f64>() / good.len() as f64;
                hp.iter_mut().for_each(|y| *y -= mean);
            }
        }
        for (ch, &y) in hp.iter().enumerate() {
            let env = self.chains[ch].envelope(y);
            self.envelope[ch] = if self.chains[ch].contact() { env } else { 0.0 };
        }
    }

    /// Per channel: false while the input is railed (electrode off / no contact).
    pub fn contact(&self) -> [bool; NUM_CHANNELS] {
        std::array::from_fn(|ch| self.chains[ch].contact())
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

}
