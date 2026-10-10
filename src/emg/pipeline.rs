//! Raw µV samples → per-channel EMG envelope → normalised activation →
//! joint velocity commands from antagonist channel pairs.
//!
//! Per sample: 20 Hz high-pass on every channel → common-average reference
//! (subtract the mean of the channels that have skin contact; this cancels the
//! mains hum and noise a shared REF electrode puts on every channel alike) →
//! wide double notches at mains and its 2nd harmonic → rectify → 5 Hz low-pass.
//!
//! Tuned for intention pulses: brief, light isometric tension with the arm
//! held still. Calibration records that light level as full activation, and
//! each joint uses hysteresis so a pulse near the threshold doesn't flicker.
//!
//! Two ways to turn envelopes into joint commands:
//! - pattern (default once calibrated): the decoder classifies the pattern
//!   across all channels into one intended movement, so crosstalk onto other
//!   electrodes doesn't move other joints;
//! - direct: each joint follows its own pair of channels.

use super::cyton::{Sample, NUM_CHANNELS, SAMPLE_RATE, UV_PER_COUNT};
use super::decoder::{self, ClassReport, Features, Imputer, Label, Synergy};
use super::filter::{Biquad, BUTTER2_Q, BUTTER4_Q};
use crate::arm::NUM_JOINTS;

const HP_HZ: f32 = 20.0;
/// Envelope smoothing: higher reacts faster to short pulses, lower ripples less.
const ENVELOPE_HZ: f32 = 5.0;
/// A joint starts moving when |a+ − a−| exceeds the threshold and keeps moving
/// until it falls below this fraction of it.
const RELEASE_FRACTION: f32 = 0.5;
/// Smoothing of the pattern decoder's movement weights.
const WEIGHT_TAU_S: f32 = 0.06;
/// Movements that may drive the arm at the same time.
const MAX_SIMULTANEOUS: usize = 2;
/// The second movement only counts if it is at least this fraction of the first;
/// below that it is treated as crosstalk.
const COMPOUND_RATIO: f64 = 0.35;
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
/// Uncalibrated defaults (µV of envelope).
const DEFAULT_REST: f32 = 3.0;
const DEFAULT_MAX: f32 = 80.0;
/// Calibration floors. A rest level of ~0 (channel had no contact) or a pulse
/// barely above rest would make the tiniest noise read as full activation.
const MIN_REST: f32 = 1.0;
const MIN_SPAN: f32 = 3.0;
/// A pulse must reach this multiple of rest to be accepted; otherwise the
/// channel keeps its previous (by default insensitive) level.
const MIN_PULSE_RATIO: f32 = 1.5;
/// Percentile of the recorded envelope used as the rest floor and the max
/// level: above normal ripple, but not thrown by a single spike.
const LEVEL_PERCENTILE: f32 = 0.9;

// Calibration timing, in seconds. Record phases count only *clean* samples, so
// they last until enough good signal has been collected.
const PREPARE_FIRST_S: f32 = 3.0;
const PREPARE_S: f32 = 2.0;
const REST_RECORD_S: f32 = 4.0;
const HOLD_S: f32 = 2.5;
/// Minimum relax time; it then also waits until every channel is back near rest.
const RELAX_S: f32 = 1.5;
/// Continuous relaxed time that ends a relax phase.
const RELAXED_S: f32 = 0.3;
/// Start of each recording that is ignored while the envelope settles.
const SETTLE_S: f32 = 0.5;

// Signal checks that gate the calibration phases.
/// Rest: a channel jumping above this multiple of its recent level is movement…
const STILL_RATIO: f32 = 2.5;
const STILL_MARGIN_UV: f32 = 1.5;
/// …and nothing at rest should be this loud.
const REST_CEILING_UV: f32 = 30.0;
/// Time constant of each channel's recent-level tracker.
const RECENT_TAU_S: f32 = 1.5;
/// Pulse: the cued channel must reach this multiple of its rest floor (and at
/// least MIN_SPAN above it).
const PULSE_CLEAN_RATIO: f32 = 2.0;
/// Relaxed: every channel back within this multiple of its rest floor.
const RELAXED_RATIO: f32 = 2.5;
/// Calibration needs at least this many mapped electrodes with contact.
const MIN_WORKING_CHANNELS: usize = 2;

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

/// One step of a calibration run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CalStep {
    /// Whole arm relaxed: records every channel's rest floor.
    Rest,
    /// Maximum contraction of the muscle under this channel.
    Max(usize),
}

/// Each step is: prepare (read the cue) → record → relax (Max steps only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    Prepare,
    Record,
    Relax,
}

impl CalStep {
    fn duration(self, phase: Phase, first: bool) -> u32 {
        match (self, phase) {
            (_, Phase::Prepare) => samples(if first { PREPARE_FIRST_S } else { PREPARE_S }),
            (CalStep::Rest, Phase::Record) => samples(REST_RECORD_S),
            (CalStep::Max(_), Phase::Record) => samples(HOLD_S),
            (CalStep::Rest, Phase::Relax) => 0,
            (CalStep::Max(_), Phase::Relax) => samples(RELAX_S),
        }
    }

    fn total(self, first: bool) -> u32 {
        [Phase::Prepare, Phase::Record, Phase::Relax].iter().map(|&p| self.duration(p, first)).sum()
    }
}

/// Live signal check during a calibration phase.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Check {
    Clean,
    /// This electrode is railed.
    NoContact(usize),
    /// Rest: this channel jumped (movement, typing, or a noisy electrode).
    Moving(usize),
    /// Pulse: the cued channel isn't above its pulse threshold.
    NoPulse,
    /// Relax: this channel hasn't come back down yet.
    NotRelaxed(usize),
}

/// Where a calibration run is. The clock is the sample count, so the on-screen
/// cue and the recorded window can't drift apart.
#[derive(Clone, Copy, Debug)]
pub struct CalStatus {
    pub step: usize,
    pub steps: usize,
    pub kind: CalStep,
    pub phase: Phase,
    pub phase_progress: f32,
    pub seconds_left: f32,
    pub total_progress: f32,
    /// None while preparing (nothing is checked then).
    pub check: Option<Check>,
    /// How the checked level compares with its limit (1 = at the limit): the cued
    /// channel vs. its pulse threshold, or the worst channel vs. its rest limit.
    pub meter: f32,
    /// Seconds since the phase last made progress.
    pub stuck_seconds: f32,
    /// The cued channel's electrode is offline, so its pulse is being read from
    /// the other channels instead.
    pub proxy: bool,
}

struct CalRun {
    steps: Vec<CalStep>,
    step: usize,
    phase: Phase,
    /// Samples spent in the current phase.
    t: u32,
    /// Record: clean samples collected. Relax: consecutive relaxed samples.
    good: u32,
    /// Samples since the phase last made progress.
    stuck: u32,
    /// Clean envelope samples recorded in the current Record phase, per channel.
    rec: [Vec<f32>; NUM_CHANNELS],
    /// Per channel: how many of the recorded samples it had contact for.
    contact_count: [u32; NUM_CHANNELS],
}

impl CalRun {
    fn duration(&self) -> u32 {
        self.steps[self.step].duration(self.phase, self.step == 0)
    }

    /// Samples of progress made in the current phase (counts towards `duration`).
    fn progress(&self) -> u32 {
        match self.phase {
            Phase::Record => self.good,
            _ => self.t.min(self.duration()),
        }
    }

    /// Moves to the next phase with a non-zero duration; false when the run is over.
    fn advance(&mut self) -> bool {
        self.t = 0;
        self.good = 0;
        self.stuck = 0;
        loop {
            self.phase = match self.phase {
                Phase::Prepare => Phase::Record,
                Phase::Record => Phase::Relax,
                Phase::Relax => {
                    self.step += 1;
                    if self.step >= self.steps.len() {
                        return false;
                    }
                    Phase::Prepare
                }
            };
            if self.duration() > 0 {
                return true;
            }
        }
    }
}

fn percentile(values: &mut [f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let i = ((values.len() - 1) as f32 * p).round() as usize;
    *values.select_nth_unstable_by(i, f32::total_cmp).1
}

pub struct EmgPipeline {
    chains: [Chain; NUM_CHANNELS],
    mains_hz: f32,
    /// Common-average reference on/off (on for single-row wiring with a shared REF).
    pub car: bool,
    envelope: [f32; NUM_CHANNELS],
    /// Slow average of each envelope, the baseline for spotting movement at rest.
    recent: [f32; NUM_CHANNELS],
    /// Channels whose envelope is currently estimated (electrode offline).
    estimated: [bool; NUM_CHANNELS],
    rest: [f32; NUM_CHANNELS],
    max: [f32; NUM_CHANNELS],
    rest_done: bool,
    max_done: [bool; NUM_CHANNELS],
    /// Channel had contact during the rest recording (so its relation to the
    /// others is known); false = never observed, estimated from the decoder.
    seen: [bool; NUM_CHANNELS],
    /// Movement calibrated from the other channels because its own electrode was offline.
    via_others: [bool; NUM_CHANNELS],
    /// (positive-direction channel, negative-direction channel) per joint.
    mapping: [(Option<usize>, Option<usize>); NUM_JOINTS],
    pub threshold: f32,
    pub gain: f32,
    engaged: [bool; NUM_JOINTS],
    inputs: [f32; NUM_JOINTS],
    cal: Option<CalRun>,
    /// Use the pattern decoder when it is trained (otherwise direct channel pairs).
    pub pattern: bool,
    /// Calibration recordings per class, kept so single steps can be redone.
    training: Vec<(Label, Vec<Features>)>,
    /// Estimates offline channels from the working ones.
    imputer: Option<Imputer>,
    /// Synergy decomposition: drives the arm in pattern mode (compound movements).
    synergy: Option<Synergy>,
    report: Vec<ClassReport>,
    /// Smoothed weight of each movement, in `synergy.channels()` order.
    weights: Vec<f64>,
    /// Per movement: past the threshold (with hysteresis).
    move_engaged: Vec<bool>,
    /// Movements currently driving the arm (indices into the weights, at most two).
    active: Vec<usize>,
}

/// What the pattern decoder currently reads.
pub struct Intent {
    /// (channel whose movement it is, intensity) for up to two simultaneous movements.
    pub moves: Vec<(usize, f32)>,
    /// No movement is anywhere near its threshold.
    pub relaxed: bool,
}

impl EmgPipeline {
    pub fn new() -> Self {
        let mains_hz = 60.0;
        Self {
            chains: std::array::from_fn(|_| Chain::new(mains_hz)),
            mains_hz,
            car: true,
            envelope: [0.0; NUM_CHANNELS],
            recent: [0.0; NUM_CHANNELS],
            estimated: [false; NUM_CHANNELS],
            rest: [DEFAULT_REST; NUM_CHANNELS],
            max: [DEFAULT_MAX; NUM_CHANNELS],
            rest_done: false,
            max_done: [false; NUM_CHANNELS],
            seen: [true; NUM_CHANNELS],
            via_others: [false; NUM_CHANNELS],
            mapping: std::array::from_fn(|j| (Some(2 * j), Some(2 * j + 1))),
            threshold: 0.12,
            gain: 1.0,
            engaged: [false; NUM_JOINTS],
            inputs: [0.0; NUM_JOINTS],
            cal: None,
            pattern: true,
            training: Vec::new(),
            imputer: None,
            synergy: None,
            report: Vec::new(),
            weights: Vec::new(),
            move_engaged: Vec::new(),
            active: Vec::new(),
        }
    }

    /// Reset filter state (keeps calibration and mapping).
    pub fn reset_filters(&mut self) {
        self.chains = std::array::from_fn(|_| Chain::new(self.mains_hz));
        self.envelope = [0.0; NUM_CHANNELS];
        self.recent = [0.0; NUM_CHANNELS];
        self.engaged = [false; NUM_JOINTS];
        self.inputs = [0.0; NUM_JOINTS];
        self.clear_decoder_state();
    }

    fn clear_decoder_state(&mut self) {
        self.weights.clear();
        self.move_engaged.clear();
        self.active.clear();
    }

    pub fn set_mains(&mut self, hz: f32) {
        self.mains_hz = if hz < 55.0 { 50.0 } else { 60.0 };
        self.reset_filters();
    }

    pub fn set_mapping(&mut self, joint: usize, pos: Option<usize>, neg: Option<usize>) {
        let valid = |c: Option<usize>| c.filter(|&c| c < NUM_CHANNELS);
        if let Some(m) = self.mapping.get_mut(joint) {
            *m = (valid(pos), valid(neg));
        }
    }

    pub fn mapping(&self) -> &[(Option<usize>, Option<usize>); NUM_JOINTS] {
        &self.mapping
    }

    /// Channels used by the mapping, ascending, without duplicates.
    pub fn used_channels(&self) -> Vec<usize> {
        let mut used: Vec<usize> = self.mapping.iter().flat_map(|&(p, n)| [p, n]).flatten().collect();
        used.sort_unstable();
        used.dedup();
        used
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
        self.estimate_offline_channels();
        let k = 1.0 - (-1.0 / (RECENT_TAU_S * SAMPLE_RATE)).exp();
        for (r, &e) in self.recent.iter_mut().zip(&self.envelope) {
            *r = if *r == 0.0 { e } else { *r + (e - *r) * k };
        }
        self.update_inputs();
        self.calibration_step();
    }

    fn update_inputs(&mut self) {
        if self.pattern && self.synergy.is_some() {
            self.update_pattern_inputs();
        } else {
            self.update_direct_inputs();
        }
    }

    /// Joint and direction (+1 / −1) that a channel's movement drives, from the mapping.
    fn joint_for(&self, ch: usize) -> Option<(usize, f32)> {
        self.mapping.iter().enumerate().find_map(|(j, &(pos, neg))| {
            if pos == Some(ch) {
                Some((j, 1.0))
            } else if neg == Some(ch) {
                Some((j, -1.0))
            } else {
                None
            }
        })
    }

    /// Pattern mode: decompose the activity into the calibrated movements. Each
    /// movement engages past the threshold (hysteresis down to half of it); at
    /// most two drive the arm at once, the second only if it carries a real share
    /// of the mix. Opposite movements on one joint net out.
    fn update_pattern_inputs(&mut self) {
        let Some(syn) = &self.synergy else { return };
        let w = syn.decompose(&self.envelope, &self.contact());
        let channels = syn.channels().to_vec();
        if self.weights.len() != w.len() {
            self.weights = vec![0.0; w.len()];
            self.move_engaged = vec![false; w.len()];
        }
        let alpha = 1.0 - (-1.0 / (WEIGHT_TAU_S * SAMPLE_RATE) as f64).exp();
        for (s, v) in self.weights.iter_mut().zip(&w) {
            *s += (v - *s) * alpha;
        }

        let thr = self.threshold.clamp(0.0, 0.95) as f64;
        let release = thr * RELEASE_FRACTION as f64;
        for (e, &v) in self.move_engaged.iter_mut().zip(&self.weights) {
            *e = v > if *e { release } else { thr };
        }
        let mut active: Vec<usize> = (0..self.weights.len()).filter(|&k| self.move_engaged[k]).collect();
        active.sort_by(|&a, &b| self.weights[b].total_cmp(&self.weights[a]));
        active.truncate(MAX_SIMULTANEOUS);
        if active.len() == 2 && self.weights[active[1]] < COMPOUND_RATIO * self.weights[active[0]] {
            active.pop(); // the smaller one is just crosstalk from the bigger
        }

        self.inputs = [0.0; NUM_JOINTS];
        for &k in &active {
            if let Some((j, dir)) = self.joint_for(channels[k]) {
                let t = ((self.weights[k] - release) / (1.0 - release)) as f32;
                self.inputs[j] = (self.inputs[j] + dir * t * self.gain).clamp(-1.0, 1.0);
            }
        }
        self.active = active;
    }

    fn update_direct_inputs(&mut self) {
        let a = self.activations();
        let act = |c: Option<usize>| c.map_or(0.0, |c| a[c]);
        let thr = self.threshold.clamp(0.0, 0.95);
        let release = thr * RELEASE_FRACTION;
        for (j, &(pos, neg)) in self.mapping.iter().enumerate() {
            let d = act(pos) - act(neg);
            self.engaged[j] = d.abs() > if self.engaged[j] { release } else { thr };
            self.inputs[j] = if self.engaged[j] {
                (d.signum() * (d.abs() - release) / (1.0 - release) * self.gain).clamp(-1.0, 1.0)
            } else {
                0.0
            };
        }
    }

    /// Per channel: false while the input is railed (electrode off / no contact).
    pub fn contact(&self) -> [bool; NUM_CHANNELS] {
        std::array::from_fn(|ch| self.chains[ch].contact())
    }

    /// Per channel: true while its envelope is an estimate from the other channels.
    pub fn estimated(&self) -> [bool; NUM_CHANNELS] {
        self.estimated
    }

    /// Once the decoder is trained, fill in channels whose electrode is offline
    /// with an estimate from the working ones, so both decoders keep working.
    /// Not during calibration, which must record only real signal.
    ///
    /// - A channel that had contact during calibration is predicted from how it
    ///   varied with the others (class-conditional Gaussian mean).
    /// - A channel that was never observed has no such relation, but its own
    ///   movement was still trained from the other channels: its activation is
    ///   that movement's weight in the synergy decomposition.
    fn estimate_offline_channels(&mut self) {
        self.estimated = [false; NUM_CHANNELS];
        let contact = self.contact();
        let missing = contact.map(|c| !c);
        if self.cal.is_some() || !missing.iter().any(|&m| m) {
            return;
        }
        let Some(imputer) = &mut self.imputer else { return };
        let mut x = decoder::features(&self.envelope);
        if !imputer.impute(&mut x, &missing) {
            return;
        }
        let weights = self.synergy.as_ref().map(|s| (s.channels().to_vec(), s.decompose(&self.envelope, &contact)));
        for ch in (0..NUM_CHANNELS).filter(|&c| missing[c]) {
            if self.seen[ch] {
                self.envelope[ch] = decoder::envelope_from_feature(x[ch]);
            } else if let Some(k) = weights.as_ref().and_then(|(chs, _)| chs.iter().position(|&c| c == ch)) {
                let a = weights.as_ref().unwrap().1[k].clamp(0.0, 1.0) as f32;
                self.envelope[ch] = self.rest[ch] + a * (self.max[ch] - self.rest[ch]);
            } else {
                continue;
            }
            self.estimated[ch] = true;
        }
    }

    // ---- calibration ----------------------------------------------------------

    pub fn start_calibration(&mut self, steps: Vec<CalStep>) {
        let steps: Vec<CalStep> =
            steps.into_iter().filter(|s| !matches!(s, CalStep::Max(ch) if *ch >= NUM_CHANNELS)).collect();
        self.cal = (!steps.is_empty()).then(|| CalRun {
            steps,
            step: 0,
            phase: Phase::Prepare,
            t: 0,
            good: 0,
            stuck: 0,
            rec: std::array::from_fn(|_| Vec::new()),
            contact_count: [0; NUM_CHANNELS],
        });
    }

    /// Give up on the current step (e.g. an electrode that won't make contact)
    /// and move on. Whatever that channel had calibrated before is kept.
    pub fn skip_calibration_step(&mut self) {
        let Some(mut run) = self.cal.take() else { return };
        run.rec.iter_mut().for_each(Vec::clear);
        run.contact_count = [0; NUM_CHANNELS];
        run.phase = Phase::Relax; // advance() continues with the next step
        if run.advance() {
            self.cal = Some(run);
        }
    }

    fn pulse_threshold(&self, ch: usize) -> f32 {
        (self.rest[ch] * PULSE_CLEAN_RATIO).max(self.rest[ch] + MIN_SPAN)
    }

    fn relaxed_level(&self, ch: usize) -> f32 {
        (self.rest[ch] * RELAXED_RATIO).max(self.rest[ch] + MIN_SPAN)
    }

    fn still_limit(&self, ch: usize) -> f32 {
        (self.recent[ch] * STILL_RATIO + STILL_MARGIN_UV).min(REST_CEILING_UV)
    }

    /// Is the signal right now good enough for this phase? Also returns the meter
    /// value (level / limit). None while preparing.
    fn signal_check(&self, step: CalStep, phase: Phase) -> Option<(Check, f32)> {
        let contact = self.contact();
        let used = self.used_channels();
        // worst ratio of envelope to a per-channel limit, over the used channels with contact
        let worst = |limit: &dyn Fn(usize) -> f32| {
            used.iter()
                .filter(|&&ch| contact[ch])
                .map(|&ch| (ch, self.envelope[ch] / limit(ch).max(1e-3)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
        };
        match (step, phase) {
            (_, Phase::Prepare) | (CalStep::Rest, Phase::Relax) => None,
            (CalStep::Rest, Phase::Record) => {
                // offline electrodes don't block: they get estimated later. It only
                // needs enough working ones to measure anything.
                if used.iter().filter(|&&ch| contact[ch]).count() < MIN_WORKING_CHANNELS {
                    let ch = used.iter().copied().find(|&ch| !contact[ch]).unwrap_or(0);
                    return Some((Check::NoContact(ch), 0.0));
                }
                let (ch, r) = worst(&|ch| self.still_limit(ch)).unwrap_or((0, 0.0));
                Some((if r > 1.0 { Check::Moving(ch) } else { Check::Clean }, r))
            }
            (CalStep::Max(ch), Phase::Record) => {
                let r = if contact[ch] {
                    self.envelope[ch] / self.pulse_threshold(ch)
                } else {
                    // electrode offline: read the pulse from the crosstalk it leaves
                    // on the working channels instead
                    let best = used
                        .iter()
                        .filter(|&&c| c != ch && contact[c])
                        .map(|&c| self.envelope[c] / self.pulse_threshold(c))
                        .fold(None, |m: Option<f32>, r| Some(m.map_or(r, |m| m.max(r))));
                    match best {
                        Some(r) => r,
                        None => return Some((Check::NoContact(ch), 0.0)),
                    }
                };
                Some((if r >= 1.0 { Check::Clean } else { Check::NoPulse }, r))
            }
            (CalStep::Max(_), Phase::Relax) => {
                let (ch, r) = worst(&|ch| self.relaxed_level(ch)).unwrap_or((0, 0.0));
                Some((if r > 1.0 { Check::NotRelaxed(ch) } else { Check::Clean }, r))
            }
        }
    }

    /// Rest, then a max contraction for every mapped channel.
    pub fn start_full_calibration(&mut self) {
        let steps = std::iter::once(CalStep::Rest).chain(self.used_channels().into_iter().map(CalStep::Max)).collect();
        self.start_calibration(steps);
    }

    pub fn cancel_calibration(&mut self) {
        self.cal = None;
    }

    pub fn calibration_plan(&self) -> Vec<CalStep> {
        self.cal.as_ref().map_or_else(Vec::new, |r| r.steps.clone())
    }

    fn calibration_step(&mut self) {
        let Some(mut run) = self.cal.take() else { return };
        let clean = matches!(self.signal_check(run.steps[run.step], run.phase), Some((Check::Clean, _)));
        run.t += 1;
        let done = match run.phase {
            Phase::Prepare => run.t >= run.duration(),
            Phase::Record => {
                // only clean samples are recorded, and only they count down
                if clean && run.t > samples(SETTLE_S) {
                    for (r, &e) in run.rec.iter_mut().zip(&self.envelope) {
                        r.push(e);
                    }
                    for (n, ok) in run.contact_count.iter_mut().zip(self.contact()) {
                        *n += u32::from(ok);
                    }
                    run.good += 1;
                    run.stuck = 0;
                } else {
                    run.stuck += 1;
                }
                run.good >= run.duration()
            }
            Phase::Relax => {
                if clean {
                    run.good += 1;
                    run.stuck = 0;
                } else {
                    run.good = 0;
                    run.stuck += 1;
                }
                run.t >= run.duration() && run.good >= samples(RELAXED_S)
            }
        };
        if !done {
            self.cal = Some(run);
            return;
        }
        if run.phase == Phase::Record {
            let n = run.rec[0].len();
            let samples: Vec<Features> =
                (0..n).map(|i| decoder::features(&std::array::from_fn(|ch| run.rec[ch][i]))).collect();
            let levels = run.rec.each_mut().map(|r| percentile(r, LEVEL_PERCENTILE));
            let observed = run.contact_count.map(|c| c as usize * 2 >= n.max(1));
            run.rec.iter_mut().for_each(Vec::clear);
            run.contact_count = [0; NUM_CHANNELS];
            self.apply_calibration(run.steps[run.step], levels, samples, observed);
        }
        if run.advance() {
            self.cal = Some(run);
        }
    }

    /// `observed`: which channels had contact for most of the recording.
    fn apply_calibration(
        &mut self,
        step: CalStep,
        levels: [f32; NUM_CHANNELS],
        samples: Vec<Features>,
        observed: [bool; NUM_CHANNELS],
    ) {
        // keep max meaningfully above rest so the activation span never collapses
        let floor = |rest: f32| rest + (rest).max(MIN_SPAN);
        let label = match step {
            CalStep::Rest => {
                self.rest = levels.map(|l| l.max(MIN_REST));
                for ch in 0..NUM_CHANNELS {
                    self.max[ch] = self.max[ch].max(floor(self.rest[ch]));
                }
                self.seen = observed;
                self.rest_done = true;
                Label::Rest
            }
            CalStep::Max(ch) => {
                self.via_others[ch] = false;
                if !observed[ch] {
                    // electrode offline: the pulse was read from the other channels.
                    // Its own level is unknown, but the movement still trains the decoder.
                    self.max[ch] = DEFAULT_MAX.max(floor(self.rest[ch]));
                    self.max_done[ch] = false;
                    self.via_others[ch] = true;
                } else if levels[ch] >= self.rest[ch] * MIN_PULSE_RATIO {
                    self.max[ch] = levels[ch].max(floor(self.rest[ch]));
                    self.max_done[ch] = true;
                } else {
                    // no pulse: don't let this channel become hair-trigger
                    self.max[ch] = DEFAULT_MAX.max(floor(self.rest[ch]));
                    self.max_done[ch] = false;
                }
                Label::Move(ch)
            }
        };

        // the recording replaces that class's training data (a rejected pulse just removes it)
        self.training.retain(|(l, _)| *l != label);
        let accepted = !matches!(step, CalStep::Max(ch) if !self.max_done[ch] && !self.via_others[ch]);
        if accepted {
            self.training.push((label, samples));
        }
        self.retrain();
    }

    fn retrain(&mut self) {
        // fixed order (rest, then channels) so reports and posteriors are stable
        self.training.sort_by_key(|(l, _)| match l {
            Label::Rest => -1,
            Label::Move(ch) => *ch as i32,
        });
        self.imputer = Imputer::train(&self.training);
        self.synergy = Synergy::train(&self.training);
        let has_movement = self.training.iter().any(|(l, _)| *l != Label::Rest);
        self.report = if self.synergy.is_some() && has_movement {
            decoder::cross_validate_synergy(&self.training)
        } else {
            Vec::new()
        };
        self.clear_decoder_state();
    }

    pub fn decoder_trained(&self) -> bool {
        self.synergy.is_some() && self.training.iter().any(|(l, _)| *l != Label::Rest)
    }

    /// The movements the pattern decoder currently drives the arm with.
    pub fn intent(&self) -> Intent {
        let Some(syn) = &self.synergy else { return Intent { moves: Vec::new(), relaxed: false } };
        let release = (self.threshold.clamp(0.0, 0.95) * RELEASE_FRACTION) as f64;
        Intent {
            moves: self.active.iter().map(|&k| (syn.channels()[k], self.weights[k] as f32)).collect(),
            relaxed: self.weights.iter().all(|&w| w < release),
        }
    }

    pub fn decoder_report(&self) -> &[ClassReport] {
        &self.report
    }

    pub fn calibration_status(&self) -> Option<CalStatus> {
        let run = self.cal.as_ref()?;
        let kind = run.steps[run.step];
        let duration = run.duration().max(1);
        let first = |i: usize| i == 0;
        let total: u32 = run.steps.iter().enumerate().map(|(i, s)| s.total(first(i))).sum();
        let done_steps: u32 = run.steps[..run.step].iter().enumerate().map(|(i, s)| s.total(first(i))).sum();
        let done_phases: u32 = [Phase::Prepare, Phase::Record]
            .iter()
            .take_while(|&&p| p != run.phase)
            .map(|&p| kind.duration(p, first(run.step)))
            .sum();
        let progress = run.progress().min(duration);
        let check = self.signal_check(kind, run.phase);
        Some(CalStatus {
            step: run.step,
            steps: run.steps.len(),
            kind,
            phase: run.phase,
            phase_progress: progress as f32 / duration as f32,
            seconds_left: (duration - progress) as f32 / SAMPLE_RATE,
            total_progress: ((done_steps + done_phases + progress) as f32 / total.max(1) as f32).min(1.0),
            check: check.map(|(c, _)| c),
            meter: check.map_or(0.0, |(_, m)| m),
            stuck_seconds: run.stuck as f32 / SAMPLE_RATE,
            proxy: matches!(kind, CalStep::Max(ch) if !self.contact()[ch]),
        })
    }

    pub fn is_calibrated(&self) -> bool {
        self.rest_done && self.used_channels().into_iter().all(|ch| self.max_done[ch] || self.via_others[ch])
    }

    /// Per channel: its movement was calibrated from the other channels because
    /// its own electrode was offline.
    pub fn via_others(&self) -> [bool; NUM_CHANNELS] {
        self.via_others
    }

    pub fn max_done(&self) -> [bool; NUM_CHANNELS] {
        self.max_done
    }

    pub fn envelopes(&self) -> [f32; NUM_CHANNELS] {
        self.envelope
    }

    pub fn rest_levels(&self) -> [f32; NUM_CHANNELS] {
        self.rest
    }

    pub fn max_levels(&self) -> [f32; NUM_CHANNELS] {
        self.max
    }

    pub fn activations(&self) -> [f32; NUM_CHANNELS] {
        std::array::from_fn(|ch| {
            let span = (self.max[ch] - self.rest[ch]).max(1e-3);
            ((self.envelope[ch] - self.rest[ch]) / span).clamp(0.0, 1.0)
        })
    }

    /// Joint velocity commands (-1..1), updated every sample.
    pub fn joint_inputs(&self) -> [f32; NUM_JOINTS] {
        self.inputs
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
        /// A pulse also shows up on two other electrodes (60% and 40%).
        crosstalk: bool,
        /// A second muscle pulsing at the same time as urst.
        also: Option<usize>,
    }

    impl Sim {
        fn new(seed: u32) -> Self {
            Self { rng: seed, n: 0, common: 0.0, railed: None, crosstalk: false, also: None }
        }

        fn drive(&self, ch: usize, burst: Option<usize>) -> f32 {
            match burst {
                Some(b) if b == ch => 1.0,
                Some(b) if self.crosstalk && ch == (b + 2) % NUM_CHANNELS => 0.6,
                Some(b) if self.crosstalk && ch == (b + 5) % NUM_CHANNELS => 0.4,
                _ => 0.0,
            }
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
                let d = self.drive(ch, burst) + if burst.is_some() { self.drive(ch, self.also) } else { 0.0 };
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
    fn hum_and_dc_are_rejected_burst_drives_joint() {
        let mut p = EmgPipeline::new();
        let mut sim = Sim::new(12345);
        sim.run(&mut p, 2.0, None);
        assert!(p.envelopes().iter().all(|&e| e < 5.0), "rest envelope {:?}", p.envelopes());
        assert_eq!(p.joint_inputs(), [0.0; NUM_JOINTS]);

        sim.run(&mut p, 1.0, Some(4)); // ch5 = elbow +
        assert!(p.envelopes()[4] > 40.0, "burst envelope {:?}", p.envelopes());
        let inputs = p.joint_inputs();
        assert!(inputs[2] > 0.3, "elbow input {inputs:?}");
        assert_eq!(inputs[0], 0.0);
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
        assert_eq!(p.joint_inputs()[0], 0.0, "a dead electrode must not drive its joint");
    }

    /// Feed samples until the calibration reaches `phase` of `step` (or gives up).
    fn run_until(p: &mut EmgPipeline, sim: &mut Sim, step: usize, phase: Phase) {
        for _ in 0..samples(30.0) {
            match p.calibration_status() {
                Some(st) if st.step == step && st.phase == phase => return,
                None => panic!("calibration ended before step {step} {phase:?}"),
                _ => {}
            }
            let s = sim.sample(None);
            p.process(&s);
        }
        panic!("never reached step {step} {phase:?}");
    }

    #[test]
    fn missing_pulse_holds_the_phase_until_skipped() {
        let mut p = EmgPipeline::new();
        let mut sim = Sim::new(21);
        sim.run(&mut p, 1.0, None);
        p.start_calibration(vec![CalStep::Rest, CalStep::Max(1)]);
        run_until(&mut p, &mut sim, 1, Phase::Record);
        sim.run(&mut p, 6.0, None); // never pulses: the phase must wait, not time out
        let st = p.calibration_status().unwrap();
        assert_eq!((st.step, st.phase, st.check), (1, Phase::Record, Some(Check::NoPulse)));
        assert_eq!(st.phase_progress, 0.0);
        assert!(st.stuck_seconds > 5.0);

        p.skip_calibration_step();
        assert!(p.calibration_status().is_none());
        assert!(!p.max_done()[1], "a skipped pulse is not calibrated");
        assert!(p.max_levels()[1] >= DEFAULT_MAX, "and stays insensitive");
        assert!(p.rest_levels().iter().all(|&r| r >= MIN_REST));
        sim.run(&mut p, 1.0, None);
        assert_eq!(p.joint_inputs(), [0.0; NUM_JOINTS]);
    }

    #[test]
    fn pulse_countdown_only_runs_while_the_pulse_is_clean() {
        let mut p = EmgPipeline::new();
        let mut sim = Sim::new(22);
        sim.run(&mut p, 1.0, None);
        p.start_calibration(vec![CalStep::Rest, CalStep::Max(4)]);
        run_until(&mut p, &mut sim, 1, Phase::Record);
        // pulse for 1 s, drop it for 2 s, pulse again: progress pauses in between
        sim.run(&mut p, 1.0, Some(4));
        let mid = p.calibration_status().unwrap();
        assert_eq!(mid.check, Some(Check::Clean));
        sim.run(&mut p, 2.0, None);
        let paused = p.calibration_status().unwrap();
        assert_eq!(paused.check, Some(Check::NoPulse));
        // only the envelope's decay tail (~0.2 s of 2.5 s) may still count after the pulse stops
        assert!(paused.phase_progress - mid.phase_progress < 0.1, "{mid:?} → {paused:?}");
        sim.run(&mut p, 2.0, Some(4));
        let st = p.calibration_status().unwrap();
        assert_eq!(st.phase, Phase::Relax, "enough clean pulse collected");
        assert!(p.max_done()[4]);
    }

    #[test]
    fn rest_ignores_movement_and_relax_waits_for_it_to_end() {
        let mut p = EmgPipeline::new();
        let mut sim = Sim::new(23);
        sim.run(&mut p, 2.0, None);
        p.start_calibration(vec![CalStep::Rest, CalStep::Max(4)]);
        run_until(&mut p, &mut sim, 0, Phase::Record);
        sim.run(&mut p, 1.5, None);
        // the user fidgets mid-rest: flagged, and not recorded
        for _ in 0..samples(0.3) {
            let s = sim.sample(Some(2));
            p.process(&s);
        }
        assert_eq!(p.calibration_status().unwrap().check, Some(Check::Moving(2)));
        run_until(&mut p, &mut sim, 1, Phase::Prepare);
        assert!(p.rest_levels()[2] < 5.0, "movement leaked into the rest floor: {:?}", p.rest_levels());

        run_until(&mut p, &mut sim, 1, Phase::Record);
        sim.run(&mut p, 3.0, Some(4));
        assert_eq!(p.calibration_status().unwrap().phase, Phase::Relax);
        // still tensed after the cue says relax: the run waits
        sim.run(&mut p, 3.0, Some(4));
        let st = p.calibration_status().expect("must not finish while the muscle is still active");
        assert_eq!(st.check, Some(Check::NotRelaxed(4)));
        sim.run(&mut p, 2.5, None);
        assert!(p.calibration_status().is_none());
    }

    #[test]
    fn hysteresis_holds_a_pulse_that_dips_below_threshold() {
        let mut p = EmgPipeline::new();
        p.threshold = 0.2;
        // activation 0.15: below the engage threshold, above the release level
        let at = |p: &mut EmgPipeline, a: f32| {
            p.envelope = [0.0; NUM_CHANNELS];
            p.envelope[4] = p.rest[4] + a * (p.max[4] - p.rest[4]);
            p.update_inputs();
            p.joint_inputs()[2]
        };
        assert_eq!(at(&mut p, 0.15), 0.0, "must not engage below threshold");
        assert!(at(&mut p, 0.5) > 0.0);
        assert!(at(&mut p, 0.15) > 0.0, "stays engaged above the release level");
        assert_eq!(at(&mut p, 0.05), 0.0, "releases below it");
    }

    #[test]
    fn guided_calibration_follows_the_sample_clock() {
        let mut p = EmgPipeline::new();
        p.set_mapping(0, Some(0), Some(1));
        for j in 1..NUM_JOINTS {
            p.set_mapping(j, None, None);
        }
        let mut sim = Sim::new(7);
        sim.run(&mut p, 1.0, None);
        p.start_full_calibration();
        assert_eq!(p.calibration_plan(), vec![CalStep::Rest, CalStep::Max(0), CalStep::Max(1)]);

        let mut last_progress = -1.0;
        let mut phases = Vec::new();
        while let Some(st) = p.calibration_status() {
            assert!(st.total_progress >= last_progress && st.total_progress <= 1.0);
            last_progress = st.total_progress;
            if phases.last() != Some(&(st.step, st.phase)) {
                phases.push((st.step, st.phase));
            }
            // the "user" contracts exactly when the cue says HOLD
            let burst = match (st.kind, st.phase) {
                (CalStep::Max(ch), Phase::Record) => Some(ch),
                _ => None,
            };
            let s = sim.sample(burst);
            p.process(&s);
        }
        use Phase::*;
        assert_eq!(phases, vec![(0, Prepare), (0, Record), (1, Prepare), (1, Record), (1, Relax), (2, Prepare), (2, Record), (2, Relax)]);
        assert!(p.is_calibrated());
        let (rest, max) = (p.rest_levels(), p.max_levels());
        for ch in 0..2 {
            assert!(max[ch] > rest[ch] * 10.0, "ch{ch}: rest {} max {}", rest[ch], max[ch]);
        }
    }

    /// Calibrate every channel with the simulated user pulsing on cue.
    fn calibrate(p: &mut EmgPipeline, sim: &mut Sim) {
        p.start_full_calibration();
        while let Some(st) = p.calibration_status() {
            let burst = match (st.kind, st.phase) {
                (CalStep::Max(ch), Phase::Record) => Some(ch),
                _ => None,
            };
            let s = sim.sample(burst);
            p.process(&s);
        }
    }

    /// Joints that end up moving while the biceps (CH5, elbow +) is pulsed.
    fn moved_joints(p: &mut EmgPipeline, sim: &mut Sim) -> Vec<(usize, f32)> {
        sim.run(p, 1.0, None);
        let mut peak = [0.0f32; NUM_JOINTS];
        for _ in 0..samples(1.5) {
            let s = sim.sample(Some(4));
            p.process(&s);
            for (pk, v) in peak.iter_mut().zip(p.joint_inputs()) {
                if v.abs() > pk.abs() {
                    *pk = v;
                }
            }
        }
        peak.iter().copied().enumerate().filter(|(_, v)| *v != 0.0).collect()
    }

    #[test]
    fn pattern_decoder_ignores_crosstalk_that_moves_other_joints_in_direct_mode() {
        let mut p = EmgPipeline::new();
        let mut sim = Sim::new(31);
        sim.crosstalk = true;
        sim.common = 20.0;
        sim.run(&mut p, 1.0, None);
        calibrate(&mut p, &mut sim);
        assert!(p.decoder_trained());
        assert!(p.decoder_report().iter().all(|r| r.accuracy > 0.9), "{:?}",
            p.decoder_report().iter().map(|r| (r.label, r.accuracy)).collect::<Vec<_>>());

        p.pattern = false;
        let direct = moved_joints(&mut p, &mut sim);
        assert!(direct.len() > 1, "crosstalk should leak into other joints in direct mode: {direct:?}");

        p.pattern = true;
        let decoded = moved_joints(&mut p, &mut sim);
        assert_eq!(decoded.len(), 1, "only the intended joint should move: {decoded:?}");
        assert_eq!(decoded[0].0, 2);
        assert!(decoded[0].1 > 0.3, "elbow should move up: {decoded:?}");
        assert_eq!(p.intent().moves.iter().map(|m| m.0).collect::<Vec<_>>(), vec![4]);

        sim.run(&mut p, 1.0, None);
        assert_eq!(p.joint_inputs(), [0.0; NUM_JOINTS]);
        assert!(p.intent().relaxed && p.intent().moves.is_empty());
    }

    #[test]
    fn compound_movement_drives_two_joints_at_once() {
        let mut p = EmgPipeline::new();
        let mut sim = Sim::new(61);
        sim.crosstalk = true;
        sim.common = 20.0;
        sim.run(&mut p, 1.0, None);
        calibrate(&mut p, &mut sim);

        // biceps (elbow +) and wrist flexors (wrist +) together, each with its crosstalk
        sim.also = Some(6);
        let moved = moved_joints(&mut p, &mut sim);
        let joints: Vec<usize> = moved.iter().map(|m| m.0).collect();
        assert_eq!(joints, vec![2, 3], "elbow and wrist only: {moved:?}");
        assert!(moved.iter().all(|m| m.1 > 0.3), "both forward: {moved:?}");
        let mut seen: Vec<usize> = p.intent().moves.iter().map(|m| m.0).collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![4, 6]);

        // one of them alone is decoded as just that one
        sim.also = None;
        let moved = moved_joints(&mut p, &mut sim);
        assert_eq!(moved.iter().map(|m| m.0).collect::<Vec<_>>(), vec![2], "{moved:?}");
    }

    #[test]
    fn calibration_works_around_an_electrode_that_is_offline_throughout() {
        let mut p = EmgPipeline::new();
        let mut sim = Sim::new(51);
        sim.crosstalk = true;
        sim.common = 20.0;
        sim.railed = Some(1); // the brachioradialis electrode (CH2, base yaw −) never makes contact
        sim.run(&mut p, 3.0, None);

        // the whole run completes: rest doesn't block, and CH2's pulse is read from the others
        p.start_full_calibration();
        let mut proxied = false;
        for _ in 0..samples(120.0) {
            let Some(st) = p.calibration_status() else { break };
            assert_ne!(st.check, Some(Check::NoContact(1)), "must not block on the offline electrode");
            proxied |= st.proxy && st.kind == CalStep::Max(1) && st.check == Some(Check::Clean);
            let burst = match (st.kind, st.phase) {
                (CalStep::Max(ch), Phase::Record) => Some(ch),
                _ => None,
            };
            let s = sim.sample(burst);
            p.process(&s);
        }
        assert!(p.calibration_status().is_none(), "calibration did not finish");
        assert!(proxied, "CH2's pulse should have been detected via the other channels");
        assert!(p.via_others()[1] && !p.max_done()[1]);
        assert!(p.is_calibrated());
        let r = p.decoder_report().iter().find(|r| r.label == Label::Move(1)).expect("CH2's movement trained");
        assert!(r.accuracy > 0.8, "CH2's movement recognised from the other channels: {}", r.accuracy);

        // at rest: CH2's estimate stays low and nothing moves
        sim.run(&mut p, 1.0, None);
        assert!(p.estimated()[1]);
        assert_eq!(p.joint_inputs(), [0.0; NUM_JOINTS]);
        // pulsing CH2's muscle: decoded from the others, base yaw turns negative, and
        // the CH2 estimate rises to a strong activation
        for _ in 0..samples(1.0) {
            let s = sim.sample(Some(1));
            p.process(&s);
        }
        assert_eq!(p.intent().moves.iter().map(|m| m.0).collect::<Vec<_>>(), vec![1]);
        assert!(p.joint_inputs()[0] < -0.3, "{:?}", p.joint_inputs());
        assert!(p.activations()[1] > 0.5, "estimated CH2 activation {:?}", p.activations());
    }

    #[test]
    fn offline_electrode_is_estimated_from_the_others() {
        let mut p = EmgPipeline::new();
        let mut sim = Sim::new(41);
        sim.crosstalk = true;
        sim.common = 20.0;
        sim.run(&mut p, 1.0, None);
        calibrate(&mut p, &mut sim);

        // the biceps electrode (CH5) comes off after calibration
        sim.railed = Some(4);
        sim.run(&mut p, 2.5, None);
        assert!(!p.contact()[4] && p.estimated()[4]);
        assert_eq!(p.joint_inputs(), [0.0; NUM_JOINTS], "relaxed must stay still with the estimate");
        assert!(p.envelopes()[4] < p.pulse_threshold(4), "resting estimate {:?}", p.envelopes());

        // pulsing the biceps now only shows up as crosstalk on other electrodes
        let moved = moved_joints(&mut p, &mut sim);
        assert_eq!(moved.len(), 1, "{moved:?}");
        assert_eq!(moved[0].0, 2, "the elbow should still move: {moved:?}");
        assert!(moved[0].1 > 0.3);
        assert!(p.envelopes()[4] > p.pulse_threshold(4), "estimated envelope {:?}", p.envelopes());

        // no estimates while calibrating: it must record only real signal
        p.start_calibration(vec![CalStep::Rest]);
        let s = sim.sample(None);
        p.process(&s);
        assert!(!p.estimated()[4]);
        assert_eq!(p.envelopes()[4], 0.0);
    }

    #[test]
    fn single_step_and_cancel() {
        let mut p = EmgPipeline::new();
        let mut sim = Sim::new(3);
        p.start_calibration(vec![CalStep::Max(9)]); // out of range → nothing to do
        assert!(p.calibration_status().is_none());
        p.start_calibration(vec![CalStep::Rest]);
        sim.run(&mut p, 1.0, None);
        assert_eq!(p.calibration_status().unwrap().phase, Phase::Prepare);
        p.cancel_calibration();
        assert!(p.calibration_status().is_none());
        assert!(!p.is_calibrated());
    }
}
