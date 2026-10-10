//! Finger control from a 2×4 electrode grid over the forearm flexors (the
//! underside of the forearm, which is rarely hairy).
//!
//! Four finger movements each drive one joint, and how firmly the finger is
//! *tensed*, never actually moved, picks the direction. Power is 0 at rest
//! and 1 at the ceiling, the softest level at which the finger really moves:
//!
//! | power                    | joint                                    |
//! |--------------------------|------------------------------------------|
//! | within the stasis band   | holds                                    |
//! | faint (around `level`)   | − (after a short commit delay)           |
//! | firmer (above the split) | +                                        |
//! | above the ceiling        | everything holds: the hand really moved  |
//!
//! Phase 1 (calibration) learns each finger's pattern across the grid and its
//! ceiling. It works down from a hard flex, to about half, to fainter and
//! fainter tension, and re-learns the pattern from every round so the decoder
//! picks up faint signals.
//! Phase 2 (training) has the user hit the faint and firmer bands, one finger
//! at a time with only that finger's joint moving, and lowers the levels as
//! they get consistent.

use super::cyton::{NUM_CHANNELS, SAMPLE_RATE};
use super::decoder::{nnls, Features};
use crate::arm::NUM_JOINTS;

pub const NUM_FINGERS: usize = 4;

/// Smoothing of the finger powers.
const POWER_TAU_S: f32 = 0.06;
/// The stasis band reaches at least this far: a tenth of the way to the ceiling.
const STASIS_MIN: f64 = 0.1;
/// …and clears this percentile of resting power by a margin.
const STASIS_PERCENTILE: f64 = 0.99;
const STASIS_MARGIN: f64 = 0.03;
/// The ceiling sits at this percentile of the medium flex: tension past the
/// softest part of a real flex counts as real movement.
const CEILING_PERCENTILE: f64 = 0.2;
/// Targets of the faint rounds, as fractions of the ceiling.
pub const FAINT_TARGETS: [f64; 4] = [0.6, 0.4, 0.25, 0.15];
/// A faint target needs at least this multiple of the stasis band.
const TARGET_ROOM: f64 = 2.0;
/// A faint round must come in below this fraction of the last one to go on.
const PROGRESS_RATIO: f64 = 0.9;
/// Faint rounds record anything clearly above stasis (this multiple of it).
const FAINT_MIN: f64 = 1.3;
/// Command level (the faint, − target): at most this, at least this multiple of stasis.
const LEVEL_MAX: f64 = 0.6;
const LEVEL_MIN_ROOM: f64 = 2.2;
/// Leaving rest needs this fraction of the way from stasis to the level: the
/// large buffer that keeps resting noise from twitching the joint.
const REST_BUFFER: f64 = 0.4;
/// Hysteresis around the −/+ split, as a fraction of split − stasis.
const SPLIT_HYSTERESIS: f64 = 0.1;
/// A faint level only drives − after holding this long, so rising to + doesn't
/// start with a − twitch.
const COMMIT_S: f32 = 0.25;
/// After crossing the ceiling everything holds until power stays below
/// OVER_EXIT for this long.
const OVER_EXIT: f64 = 0.9;
const OVER_HOLD_S: f32 = 0.4;
/// Default joint speed (fraction of its max) of a committed band.
const DEFAULT_SPEED: f32 = 0.5;

// Signal checks.
/// Hard flex: RMS of the positive z-scores across the grid.
const HARD_MIN_ACTIVITY: f64 = 5.0;
const MEDIUM_MIN_ACTIVITY: f64 = 2.0;
/// Medium flex, as a fraction of the hard one.
const MEDIUM_MIN: f64 = 0.15;
const MEDIUM_MAX: f64 = 0.85;
pub const MEDIUM_TARGET: (f64, f64) = (0.35, 0.65);
/// Faint rounds aim for target × (1 ± this).
const TARGET_WIDTH: f64 = 0.3;
/// Relaxed: no channel above this many resting standard deviations.
const RELAX_Z: f64 = 4.0;
/// Rest: a channel jumping above this multiple of its recent level is movement.
const STILL_RATIO: f32 = 2.5;
const STILL_MARGIN_UV: f32 = 1.5;
const REST_CEILING_UV: f32 = 30.0;
const RECENT_TAU_S: f32 = 1.5;
/// Fewer working electrodes than this and there is nothing to learn from.
const MIN_WORKING: usize = 4;
/// Resting-noise floor of a channel (µV of envelope), so a dead-quiet channel
/// can't turn the tiniest flicker into a huge z-score.
const MIN_SCALE_UV: f64 = 0.2;

// Timing, in seconds. Record phases count only clean samples.
const PREPARE_FIRST_S: f32 = 3.0;
const PREPARE_S: f32 = 2.0;
const REST_S: f32 = 5.0;
const HARD_S: f32 = 2.0;
const MEDIUM_S: f32 = 2.5;
const FAINT_S: f32 = 3.0;
const RELAX_S: f32 = 1.5;
const RELAXED_S: f32 = 0.3;
const SETTLE_S: f32 = 0.5;

// Training.
pub const TRIALS: usize = 8;
const CUE_S: f32 = 1.5;
const TRIAL_TIMEOUT_S: f32 = 10.0;
const TRIAL_HOLD_S: f32 = 1.0;
const TRAIN_RELAX_S: f32 = 1.0;
/// Hits in a row before the levels drop, and by how much; a miss raises them.
const STREAK: u32 = 2;
const LEVEL_DOWN: f64 = 0.85;
const LEVEL_UP: f64 = 1.12;

fn samples(seconds: f32) -> u32 {
    (seconds * SAMPLE_RATE).round() as u32
}

fn dot(a: &Features, b: &Features, usable: &[bool; NUM_CHANNELS]) -> f64 {
    (0..NUM_CHANNELS).filter(|&c| usable[c]).map(|c| a[c] * b[c]).sum()
}

fn mean(xs: &[Features]) -> Features {
    let mut m = [0.0; NUM_CHANNELS];
    for x in xs {
        for (mi, xi) in m.iter_mut().zip(x) {
            *mi += xi;
        }
    }
    m.map(|v| v / xs.len().max(1) as f64)
}

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let i = ((v.len() - 1) as f64 * p).round() as usize;
    *v.select_nth_unstable_by(i, f64::total_cmp).1
}

/// Unit vector along `x` over the usable channels; None if it is ~zero.
fn direction(x: &Features, usable: &[bool; NUM_CHANNELS]) -> Option<Features> {
    let n = dot(x, x, usable).sqrt();
    (n > 1e-9).then(|| std::array::from_fn(|c| if usable[c] { x[c] / n } else { 0.0 }))
}

/// One step of a calibration or training run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    /// Whole hand relaxed: resting level and noise of every channel.
    Rest,
    /// Flex this finger hard (it moves): the top of its range.
    Hard(usize),
    /// About half as hard (it still moves): sets the ceiling.
    Medium(usize),
    /// Faint round k: tension only, aiming at FAINT_TARGETS[k].
    Faint(usize, usize),
    /// Training: hit this finger's faint and firmer bands.
    Train(usize),
}

impl Step {
    pub fn finger(self) -> Option<usize> {
        match self {
            Step::Rest => None,
            Step::Hard(f) | Step::Medium(f) | Step::Faint(f, _) | Step::Train(f) => Some(f),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    /// Read the cue (training: the cue for the next trial).
    Prepare,
    /// Recording (training: the trial itself).
    Record,
    Relax,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Check {
    Clean,
    /// Too few electrodes with contact; this one is off.
    NoContact(usize),
    /// Rest: this channel jumped.
    Moving(usize),
    /// Not enough tension for this step.
    TooWeak,
    /// Past the step's range (faint rounds and training: the ceiling, a real movement).
    TooStrong,
    /// Another finger's pattern is stronger than the cued one.
    OtherFinger(usize),
    /// Training: this finger hasn't let go; otherwise this channel.
    NotRelaxed(usize),
}

/// Where a finger's power currently sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Band {
    Rest,
    /// In the faint band, not committed yet.
    Pending,
    Minus,
    Plus,
    /// Past the ceiling: a real movement, rejected.
    Over,
}

/// Band edges for one finger, in power units (0 = rest, 1 = ceiling).
#[derive(Clone, Copy, Debug)]
pub struct Bands {
    /// Below this the joint holds.
    pub stasis: f64,
    /// Leaving rest needs this much (the buffer above stasis).
    pub rest_exit: f64,
    /// The faint (−) target.
    pub level: f64,
    /// Between faint (−) and firmer (+).
    pub split: f64,
    pub plus_enter: f64,
    pub plus_exit: f64,
}

/// What the coach's bar shows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Meter {
    /// Hard flex: activity relative to the minimum needed (1 = enough).
    Activity,
    /// Medium flex: fraction of the hard one.
    OfHard,
    /// Finger power: 0 = rest, 1 = ceiling.
    Power,
    /// Rest / relax: the busiest channel against its limit (1 = at the limit).
    Stillness,
}

#[derive(Clone, Copy, Debug)]
pub struct Status {
    pub step: Step,
    pub phase: Phase,
    pub phase_progress: f32,
    pub seconds_left: f32,
    pub check: Option<Check>,
    pub stuck_seconds: f32,
    pub meter_kind: Meter,
    pub meter: f64,
    /// Range to aim for on the meter (NaN when there is none).
    pub target: (f64, f64),
    /// Current segment (rest, each finger's calibration, each finger's
    /// training) and how many there are.
    pub segment: usize,
    pub segments: usize,
    /// Faint round or training trial (0-based), and how many at most (0 = n/a).
    pub round: usize,
    pub rounds: usize,
    /// Training: the band the trial asks for.
    pub goal: Option<Band>,
}

/// What phases 1 and 2 found for one finger.
#[derive(Clone, Copy, Debug, Default)]
pub struct FingerReport {
    pub calibrated: bool,
    /// Faintest level reached in phase 1, as a fraction of the ceiling.
    pub faint: f64,
    /// Faint rounds completed.
    pub rounds: usize,
    /// Share of the faintest round in which this finger came out on top and above stasis.
    pub isolation: f64,
    pub hits: u32,
    pub trials: u32,
    /// Current faint (−) level.
    pub level: f64,
    pub stasis: f64,
}

/// Per-finger calibration recordings.
#[derive(Default)]
struct FingerData {
    hard: Option<Features>,
    /// z samples of the medium flex (they define the ceiling).
    medium: Vec<Features>,
    /// Unit patterns from the medium flex and each faint round.
    shapes: Vec<Features>,
    /// Level reached in each faint round.
    faint_levels: Vec<f64>,
    /// z samples of the faintest round (for the level and isolation).
    faintest: Vec<Features>,
}

struct RestModel {
    mean: Features,
    scale: Features,
    /// Channel had contact during the rest recording.
    seen: [bool; NUM_CHANNELS],
    /// z samples of the rest recording.
    z: Vec<Features>,
}

struct Run {
    step: Step,
    phase: Phase,
    /// Steps entered before this one (the first gets a longer prepare).
    steps_done: usize,
    t: u32,
    good: u32,
    stuck: u32,
    /// Clean samples of the current Record phase: envelopes for Rest, z otherwise.
    rec: Vec<Features>,
    /// Decided when a recording finishes, entered after its relax phase.
    next: Option<Step>,
    has_rest: bool,
    /// Fingers to calibrate in phase 1, and to train in phase 2.
    calibrate: Vec<usize>,
    train: Vec<usize>,
    train_after: bool,
    /// Training: trial number, samples on target, hits in a row.
    trial: usize,
    hold: u32,
    streak: u32,
}

pub struct FingerControl {
    rest: Option<RestModel>,
    data: [FingerData; NUM_FINGERS],
    templates: [Option<Features>; NUM_FINGERS],
    stasis: [f64; NUM_FINGERS],
    level: [f64; NUM_FINGERS],
    report: [FingerReport; NUM_FINGERS],
    /// Joint each finger drives.
    joint_of: [usize; NUM_FINGERS],
    pub speed: f32,
    envelope: [f32; NUM_CHANNELS],
    contact: [bool; NUM_CHANNELS],
    recent: [f32; NUM_CHANNELS],
    z: Features,
    power: [f64; NUM_FINGERS],
    band: [Band; NUM_FINGERS],
    band_t: [f32; NUM_FINGERS],
    run: Option<Run>,
    inputs: [f32; NUM_JOINTS],
}

impl Default for FingerControl {
    fn default() -> Self {
        Self::new()
    }
}

impl FingerControl {
    pub fn new() -> Self {
        Self {
            rest: None,
            data: Default::default(),
            templates: [None; NUM_FINGERS],
            stasis: [STASIS_MIN; NUM_FINGERS],
            level: [LEVEL_MAX; NUM_FINGERS],
            report: [FingerReport::default(); NUM_FINGERS],
            joint_of: std::array::from_fn(|f| f % NUM_JOINTS),
            speed: DEFAULT_SPEED,
            envelope: [0.0; NUM_CHANNELS],
            contact: [true; NUM_CHANNELS],
            recent: [0.0; NUM_CHANNELS],
            z: [0.0; NUM_CHANNELS],
            power: [0.0; NUM_FINGERS],
            band: [Band::Rest; NUM_FINGERS],
            band_t: [0.0; NUM_FINGERS],
            run: None,
            inputs: [0.0; NUM_JOINTS],
        }
    }

    /// Clears the live state after a reconnect (keeps everything learned).
    pub fn reset_state(&mut self) {
        self.recent = [0.0; NUM_CHANNELS];
        self.power = [0.0; NUM_FINGERS];
        self.band = [Band::Rest; NUM_FINGERS];
        self.band_t = [0.0; NUM_FINGERS];
        self.inputs = [0.0; NUM_JOINTS];
    }

    pub fn set_joint(&mut self, finger: usize, joint: usize) {
        if finger < NUM_FINGERS && joint < NUM_JOINTS {
            self.joint_of[finger] = joint;
        }
    }

    pub fn joints(&self) -> [usize; NUM_FINGERS] {
        self.joint_of
    }

    pub fn trained(&self) -> bool {
        self.templates.iter().any(Option::is_some)
    }

    pub fn powers(&self) -> [f64; NUM_FINGERS] {
        self.power
    }

    pub fn band_states(&self) -> [Band; NUM_FINGERS] {
        self.band
    }

    pub fn report(&self) -> [FingerReport; NUM_FINGERS] {
        std::array::from_fn(|f| FingerReport {
            calibrated: self.templates[f].is_some(),
            level: self.bands(f).level,
            stasis: self.stasis[f],
            ..self.report[f]
        })
    }

    pub fn joint_inputs(&self) -> [f32; NUM_JOINTS] {
        self.inputs
    }

    pub fn bands(&self, f: usize) -> Bands {
        let stasis = self.stasis[f];
        let lo = stasis * LEVEL_MIN_ROOM;
        let level = self.level[f].clamp(lo, LEVEL_MAX.max(lo));
        let split = (2.0 * level).min(level + 0.35 * (1.0 - level));
        let h = SPLIT_HYSTERESIS * (split - stasis) / 2.0;
        Bands {
            stasis,
            rest_exit: stasis + REST_BUFFER * (level - stasis),
            level,
            split,
            plus_enter: split + h,
            plus_exit: split - h,
        }
    }

    /// Channels that take part in the fit: seen at rest and in contact now.
    fn usable(&self) -> [bool; NUM_CHANNELS] {
        let seen = self.rest.as_ref().map_or([false; NUM_CHANNELS], |r| r.seen);
        std::array::from_fn(|c| seen[c] && self.contact[c])
    }

    fn seen(&self) -> [bool; NUM_CHANNELS] {
        self.rest.as_ref().map_or([false; NUM_CHANNELS], |r| r.seen)
    }

    /// Power of each calibrated finger: the sample explained as a non-negative
    /// mix of the finger templates, fitted over the usable channels.
    fn fit(templates: &[Option<Features>; NUM_FINGERS], z: &Features, usable: &[bool; NUM_CHANNELS]) -> [f64; NUM_FINGERS] {
        let idx: Vec<usize> = (0..NUM_FINGERS).filter(|&f| templates[f].is_some()).collect();
        let mut out = [0.0; NUM_FINGERS];
        if idx.is_empty() {
            return out;
        }
        let t = |f: usize| templates[f].as_ref().unwrap();
        let mut g: Vec<Vec<f64>> = idx.iter().map(|&a| idx.iter().map(|&b| dot(t(a), t(b), usable)).collect()).collect();
        // a small ridge keeps similar fingers from trading wild, opposite weights
        let ridge = 0.02 * (0..idx.len()).map(|i| g[i][i]).sum::<f64>() / idx.len() as f64;
        for (i, row) in g.iter_mut().enumerate() {
            row[i] += ridge;
        }
        let h: Vec<f64> = idx.iter().map(|&a| dot(t(a), z, usable)).collect();
        for (k, w) in nnls(&g, &h).into_iter().enumerate() {
            out[idx[k]] = w;
        }
        out
    }

    /// RMS of the positive z-scores over the usable channels.
    fn activity(z: &Features, usable: &[bool; NUM_CHANNELS]) -> f64 {
        let n = usable.iter().filter(|&&u| u).count().max(1);
        ((0..NUM_CHANNELS).filter(|&c| usable[c]).map(|c| z[c].max(0.0).powi(2)).sum::<f64>() / n as f64).sqrt()
    }

    /// Feed one sample's envelopes (µV) and contact flags.
    pub fn process(&mut self, env: &[f32; NUM_CHANNELS], contact: &[bool; NUM_CHANNELS]) {
        self.envelope = *env;
        self.contact = *contact;
        let k = 1.0 - (-1.0 / (RECENT_TAU_S * SAMPLE_RATE)).exp();
        for (r, &e) in self.recent.iter_mut().zip(env) {
            *r = if *r == 0.0 { e } else { *r + (e - *r) * k };
        }
        if let Some(rest) = &self.rest {
            // channels without contact read as resting rather than as a huge negative
            self.z = std::array::from_fn(|c| if contact[c] { (env[c] as f64 - rest.mean[c]) / rest.scale[c] } else { 0.0 });
            let raw = Self::fit(&self.templates, &self.z, &self.usable());
            let alpha = 1.0 - (-1.0 / (POWER_TAU_S * SAMPLE_RATE) as f64).exp();
            for (p, r) in self.power.iter_mut().zip(raw) {
                *p += (r - *p) * alpha;
            }
            for f in 0..NUM_FINGERS {
                self.update_band(f);
            }
        }
        self.calibration_step();
        self.update_inputs();
    }

    fn update_band(&mut self, f: usize) {
        if self.templates[f].is_none() {
            self.band[f] = Band::Rest;
            return;
        }
        let p = self.power[f];
        let b = self.bands(f);
        let dt = 1.0 / SAMPLE_RATE;
        let t = self.band_t[f];
        let (next, t) = if p > 1.0 {
            (Band::Over, 0.0)
        } else {
            match self.band[f] {
                Band::Over => {
                    let t = if p > OVER_EXIT { 0.0 } else { t + dt };
                    if t >= OVER_HOLD_S { (Band::Rest, 0.0) } else { (Band::Over, t) }
                }
                Band::Rest if p > b.plus_enter => (Band::Plus, 0.0),
                Band::Rest if p > b.rest_exit => (Band::Pending, 0.0),
                Band::Rest => (Band::Rest, 0.0),
                _ if p < b.stasis => (Band::Rest, 0.0),
                Band::Pending | Band::Minus if p > b.plus_enter => (Band::Plus, 0.0),
                Band::Pending if t + dt >= COMMIT_S => (Band::Minus, 0.0),
                Band::Pending => (Band::Pending, t + dt),
                Band::Minus => (Band::Minus, 0.0),
                Band::Plus if p < b.plus_exit => (Band::Pending, 0.0),
                Band::Plus => (Band::Plus, 0.0),
            }
        };
        self.band[f] = next;
        self.band_t[f] = t;
    }

    fn update_inputs(&mut self) {
        self.inputs = [0.0; NUM_JOINTS];
        // a real hand movement: hold everything
        if self.band.contains(&Band::Over) {
            return;
        }
        let fingers: Vec<usize> = match self.run.as_ref().map(|r| r.step) {
            None => (0..NUM_FINGERS).collect(),
            // training: only the finger being trained moves the arm
            Some(Step::Train(f)) => vec![f],
            Some(_) => Vec::new(),
        };
        for f in fingers {
            let v = match self.band[f] {
                Band::Minus => -self.speed,
                Band::Plus => self.speed,
                _ => 0.0,
            };
            let j = self.joint_of[f];
            self.inputs[j] = (self.inputs[j] + v).clamp(-1.0, 1.0);
        }
    }

    // ---- runs ---------------------------------------------------------------------

    /// Phase 1 for all four fingers (rest first), then phase 2 if `train_after`.
    pub fn start_calibration(&mut self, train_after: bool) {
        self.start(Step::Rest, (0..NUM_FINGERS).collect(), Vec::new(), train_after);
    }

    /// Phase 1 for one finger, keeping the rest model (a full run if there is none).
    pub fn start_finger(&mut self, finger: usize, train_after: bool) {
        if finger >= NUM_FINGERS {
            return;
        }
        let first = if self.rest.is_some() { Step::Hard(finger) } else { Step::Rest };
        self.start(first, vec![finger], Vec::new(), train_after);
    }

    /// Phase 2 only: every calibrated finger.
    pub fn start_training(&mut self) {
        let fingers: Vec<usize> = (0..NUM_FINGERS).filter(|&f| self.templates[f].is_some()).collect();
        if let Some(&first) = fingers.first() {
            self.start(Step::Train(first), Vec::new(), fingers, false);
        }
    }

    fn start(&mut self, step: Step, calibrate: Vec<usize>, train: Vec<usize>, train_after: bool) {
        self.run = Some(Run {
            step,
            phase: Phase::Prepare,
            steps_done: 0,
            t: 0,
            good: 0,
            stuck: 0,
            rec: Vec::new(),
            next: None,
            has_rest: step == Step::Rest,
            calibrate,
            train,
            train_after,
            trial: 0,
            hold: 0,
            streak: 0,
        });
    }

    pub fn cancel(&mut self) {
        self.run = None;
    }

    pub fn running(&self) -> bool {
        self.run.is_some()
    }

    /// Skip the current step. Hard or medium flex: skips that finger. A faint
    /// round: stops at the faintest level so far. Training: on to the next
    /// finger. Rest can't be skipped, since everything is measured against it.
    pub fn skip(&mut self) {
        let Some(mut run) = self.run.take() else { return };
        let next = match (run.step, run.phase) {
            (Step::Rest, _) => {
                self.run = Some(run);
                return;
            }
            // already recorded: just skip the relax
            (Step::Hard(_) | Step::Medium(_) | Step::Faint(..), Phase::Relax) => run.next.take(),
            (Step::Hard(f) | Step::Medium(f), _) => {
                self.data[f] = FingerData::default();
                self.templates[f] = None;
                self.after_finger(&mut run, f)
            }
            (Step::Faint(f, _), _) => self.after_finger(&mut run, f),
            (Step::Train(f), _) => Self::after_training(&run, f),
        };
        if self.enter(&mut run, next) {
            self.run = Some(run);
        }
    }

    fn duration(run: &Run) -> u32 {
        match (run.step, run.phase) {
            (Step::Train(_), Phase::Prepare) => samples(CUE_S),
            (_, Phase::Prepare) => samples(if run.steps_done == 0 { PREPARE_FIRST_S } else { PREPARE_S }),
            (Step::Rest, Phase::Record) => samples(REST_S),
            (Step::Hard(_), Phase::Record) => samples(HARD_S),
            (Step::Medium(_), Phase::Record) => samples(MEDIUM_S),
            (Step::Faint(..), Phase::Record) => samples(FAINT_S),
            (Step::Train(_), Phase::Record) => samples(TRIAL_TIMEOUT_S),
            (Step::Train(_), Phase::Relax) => samples(TRAIN_RELAX_S),
            (_, Phase::Relax) => samples(RELAX_S),
        }
    }

    /// Band asked for by training trial `trial`: alternately faint and firmer.
    fn goal(trial: usize) -> Band {
        if trial % 2 == 0 { Band::Minus } else { Band::Plus }
    }

    /// Medium flex: projection onto the hard pattern (1 = as hard as the hard flex).
    fn of_hard(&self, f: usize) -> f64 {
        let usable = self.usable();
        self.data[f].hard.map_or(0.0, |h| (dot(&h, &self.z, &usable) / dot(&h, &h, &usable).max(1e-9)).max(0.0))
    }

    /// The channel furthest above its recent level, against its stillness limit.
    fn stillness_recent(&self) -> (usize, f64) {
        (0..NUM_CHANNELS)
            .filter(|&c| self.contact[c])
            .map(|c| {
                let limit = (self.recent[c] * STILL_RATIO + STILL_MARGIN_UV).min(REST_CEILING_UV);
                (c, (self.envelope[c] / limit.max(1e-3)) as f64)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, 0.0))
    }

    /// The channel furthest above rest, against RELAX_Z.
    fn stillness_z(&self) -> (usize, f64) {
        let usable = self.usable();
        (0..NUM_CHANNELS)
            .filter(|&c| usable[c])
            .map(|c| (c, self.z[c] / RELAX_Z))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, 0.0))
    }

    fn meter(&self, run: &Run) -> (Meter, f64, (f64, f64)) {
        let none = (f64::NAN, f64::NAN);
        match (run.step, run.phase) {
            (Step::Rest, _) => (Meter::Stillness, self.stillness_recent().1, none),
            (Step::Hard(_), Phase::Record) => {
                (Meter::Activity, Self::activity(&self.z, &self.usable()) / HARD_MIN_ACTIVITY, (1.0, f64::INFINITY))
            }
            (Step::Medium(f), Phase::Record) => (Meter::OfHard, self.of_hard(f), MEDIUM_TARGET),
            (Step::Faint(f, k), Phase::Record) => {
                let t = FAINT_TARGETS[k];
                (Meter::Power, self.power[f], (t * (1.0 - TARGET_WIDTH), (t * (1.0 + TARGET_WIDTH)).min(1.0)))
            }
            (Step::Train(f), Phase::Relax) => (Meter::Power, self.power[f], none),
            (Step::Train(f), _) => {
                let b = self.bands(f);
                let target = match Self::goal(run.trial) {
                    Band::Minus => (b.rest_exit, b.plus_exit),
                    _ => (b.plus_enter, 1.0),
                };
                (Meter::Power, self.power[f], target)
            }
            _ => (Meter::Stillness, self.stillness_z().1, none),
        }
    }

    fn signal_check(&self, run: &Run) -> Option<Check> {
        if run.phase == Phase::Prepare {
            return None;
        }
        if self.contact.iter().filter(|&&c| c).count() < MIN_WORKING {
            let off = (0..NUM_CHANNELS).find(|&c| !self.contact[c]).unwrap_or(0);
            return Some(Check::NoContact(off));
        }
        let usable = self.usable();
        Some(match (run.step, run.phase) {
            (_, Phase::Prepare) => unreachable!(),
            (Step::Rest, _) => match self.stillness_recent() {
                (c, r) if r > 1.0 => Check::Moving(c),
                _ => Check::Clean,
            },
            (Step::Hard(_), Phase::Record) => {
                if Self::activity(&self.z, &usable) >= HARD_MIN_ACTIVITY { Check::Clean } else { Check::TooWeak }
            }
            (Step::Medium(f), Phase::Record) => {
                let h = self.of_hard(f);
                if Self::activity(&self.z, &usable) < MEDIUM_MIN_ACTIVITY || h < MEDIUM_MIN {
                    Check::TooWeak
                } else if h > MEDIUM_MAX {
                    Check::TooStrong
                } else {
                    Check::Clean
                }
            }
            (Step::Faint(f, _), Phase::Record) => {
                let p = self.power[f];
                let other = (0..NUM_FINGERS)
                    .filter(|&g| g != f && self.templates[g].is_some() && self.power[g] > self.stasis[g].max(p))
                    .max_by(|&a, &b| self.power[a].total_cmp(&self.power[b]));
                if p > 1.0 {
                    Check::TooStrong
                } else if p < FAINT_MIN * self.stasis[f] {
                    Check::TooWeak
                } else if let Some(g) = other {
                    Check::OtherFinger(g)
                } else {
                    Check::Clean
                }
            }
            (Step::Train(f), Phase::Record) => {
                let goal = Self::goal(run.trial);
                match self.band[f] {
                    s if s == goal => Check::Clean,
                    Band::Pending if goal == Band::Minus => Check::Clean,
                    Band::Over => Check::TooStrong,
                    Band::Plus => Check::TooStrong,
                    _ => Check::TooWeak,
                }
            }
            (Step::Train(f), Phase::Relax) => {
                if self.band[f] == Band::Rest { Check::Clean } else { Check::NotRelaxed(f) }
            }
            (_, Phase::Relax) => match self.stillness_z() {
                (c, r) if r > 1.0 => Check::NotRelaxed(c),
                _ => Check::Clean,
            },
        })
    }

    fn calibration_step(&mut self) {
        let Some(mut run) = self.run.take() else { return };
        let clean = self.signal_check(&run) == Some(Check::Clean);
        run.t += 1;
        let done = match (run.phase, run.step) {
            (Phase::Prepare, _) => run.t >= Self::duration(&run),
            (Phase::Record, Step::Train(f)) => {
                run.hold = if self.band[f] == Self::goal(run.trial) { run.hold + 1 } else { 0 };
                let hit = run.hold >= samples(TRIAL_HOLD_S);
                if hit || run.t >= Self::duration(&run) {
                    self.trial_result(&mut run, f, hit);
                    true
                } else {
                    false
                }
            }
            (Phase::Record, step) => {
                if clean && run.t > samples(SETTLE_S) {
                    run.rec.push(if step == Step::Rest { self.envelope.map(f64::from) } else { self.z });
                    run.good += 1;
                    run.stuck = 0;
                } else {
                    run.stuck += 1;
                }
                run.good >= Self::duration(&run)
            }
            (Phase::Relax, step) => {
                if clean {
                    run.good += 1;
                    run.stuck = 0;
                } else {
                    run.good = 0;
                    run.stuck += 1;
                }
                let need = if matches!(step, Step::Train(_)) { Self::duration(&run) } else { samples(RELAXED_S) };
                run.t >= Self::duration(&run) && run.good >= need
            }
        };
        if !done {
            self.run = Some(run);
            return;
        }
        let keep = match (run.phase, run.step) {
            (Phase::Prepare, _) => {
                run.phase = Phase::Record;
                true
            }
            (Phase::Record, Step::Rest) => {
                let next = self.finish_rest(&mut run);
                self.enter(&mut run, next)
            }
            (Phase::Record, _) => {
                if !matches!(run.step, Step::Train(_)) {
                    run.next = self.finish_recording(&mut run);
                }
                run.phase = Phase::Relax;
                true
            }
            (Phase::Relax, Step::Train(f)) => {
                if run.trial + 1 < TRIALS {
                    run.trial += 1;
                    run.phase = Phase::Prepare;
                    true
                } else {
                    let next = Self::after_training(&run, f);
                    self.enter(&mut run, next)
                }
            }
            (Phase::Relax, _) => {
                let next = run.next.take();
                self.enter(&mut run, next)
            }
        };
        run.t = 0;
        run.good = 0;
        run.stuck = 0;
        run.hold = 0;
        run.rec.clear();
        if keep {
            self.run = Some(run);
        }
    }

    /// Moves the run to `step`'s prepare phase; false when there is none (run over).
    fn enter(&mut self, run: &mut Run, step: Option<Step>) -> bool {
        let Some(step) = step else { return false };
        if let Step::Train(_) = step {
            if run.step.finger() != step.finger() || !matches!(run.step, Step::Train(_)) {
                run.trial = 0;
                run.streak = 0;
            }
        }
        run.step = step;
        run.phase = Phase::Prepare;
        run.steps_done += 1;
        run.t = 0;
        run.good = 0;
        run.stuck = 0;
        run.hold = 0;
        run.rec.clear();
        true
    }

    /// Rest recording done: the reference everything else is measured against.
    fn finish_rest(&mut self, run: &mut Run) -> Option<Step> {
        let n = run.rec.len().max(1);
        let mean = mean(&run.rec);
        let scale: Features = std::array::from_fn(|c| {
            let var = run.rec.iter().map(|x| (x[c] - mean[c]).powi(2)).sum::<f64>() / n as f64;
            var.sqrt().max(MIN_SCALE_UV)
        });
        let seen: [bool; NUM_CHANNELS] = std::array::from_fn(|c| self.contact[c] && mean[c] > 0.0);
        let z = run.rec.iter().map(|x| std::array::from_fn(|c| (x[c] - mean[c]) / scale[c])).collect();
        self.rest = Some(RestModel { mean, scale, seen, z });
        // every pattern was measured against the old rest: start over
        self.templates = [None; NUM_FINGERS];
        self.data = Default::default();
        self.stasis = [STASIS_MIN; NUM_FINGERS];
        self.level = [LEVEL_MAX; NUM_FINGERS];
        self.report = [FingerReport::default(); NUM_FINGERS];
        if seen.iter().filter(|&&s| s).count() < MIN_WORKING {
            return None;
        }
        run.calibrate.first().map(|&f| Step::Hard(f))
    }

    /// A finger recording is done: learn from it and decide the next step.
    fn finish_recording(&mut self, run: &mut Run) -> Option<Step> {
        let seen = self.seen();
        let rec = std::mem::take(&mut run.rec);
        match run.step {
            Step::Hard(f) => {
                self.data[f] = FingerData { hard: Some(mean(&rec)), ..Default::default() };
                self.templates[f] = None;
                Some(Step::Medium(f))
            }
            Step::Medium(f) => {
                let d = &mut self.data[f];
                d.shapes = direction(&mean(&rec), &seen).into_iter().collect();
                d.medium = rec;
                self.rebuild_template(f);
                match FAINT_TARGETS.first() {
                    Some(&t) if self.templates[f].is_some() && t >= TARGET_ROOM * self.stasis[f] => Some(Step::Faint(f, 0)),
                    _ => self.after_finger(run, f),
                }
            }
            Step::Faint(f, k) => {
                let level = percentile(rec.iter().map(|z| Self::fit(&self.templates, z, &seen)[f]).collect(), 0.5);
                let prev = self.data[f].faint_levels.iter().copied().fold(f64::INFINITY, f64::min);
                let d = &mut self.data[f];
                d.shapes.extend(direction(&mean(&rec), &seen));
                d.faint_levels.push(level);
                if level < prev {
                    d.faintest = rec;
                }
                self.rebuild_template(f);
                let room = FAINT_TARGETS.get(k + 1).is_some_and(|&t| t >= TARGET_ROOM * self.stasis[f]);
                if room && level < prev * PROGRESS_RATIO {
                    Some(Step::Faint(f, k + 1))
                } else {
                    self.after_finger(run, f)
                }
            }
            Step::Rest | Step::Train(_) => None,
        }
    }

    /// The finger's pattern is the average of its unit patterns from every
    /// round, scaled so that the softest part of the medium flex reads 1.
    fn rebuild_template(&mut self, f: usize) {
        let seen = self.seen();
        let d = &self.data[f];
        self.templates[f] = direction(&mean(&d.shapes), &seen).and_then(|dir| {
            let a = percentile(d.medium.iter().map(|z| dot(&dir, z, &seen)).collect(), CEILING_PERCENTILE);
            (a > 1e-6).then(|| dir.map(|v| v * a))
        });
        self.update_stasis();
    }

    /// Stasis band of each finger: above its resting noise with every template in the fit.
    fn update_stasis(&mut self) {
        let Some(rest) = &self.rest else { return };
        let fits: Vec<[f64; NUM_FINGERS]> = rest.z.iter().map(|z| Self::fit(&self.templates, z, &rest.seen)).collect();
        for f in 0..NUM_FINGERS {
            let p = percentile(fits.iter().map(|w| w[f]).collect(), STASIS_PERCENTILE);
            self.stasis[f] = (p + STASIS_MARGIN).max(STASIS_MIN);
        }
    }

    /// The step after finger `f`'s calibration: the next finger, or the end of
    /// phase 1 (then training, if asked for).
    fn after_finger(&mut self, run: &mut Run, f: usize) -> Option<Step> {
        let pos = run.calibrate.iter().position(|&g| g == f).unwrap_or(usize::MAX - 1);
        if let Some(&g) = run.calibrate.get(pos + 1) {
            return Some(Step::Hard(g));
        }
        self.finalize();
        if !run.train_after {
            return None;
        }
        run.train = run.calibrate.iter().copied().filter(|&g| self.templates[g].is_some()).collect();
        run.train.first().map(|&g| Step::Train(g))
    }

    fn after_training(run: &Run, f: usize) -> Option<Step> {
        let pos = run.train.iter().position(|&g| g == f)?;
        run.train.get(pos + 1).map(|&g| Step::Train(g))
    }

    /// End of phase 1: every finger's faint level and isolation, measured with
    /// all the templates competing for the signal.
    fn finalize(&mut self) {
        self.update_stasis();
        let seen = self.seen();
        for f in 0..NUM_FINGERS {
            if self.templates[f].is_none() {
                continue;
            }
            let fits: Vec<[f64; NUM_FINGERS]> = self.data[f].faintest.iter().map(|z| Self::fit(&self.templates, z, &seen)).collect();
            let faint = if fits.is_empty() { LEVEL_MAX } else { percentile(fits.iter().map(|w| w[f]).collect(), 0.5) };
            let on_top = fits
                .iter()
                .filter(|w| w[f] > self.stasis[f] && (0..NUM_FINGERS).all(|g| w[g] <= w[f]))
                .count();
            self.level[f] = faint.clamp(self.stasis[f] * LEVEL_MIN_ROOM, LEVEL_MAX);
            self.report[f] = FingerReport {
                faint,
                rounds: self.data[f].faint_levels.len(),
                isolation: on_top as f64 / fits.len().max(1) as f64,
                ..FingerReport::default()
            };
        }
    }

    /// A training trial ended: tally it and adapt the levels.
    fn trial_result(&mut self, run: &mut Run, f: usize, hit: bool) {
        self.report[f].trials += 1;
        if hit {
            self.report[f].hits += 1;
            run.streak += 1;
            if run.streak >= STREAK {
                run.streak = 0;
                self.level[f] *= LEVEL_DOWN;
            }
        } else {
            run.streak = 0;
            self.level[f] *= LEVEL_UP;
        }
        let lo = self.stasis[f] * LEVEL_MIN_ROOM;
        self.level[f] = self.level[f].clamp(lo, LEVEL_MAX.max(lo));
    }

    pub fn status(&self) -> Option<Status> {
        let run = self.run.as_ref()?;
        let duration = Self::duration(run).max(1);
        let (progress, left) = match (run.step, run.phase) {
            (Step::Train(_), Phase::Record) => {
                (run.hold as f32 / samples(TRIAL_HOLD_S) as f32, (duration.saturating_sub(run.t)) as f32 / SAMPLE_RATE)
            }
            (_, Phase::Record) | (Step::Train(_), Phase::Relax) => {
                (run.good as f32 / duration as f32, duration.saturating_sub(run.good) as f32 / SAMPLE_RATE)
            }
            _ => (run.t as f32 / duration as f32, duration.saturating_sub(run.t) as f32 / SAMPLE_RATE),
        };
        let (meter_kind, meter, target) = self.meter(run);
        let rest = usize::from(run.has_rest);
        let trains = if run.train_after { run.calibrate.len() } else { run.train.len() };
        let segment = match run.step {
            Step::Rest => 0,
            Step::Hard(f) | Step::Medium(f) | Step::Faint(f, _) => {
                rest + run.calibrate.iter().position(|&g| g == f).unwrap_or(0)
            }
            Step::Train(f) => rest + run.calibrate.len() + run.train.iter().position(|&g| g == f).unwrap_or(0),
        };
        let (round, rounds) = match run.step {
            Step::Faint(_, k) => (k, FAINT_TARGETS.len()),
            Step::Train(_) => (run.trial, TRIALS),
            _ => (0, 0),
        };
        Some(Status {
            step: run.step,
            phase: run.phase,
            phase_progress: progress.clamp(0.0, 1.0),
            seconds_left: left,
            check: self.signal_check(run),
            stuck_seconds: run.stuck as f32 / SAMPLE_RATE,
            meter_kind,
            meter,
            target,
            segment,
            segments: rest + run.calibrate.len() + trains,
            round,
            rounds,
            goal: matches!(run.step, Step::Train(_)).then(|| Self::goal(run.trial)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(state: &mut u32) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        (*state as f64 / u32::MAX as f64) * 2.0 - 1.0
    }

    /// Envelope share of each finger on each grid site (upper row thumb side →
    /// pinky side, then the middle row), with realistic overlap between neighbours.
    const PATTERNS: [[f64; NUM_CHANNELS]; NUM_FINGERS] = [
        [0.15, 0.05, 0.0, 0.0, 0.9, 0.35, 0.05, 0.0],
        [0.6, 0.85, 0.25, 0.0, 0.3, 0.55, 0.15, 0.0],
        [0.1, 0.45, 0.85, 0.25, 0.0, 0.3, 0.8, 0.2],
        [0.0, 0.1, 0.4, 0.9, 0.0, 0.05, 0.35, 0.85],
    ];

    /// A simulated user: tension per finger (1 = a hard flex), which they steer
    /// towards whatever the coach asks for by watching its bar.
    struct User {
        rng: u32,
        drive: [f64; NUM_FINGERS],
        off: Option<usize>,
    }

    impl User {
        fn new(seed: u32) -> Self {
            Self { rng: seed, drive: [0.0; NUM_FINGERS], off: None }
        }

        fn envelope(&mut self) -> ([f32; NUM_CHANNELS], [bool; NUM_CHANNELS]) {
            let env = std::array::from_fn(|c| {
                let base = 3.0 * (1.0 + 0.12 * noise(&mut self.rng));
                let sig: f64 = (0..NUM_FINGERS).map(|f| self.drive[f] * PATTERNS[f][c]).sum();
                let v = base + 40.0 * sig * (1.0 + 0.15 * noise(&mut self.rng));
                if Some(c) == self.off { 0.0 } else { v as f32 }
            });
            (env, std::array::from_fn(|c| Some(c) != self.off))
        }

        fn step(&mut self, fc: &mut FingerControl) {
            let (env, contact) = self.envelope();
            fc.process(&env, &contact);
        }

        /// Follow the coach for one sample.
        fn follow(&mut self, fc: &mut FingerControl) {
            if let Some(st) = fc.status() {
                let f = st.step.finger();
                self.drive = [0.0; NUM_FINGERS];
                if let (Some(f), Phase::Record) = (f, st.phase) {
                    self.drive = self.prev_drive_for(f, &st);
                }
            }
            self.step(fc);
        }

        fn prev_drive_for(&mut self, f: usize, st: &Status) -> [f64; NUM_FINGERS] {
            let mut d = [0.0; NUM_FINGERS];
            let cur = self.last[f];
            d[f] = match st.meter_kind {
                Meter::Activity => 1.0,
                _ => {
                    let aim = (st.target.0 + st.target.1) / 2.0;
                    // start from a sensible guess, then steer by the bar
                    let start = if cur == 0.0 { aim * 0.5 } else { cur };
                    (start + 0.004 * (aim - st.meter)).clamp(0.0, 1.2)
                }
            };
            self.last = d;
            d
        }
    }
}
