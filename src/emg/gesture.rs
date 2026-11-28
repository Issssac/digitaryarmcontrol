//! Gesture control for the muscle-pair scheme: seven intended movements, each
//! learned as a pattern across all eight electrodes, so it doesn't matter which
//! electrode is on which muscle. Each gesture drives one control axis in one
//! direction (see `AXES` and reach.rs), at a speed that grows with how firmly
//! it is tensed. Power is 0 at rest and 1 at the ceiling, the softest part of
//! a half-strength contraction:
//!
//! | power                   | its axis                                  |
//! |-------------------------|-------------------------------------------|
//! | within the stasis band  | holds                                     |
//! | past the rest buffer    | moves, faster up to the level             |
//! | at the level and above  | full speed (× gain)                       |
//!
//! At most two gestures drive at once: the strongest, and the runner-up only
//! when it stands 2 standard deviations above what the strongest one's
//! crosstalk would put on it (learned from the calibration recordings).
//!
//! Calibration and training follow the finger control (finger.rs), whose
//! machinery this mirrors:
//! - Rest in three postures first (limp on the table, forearm raised with the
//!   elbow down, arm held up unsupported). Each raised posture becomes a
//!   pattern of its own, a slow fit says which posture the arm is in, and that
//!   posture's baseline is taken out before the gestures are fitted.
//! - Phase 1 learns each gesture from a hard contraction, then about half as
//!   hard (which sets the ceiling), then a faint intention, re-learning the
//!   pattern from every round so faint gestures still decode.
//! - Phase 2 has the user hold each gesture at its level, with only that
//!   gesture's axis moving. Two hits in a row lower the level a step and a miss
//!   raises it, so it settles on the lightest intention they can hold reliably.

use serde::{Deserialize, Serialize};

use super::cyton::{NUM_CHANNELS, SAMPLE_RATE};
use super::decoder::{nnls, raw_features, NNLS_MAX, with_effort, CovTracker, Features, LdaModel, COV_WIN, CovWindow, MAX_LDA_CLASSES, NUM_COV, NUM_FEATS};
use crate::arm::NUM_JOINTS;

pub const NUM_GESTURES: usize = 7;
/// Control axis (base, reach, elbow, gripper) and direction each gesture
/// drives: wrist rotate in / out → base + / −, tricep extend / bicep curl →
/// reach out / in, wrist back / curl → elbow + / −, grip → gripper closed.
pub const AXES: [(usize, f32); NUM_GESTURES] =
    [(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0), (3, -1.0)];

/// Digit array: eight electrodes against a shared REF, CH1–3 on the finger flexors and CH4–6 on
/// the finger extensors (thumb side to little-finger side), CH7 biceps, CH8
/// triceps. Its movements, in order: thumb flex / extend, index flex / extend,
/// ring + little flex / extend, bicep, tricep.
pub const NUM_DIGITS: usize = 8;
/// Thumb flex / extend → gripper close / open, index flex / extend → tip down
/// / up (elbow + / −), ring + little flex / extend → base clockwise /
/// counter-clockwise seen from above (base − / +), bicep / tricep → reach in / out.
pub const DIGIT_AXES: [(usize, f32); NUM_DIGITS] =
    [(3, -1.0), (3, 1.0), (2, 1.0), (2, -1.0), (0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0)];
/// Rest postures recorded in calibration: limp on the table (the reference),
/// forearm raised with the elbow down, arm held up unsupported.
pub const POSTURES: usize = 3;
/// Posture is judged on an envelope averaged this long, so a gesture pulse
/// doesn't read as a change of posture.
const POSTURE_TAU_S: f32 = 0.5;
/// The posture fit runs on every this-many-th sample (the slow envelope barely moves in between).
const POSTURE_EVERY: u32 = 10;
/// Filler for the unused slots of a fixed-size list of patterns.
const NO_PATTERN: Features = [0.0; NUM_CHANNELS];
/// Fit ridges, relative to each pattern's size (see finger.rs).
const GESTURE_RIDGE: f64 = 0.02;
const POSTURE_RIDGE: f64 = 0.001;

/// Smoothing of the gesture powers.
const POWER_TAU_S: f32 = 0.06;
/// The stasis band sits just above resting noise: this percentile of resting
/// power plus a small margin, and never narrower than STASIS_MIN.
const STASIS_PERCENTILE: f64 = 0.99;
const STASIS_MARGIN: f64 = 0.02;
const STASIS_MIN: f64 = 0.04;
/// The ceiling sits at this percentile of the half-strength contraction.
const CEILING_PERCENTILE: f64 = 0.2;
/// Faint round: the box runs from just above noise (FAINT_MIN × stasis) up to FAINT_TOP.
const FAINT_MIN: f64 = 1.3;
const FAINT_TOP: f64 = 0.3;
/// The level (power for full speed): at most this, at least this multiple of stasis.
const LEVEL_MAX: f64 = 0.6;
const LEVEL_MIN_ROOM: f64 = 2.2;
/// Moving needs this fraction of the way from stasis to the level: the buffer
/// that keeps resting noise from twitching the axis.
const REST_BUFFER: f64 = 0.4;
/// A gesture only starts driving after this long past the buffer.
const COMMIT_S: f32 = 0.1;
/// Default gain on the axis speed.
const DEFAULT_GAIN: f32 = 1.0;
/// Up to two gestures decode at once. The strongest always counts; the
/// runner-up joins it (a compound movement) once it has read this many of its
/// standard deviations above what the strongest one's crosstalk onto it
/// predicts (its coupling) for COMPOUND_COMMIT_S, i.e. it is really being made
/// too. The pair then holds until the weaker of the two drops below
/// COMPOUND_EXIT_SIGMAS, whichever of them is the stronger by then.
const COMPOUND_SIGMAS: f64 = 2.0;
const COMPOUND_EXIT_SIGMAS: f64 = 1.0;
const COMPOUND_COMMIT_S: f32 = 0.1;
/// Calibration: a sample is a hit when it points within this cosine of its
/// own movement's pattern; the rest (wandering moments, the wrong
/// contraction) are misses and don't shape the pattern. Only its own pattern
/// is asked: movements that look alike (fingers on one forearm read 0.96–0.98
/// of each other) would otherwise throw each other's clean holds away.
const HIT_COS: f64 = 0.8;
/// A round shapes the pattern only if at least this share of it was hits.
const ROUND_MIN_HITS: f64 = 0.25;
/// Live compound learning: a pair decoded together for at least LEARN_MIN_S is
/// one episode. Its activation, per unit of the two single-movement
/// predictions, becomes a compound pattern when the singles leave at least
/// COMPOUND_RESIDUAL of it unexplained (the two don't simply add up) and it
/// isn't really one of the two (cosine above COMPOUND_SINGLE_COS). Later
/// episodes of the pair refine it, as a running mean over the last LEARN_MEMORY.
const LEARN_MIN_S: f32 = 0.4;
const COMPOUND_RESIDUAL: f64 = 0.12;
const COMPOUND_SINGLE_COS: f64 = 0.97;
const LEARN_MEMORY: f64 = 8.0;
/// Floor on a coupling's standard deviation, so a very clean calibration can't
/// make every flicker of the runner-up count as a compound.
const COUPLING_MIN_SD: f64 = 0.02;
/// Which movement it is (see decoder.rs): a shrinkage LDA over each 200 ms
/// window's relative log-RMS and inter-channel correlations, trained on the
/// calibration holds and the rest postures. The patterns say how hard; the
/// classifier gates each movement by how sure it is that this is the one (and
/// not a rest). Shrinkage of its within-class covariance, and the gate's
/// temperature in discriminant units.
const LDA_SHRINK: f64 = 0.25;
const LDA_TEMP: f64 = 2.0;
/// Calibration keeps the classifier features of every FEAT_EVERY-th sample.
const FEAT_EVERY: usize = 10;
/// Each hard and half-strength recording is this many separate holds (GESTURE_S
/// in all), so the classifier sees more than one way of making the movement.
const HOLDS: u32 = 2;
/// Contact check at the limp rest: a channel whose envelope is BAD_RATIO times
/// the median channel's (and above BAD_MIN_UV) has a bad contact. It is left
/// out of decoding and of the channel average.
const BAD_RATIO: f64 = 6.0;
const BAD_MIN_UV: f64 = 10.0;

// Signal checks.
/// Hard contraction: RMS of the positive z-scores across the electrodes.
const HARD_MIN_ACTIVITY: f64 = 5.0;
const MEDIUM_MIN_ACTIVITY: f64 = 2.0;
/// Half-strength contraction, as a fraction of the hard one.
const MEDIUM_MIN: f64 = 0.15;
const MEDIUM_MAX: f64 = 0.85;
pub const MEDIUM_TARGET: (f64, f64) = (0.35, 0.65);
/// Relaxed: no channel above this many resting standard deviations.
const RELAX_Z: f64 = 4.0;
/// Rest: a channel jumping above this multiple of its recent level is movement.
const STILL_RATIO: f32 = 2.5;
const STILL_MARGIN_UV: f32 = 1.5;
const REST_CEILING_UV: f32 = 30.0;
const RECENT_TAU_S: f32 = 1.5;
/// Fewer working electrodes than this and there is nothing to learn from.
const MIN_WORKING: usize = 4;
/// Resting-noise floor of a channel (µV of envelope).
const MIN_SCALE_UV: f64 = 0.2;

// Timing, in seconds. Every phase-1 step waits for the user to say they're
// ready (`ready`), then counts down before it records. Record phases count
// only clean samples.
const COUNTDOWN_S: f32 = 3.0;
const REST_S: f32 = 10.0;
const GESTURE_S: f32 = 5.0;
const RELAX_S: f32 = 1.5;
const RELAXED_S: f32 = 0.3;
const SETTLE_S: f32 = 0.5;

// Training.
pub const TRIALS: usize = 6;
const CUE_S: f32 = 1.5;
const TRIAL_TIMEOUT_S: f32 = 10.0;
const TRIAL_HOLD_S: f32 = 1.0;
const TRAIN_RELAX_S: f32 = 1.0;
/// A trial asks for power within these multiples of the level.
const TRIAL_LO: f64 = 0.7;
const TRIAL_HI: f64 = 1.6;
/// Staircase: two hits in a row lower the level by LEVEL_DOWN, a miss raises
/// it by LEVEL_UP. That settles where about 70% of attempts succeed.
const LEVEL_DOWN: f64 = 0.85;
const LEVEL_UP: f64 = 1.15;

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

/// Channels far above the others at rest: BAD_RATIO times the median channel and above BAD_MIN_UV.
fn outliers(mean: &Features) -> [bool; NUM_CHANNELS] {
    let mut v: Vec<f64> = mean.iter().copied().filter(|&m| m > 0.0).collect();
    if v.len() < 3 {
        return [false; NUM_CHANNELS];
    }
    v.sort_by(f64::total_cmp);
    let med = v[v.len() / 2];
    std::array::from_fn(|c| mean[c] > BAD_RATIO * med && mean[c] > BAD_MIN_UV)
}

fn cosine(a: &Features, b: &Features, usable: &[bool; NUM_CHANNELS]) -> f64 {
    dot(a, b, usable) / (dot(a, a, usable) * dot(b, b, usable)).sqrt().max(1e-12)
}

/// Unit vector along `x` over the usable channels; None if it is ~zero.
fn direction(x: &Features, usable: &[bool; NUM_CHANNELS]) -> Option<Features> {
    let n = dot(x, x, usable).sqrt();
    (n > 1e-9).then(|| std::array::from_fn(|c| if usable[c] { x[c] / n } else { 0.0 }))
}

/// One step of a calibration or training run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    /// Arm relaxed in posture p: 0 sets the resting level and noise of every
    /// channel, the others record how a raised arm shifts them.
    Rest(usize),
    /// A hard contraction of this gesture: the top of its range.
    Hard(usize),
    /// About half as hard: sets the ceiling.
    Medium(usize),
    /// A faint intention: tension only.
    Faint(usize),
    /// Training: hold this gesture at its level.
    Train(usize),
}

impl Step {
    pub fn gesture(self) -> Option<usize> {
        match self {
            Step::Rest(_) => None,
            Step::Hard(g) | Step::Medium(g) | Step::Faint(g) | Step::Train(g) => Some(g),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    /// Read the cue, then (once ready) count down. Training: the cue for the next trial.
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
    /// Faint round: above the top of the box.
    Fainter,
    /// Training: above the trial's band; half-strength: as hard as the hard one.
    TooStrong,
    /// Another gesture's pattern is stronger than the cued one.
    OtherGesture(usize),
    /// Training: this gesture hasn't let go; otherwise this channel.
    NotRelaxed(usize),
}

/// What the coach's bar shows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Meter {
    /// Hard contraction: activity relative to the minimum needed (1 = enough).
    Activity,
    /// Half-strength: fraction of the hard one.
    OfHard,
    /// Gesture power: 0 = rest, 1 = ceiling.
    Power,
    /// Rest / relax: the busiest channel against its limit (1 = at the limit).
    Stillness,
}

/// Power edges for one gesture (0 = rest, 1 = ceiling).
#[derive(Clone, Copy, Debug)]
pub struct Bands {
    /// Below this the axis holds.
    pub stasis: f64,
    /// Moving needs this much (the buffer above stasis).
    pub rest_exit: f64,
    /// Full speed from here.
    pub level: f64,
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
    /// Training: the level (NaN otherwise).
    pub aim: f64,
    /// Current segment (each rest posture, each calibration round, each
    /// gesture's training) and how many there are.
    pub segment: usize,
    pub segments: usize,
    /// Training trial (0-based) and how many (0 = n/a).
    pub round: usize,
    pub rounds: usize,
    /// Prepare: holding until the user says they're ready (the countdown hasn't started).
    pub waiting: bool,
}

/// What phases 1 and 2 found for one gesture.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct GestureReport {
    pub calibrated: bool,
    /// Level of the faint round, as a fraction of the ceiling.
    pub faint: f64,
    /// Share of the faint round in which this gesture came out on top and above stasis.
    pub isolation: f64,
    pub hits: u32,
    pub trials: u32,
    /// Current level (full speed) and noise floor.
    pub level: f64,
    pub stasis: f64,
    /// The faint round looked more like another gesture, so it wasn't used.
    pub mixed: bool,
}

/// Per-gesture calibration recordings.
#[derive(Clone, Default, Serialize, Deserialize)]
struct GestureData {
    hard: Option<Features>,
    /// z samples of the hard contraction (for the coupling between gestures).
    hard_rec: Vec<Features>,
    /// z samples of the half-strength contraction (they define the ceiling).
    medium: Vec<Features>,
    /// Not used any more; kept so calibrations saved with it still load.
    shapes: Vec<Features>,
    /// z samples of the faint round (for the level and isolation).
    faint: Vec<Features>,
    mixed: bool,
}

/// The limp rest recording: the reference every z-score is measured from.
#[derive(Clone, Serialize, Deserialize)]
struct RestModel {
    mean: Features,
    scale: Features,
    /// Channel had contact during the rest recording.
    seen: [bool; NUM_CHANNELS],
}

/// Two movements made together, learned in live use: the activation they make
/// per unit of their combined power, and the share of it that is `pair.0`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Compound {
    pair: (usize, usize),
    pattern: Features,
    mix: f64,
    /// Episodes it was learned from (capped at LEARN_MEMORY).
    n: f64,
}

/// The pair being decoded together right now, and what it has made so far.
#[derive(Default)]
struct Episode {
    pair: Option<(usize, usize)>,
    sum: Features,
    mix: f64,
    /// Combined single-movement power of the two, summed over the episode.
    total: f64,
    n: u32,
    /// A channel lost contact during it, so its activation is incomplete.
    lost: bool,
}

/// The classifier's input over one calibration recording: the filtered
/// signal's 200 ms covariance at every FEAT_EVERY-th sample (NUM_COV each,
/// flattened; NaN while the window was still filling). Kept as covariances,
/// not features, so the features follow the channels in use when it is fitted.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct FeatRec {
    /// The movement, or G + posture for a rest.
    class: usize,
    /// 0 rest, 1 hard, 2 half-strength, 3 faint.
    kind: u8,
    data: Vec<f32>,
}

impl FeatRec {
    fn rows(&self) -> impl Iterator<Item = CovWindow> + '_ {
        self.data.chunks_exact(NUM_COV).filter(|r| !r[0].is_nan()).map(|r| std::array::from_fn(|d| r[d] as f64))
    }

    /// Covariance at sample `i` of the recording.
    fn at(&self, i: usize) -> Option<CovWindow> {
        let k = i / FEAT_EVERY;
        let r = self.data.get(k * NUM_COV..(k + 1) * NUM_COV)?;
        (!r[0].is_nan()).then(|| std::array::from_fn(|d| r[d] as f64))
    }
}

fn kind_of(step: Step) -> u8 {
    match step {
        Step::Rest(_) => 0,
        Step::Hard(_) => 1,
        Step::Medium(_) => 2,
        Step::Faint(_) | Step::Train(_) => 3,
    }
}

/// The fitted classifier, and the limp rest's mean log-RMS its effort feature is measured from.
#[derive(Clone)]
struct Lda {
    model: LdaModel,
    rest_mlr: f64,
}

struct Run {
    step: Step,
    phase: Phase,
    /// Phase 1: the user said they're ready for this step, so the countdown runs.
    ready: bool,
    t: u32,
    good: u32,
    stuck: u32,
    /// Clean samples of the current Record phase: envelopes for the limp rest,
    /// raw z for the other postures, posture-corrected z for the gestures.
    rec: Vec<Features>,
    /// Classifier features of `rec` (see FeatRec).
    feats: Vec<f32>,
    /// Holds of the current hard or half-strength recording already made.
    holds: u32,
    /// Decided when a recording finishes, entered after its relax phase.
    next: Option<Step>,
    /// Rest postures in this run (all of them, or none for a single gesture / training).
    rests: usize,
    calibrate: Vec<usize>,
    train: Vec<usize>,
    train_after: bool,
    /// Training: trial number, samples in the band, and hits in a row.
    trial: usize,
    hold: u32,
    streak: u32,
}

pub struct GestureControl<const G: usize> {
    /// Axis (base, reach, elbow, gripper) and direction each movement drives.
    axes: [(usize, f32); G],
    rest: Option<RestModel>,
    /// Pattern of each raised posture (posture p at index p − 1), in z units.
    postures: Vec<Features>,
    /// Raw z samples of each posture's rest recording (index = posture).
    posture_rest: Vec<Vec<Features>>,
    /// Each gesture's noise floor in each recorded posture.
    stasis_by: Vec<[f64; G]>,
    z_slow: Features,
    posture_w: Vec<f64>,
    z_raw: Features,
    data: [GestureData; G],
    templates: [Option<Features>; G],
    /// Current noise floor of each gesture: the posture floors, blended by posture.
    stasis: [f64; G],
    /// Power at which each gesture drives its axis at full speed.
    level: [f64; G],
    report: [GestureReport; G],
    pub gain: f32,
    envelope: [f32; NUM_CHANNELS],
    contact: [bool; NUM_CHANNELS],
    recent: [f32; NUM_CHANNELS],
    z: Features,
    /// What the patterns are fitted to this sample: z, with the posture taken
    /// out when decoding by the patterns alone.
    x: Features,
    /// Samples processed, for the posture fit (every POSTURE_EVERY-th).
    tick: u32,
    /// A compound was learned: the noise floors and coupling are worked out
    /// again once nothing is moving, so the recompute never stalls a movement.
    refit_due: bool,
    /// Compound patterns learned in live use.
    compounds: Vec<Compound>,
    episode: Episode,
    /// Bumped whenever a compound is learned or refined, so the page knows to save.
    compound_updates: u32,
    /// Smoothed powers from the single movements alone (no compounds in the fit):
    /// what compound learning watches.
    singles: [f64; G],
    power: [f64; G],
    /// Driving its axis (with hysteresis down to stasis).
    active: [bool; G],
    /// Time each gesture has been past its buffer while not yet active.
    rising_t: [f32; G],
    /// coupling[a][b]: b's power when only a is made, as (slope on a's power,
    /// standard deviation around it). Worked out from the recordings, not saved.
    coupling: [[(f64, f64); G]; G],
    /// The two gestures currently decoded together, if any.
    compound: Option<(usize, usize)>,
    /// Time the runner-up has stood out from the strongest one's crosstalk.
    compound_t: f32,
    run: Option<Run>,
    inputs: [f32; NUM_JOINTS],
    /// Rolling covariance of the filtered (not rectified) signal, for the classifier.
    cov: CovTracker,
    /// Classifier features of the current sample (raw effort slot), once the window is full.
    feat: Option<[f64; NUM_FEATS]>,
    feats: Vec<FeatRec>,
    lda: Option<Lda>,
    /// Use the classifier when there is one (off: the patterns alone, as before).
    pub use_lda: bool,
    /// Learn compound patterns in live use (only when decoding by the patterns alone).
    pub learn_in_use: bool,
    /// The pipeline subtracts the channel average, so a bad contact found at
    /// rest spoiled every channel: the rest is recorded again without it.
    pub car: bool,
}

/// Everything calibration and training learned, to save and restore it.
/// Saved field by field as fixed-length tuples, the way serde saves a plain
/// array, so muscle-pair calibrations saved before this was generic still load.
#[derive(Serialize, Deserialize)]
pub struct Calibration<const G: usize> {
    rest: Option<RestModel>,
    postures: Vec<Features>,
    posture_rest: Vec<Vec<Features>>,
    #[serde(with = "fixed")]
    data: [GestureData; G],
    #[serde(with = "fixed")]
    templates: [Option<Features>; G],
    #[serde(with = "fixed")]
    level: [f64; G],
    #[serde(with = "fixed")]
    report: [GestureReport; G],
    compounds: Vec<Compound>,
    feats: Vec<FeatRec>,
}

/// A calibration as format 6 saved it: before the classifier's features.
#[derive(Deserialize)]
pub struct CalibrationV6<const G: usize> {
    rest: Option<RestModel>,
    postures: Vec<Features>,
    posture_rest: Vec<Vec<Features>>,
    #[serde(with = "fixed")]
    data: [GestureData; G],
    #[serde(with = "fixed")]
    templates: [Option<Features>; G],
    #[serde(with = "fixed")]
    level: [f64; G],
    #[serde(with = "fixed")]
    report: [GestureReport; G],
    compounds: Vec<Compound>,
}

impl<const G: usize> From<CalibrationV6<G>> for Calibration<G> {
    fn from(c: CalibrationV6<G>) -> Self {
        Calibration {
            rest: c.rest,
            postures: c.postures,
            posture_rest: c.posture_rest,
            data: c.data,
            templates: c.templates,
            level: c.level,
            report: c.report,
            compounds: c.compounds,
            feats: Vec::new(),
        }
    }
}

/// A calibration as format 5 (and 4) saved it: before compounds were learned.
#[derive(Deserialize)]
pub struct CalibrationV5<const G: usize> {
    rest: Option<RestModel>,
    postures: Vec<Features>,
    posture_rest: Vec<Vec<Features>>,
    #[serde(with = "fixed")]
    data: [GestureData; G],
    #[serde(with = "fixed")]
    templates: [Option<Features>; G],
    #[serde(with = "fixed")]
    level: [f64; G],
    #[serde(with = "fixed")]
    report: [GestureReport; G],
}

impl<const G: usize> From<CalibrationV5<G>> for Calibration<G> {
    fn from(c: CalibrationV5<G>) -> Self {
        Calibration {
            rest: c.rest,
            postures: c.postures,
            posture_rest: c.posture_rest,
            data: c.data,
            templates: c.templates,
            level: c.level,
            report: c.report,
            compounds: Vec::new(),
            feats: Vec::new(),
        }
    }
}

/// Serde for `[T; N]` of any length, as a tuple of N elements (serde's own
/// array impls stop at 32 and don't cover a const-generic length).
mod fixed {
    use serde::de::{Error, SeqAccess, Visitor};
    use serde::ser::SerializeTuple;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::marker::PhantomData;

    pub fn serialize<S: Serializer, T: Serialize, const N: usize>(v: &[T; N], s: S) -> Result<S::Ok, S::Error> {
        let mut t = s.serialize_tuple(N)?;
        for x in v {
            t.serialize_element(x)?;
        }
        t.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>, T: Deserialize<'de>, const N: usize>(d: D) -> Result<[T; N], D::Error> {
        struct V<T, const N: usize>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>, const N: usize> Visitor<'de> for V<T, N> {
            type Value = [T; N];
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "a tuple of {N} elements")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<[T; N], A::Error> {
                let mut v = Vec::with_capacity(N);
                for i in 0..N {
                    v.push(seq.next_element()?.ok_or_else(|| A::Error::invalid_length(i, &self))?);
                }
                v.try_into().map_err(|_| A::Error::custom("wrong length"))
            }
        }
        d.deserialize_tuple(N, V::<T, N>(PhantomData))
    }
}

impl GestureControl<NUM_GESTURES> {
    /// Muscle pairs: the seven gestures of `AXES`.
    pub fn muscle() -> Self {
        Self::new(AXES)
    }
}

impl GestureControl<NUM_DIGITS> {
    /// The digit array: the eight movements of `DIGIT_AXES`.
    pub fn digits() -> Self {
        Self::new(DIGIT_AXES)
    }
}

impl<const G: usize> GestureControl<G> {
    pub fn new(axes: [(usize, f32); G]) -> Self {
        Self {
            axes,
            rest: None,
            postures: Vec::new(),
            posture_rest: Vec::new(),
            stasis_by: Vec::new(),
            z_slow: [0.0; NUM_CHANNELS],
            posture_w: Vec::new(),
            z_raw: [0.0; NUM_CHANNELS],
            data: std::array::from_fn(|_| GestureData::default()),
            templates: [None; G],
            stasis: [STASIS_MIN; G],
            level: [LEVEL_MAX; G],
            report: [GestureReport::default(); G],
            gain: DEFAULT_GAIN,
            envelope: [0.0; NUM_CHANNELS],
            contact: [true; NUM_CHANNELS],
            recent: [0.0; NUM_CHANNELS],
            z: [0.0; NUM_CHANNELS],
            tick: 0,
            refit_due: false,
            x: [0.0; NUM_CHANNELS],
            compounds: Vec::new(),
            episode: Episode::default(),
            compound_updates: 0,
            singles: [0.0; G],
            power: [0.0; G],
            active: [false; G],
            rising_t: [0.0; G],
            coupling: [[(0.0, COUPLING_MIN_SD); G]; G],
            compound: None,
            compound_t: 0.0,
            run: None,
            inputs: [0.0; NUM_JOINTS],
            cov: CovTracker::new(),
            feat: None,
            feats: Vec::new(),
            lda: None,
            use_lda: true,
            learn_in_use: true,
            car: false,
        }
    }

    pub fn calibration(&self) -> Calibration<G> {
        Calibration {
            rest: self.rest.clone(),
            postures: self.postures.clone(),
            posture_rest: self.posture_rest.clone(),
            data: self.data.clone(),
            templates: self.templates,
            level: self.level,
            report: self.report,
            compounds: self.compounds.clone(),
            feats: self.feats.clone(),
        }
    }

    pub fn restore(&mut self, cal: Calibration<G>) {
        self.run = None;
        self.posture_w = vec![0.0; cal.postures.len()];
        self.rest = cal.rest;
        self.postures = cal.postures;
        self.posture_rest = cal.posture_rest;
        self.data = cal.data;
        self.templates = cal.templates;
        self.level = cal.level;
        self.report = cal.report;
        self.compounds = cal
            .compounds
            .into_iter()
            .filter(|k| k.pair.0 < G && k.pair.1 < G && self.axes[k.pair.0].0 != self.axes[k.pair.1].0)
            .collect();
        self.feats = cal.feats;
        // patterns (and the classifier) are re-learned from the recordings, so a
        // calibration saved by an older version decodes as this one would
        self.relearn();
        self.update_coupling();
        self.reset_state();
    }

    /// Clears the live state after a reconnect (keeps everything learned).
    pub fn reset_state(&mut self) {
        self.recent = [0.0; NUM_CHANNELS];
        self.power = [0.0; G];
        self.singles = [0.0; G];
        self.episode = Episode::default();
        self.z_slow = [0.0; NUM_CHANNELS];
        self.posture_w.iter_mut().for_each(|w| *w = 0.0);
        self.active = [false; G];
        self.rising_t = [0.0; G];
        self.compound = None;
        self.compound_t = 0.0;
        self.inputs = [0.0; NUM_JOINTS];
        self.cov.reset();
        self.feat = None;
    }

    /// Some gesture is calibrated, so the decoder runs.
    pub fn trained(&self) -> bool {
        self.templates.iter().any(Option::is_some)
    }

    /// Every gesture is calibrated.
    pub fn calibrated(&self) -> bool {
        self.templates.iter().all(Option::is_some)
    }

    /// The gesture driving `axis` in either direction is calibrated.
    pub fn drives(&self, axis: usize) -> bool {
        self.axes.iter().enumerate().any(|(g, &(a, _))| a == axis && self.templates[g].is_some())
    }

    /// Axis and direction of each movement.
    pub fn axes(&self) -> [(usize, f32); G] {
        self.axes
    }

    pub fn powers(&self) -> [f64; G] {
        self.power
    }

    pub fn active(&self) -> [bool; G] {
        self.active
    }

    /// Axis commands (-1..1): base, reach, elbow, gripper.
    pub fn axis_inputs(&self) -> [f32; NUM_JOINTS] {
        self.inputs
    }

    pub fn report(&self) -> [GestureReport; G] {
        std::array::from_fn(|g| GestureReport {
            calibrated: self.templates[g].is_some(),
            level: self.bands(g).level,
            stasis: self.stasis[g],
            mixed: self.data[g].mixed,
            ..self.report[g]
        })
    }

    pub fn bands(&self, g: usize) -> Bands {
        let stasis = self.stasis[g];
        let (lo, hi) = self.level_limits(g);
        let level = self.level[g].clamp(lo, hi);
        Bands { stasis, rest_exit: stasis + REST_BUFFER * (level - stasis), level }
    }

    /// Range the level may take: a little above the noise floor, at most LEVEL_MAX.
    fn level_limits(&self, g: usize) -> (f64, f64) {
        let stasis = self.stasis[g];
        let room = (1.0 - stasis).max(0.05);
        let lo = (stasis * LEVEL_MIN_ROOM).min(stasis + 0.3 * room);
        (lo, LEVEL_MAX.max(lo))
    }

    /// Band a training trial asks for, around the level.
    fn trial_band(&self, g: usize) -> (f64, f64) {
        let b = self.bands(g);
        let lo = (TRIAL_LO * b.level).max(b.rest_exit);
        (lo, (TRIAL_HI * b.level).max(lo + 0.05))
    }

    /// Per channel: the gesture whose pattern leans on it most, if any does clearly.
    pub fn channel_gestures(&self) -> [Option<usize>; NUM_CHANNELS] {
        let seen = self.seen();
        std::array::from_fn(|c| {
            (0..G)
                .filter_map(|g| {
                    let t = self.templates[g].as_ref()?;
                    let share = t[c] / dot(t, t, &seen).sqrt().max(1e-9);
                    (share > 0.3).then_some((g, share))
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(g, _)| g)
        })
    }

    /// Per channel: activity from rest (0) to the strongest hard contraction
    /// seen there (1); None before the rest recording.
    pub fn channel_activation(&self) -> Option<[f32; NUM_CHANNELS]> {
        self.rest.as_ref()?;
        Some(std::array::from_fn(|c| {
            let top = self.data.iter().filter_map(|d| d.hard.map(|h| h[c])).fold(0.0, f64::max).max(RELAX_Z * 2.0);
            if self.contact[c] { (self.z_raw[c] / top).clamp(0.0, 1.0) as f32 } else { 0.0 }
        }))
    }

    /// Compound movements learned in live use, as (movement, movement, share of the first).
    pub fn compounds(&self) -> Vec<(usize, usize, f64)> {
        self.compounds.iter().map(|k| (k.pair.0, k.pair.1, k.mix)).collect()
    }

    /// Changes whenever a compound is learned or refined in live use.
    pub fn compound_updates(&self) -> u32 {
        self.compound_updates
    }

    /// Forgets the compound movements learned in live use.
    pub fn clear_compounds(&mut self) {
        self.compound_updates = self.compound_updates.wrapping_add(1);
        self.compounds.clear();
        self.episode = Episode::default();
        self.refit_due = false;
        self.update_stasis();
        self.update_coupling();
    }

    /// Channels that take part in the fit: seen at rest and in contact now.
    fn usable(&self) -> [bool; NUM_CHANNELS] {
        let seen = self.seen();
        std::array::from_fn(|c| seen[c] && self.contact[c])
    }

    fn seen(&self) -> [bool; NUM_CHANNELS] {
        self.rest.as_ref().map_or([false; NUM_CHANNELS], |r| r.seen)
    }

    /// Feed one sample of the filtered (not rectified) signal, before `process`.
    pub fn push_filtered(&mut self, x: &[f32; NUM_CHANNELS]) {
        self.cov.push(x);
    }

    /// Channels with a bad contact, found at the limp rest: left out of
    /// decoding (and, by the page, of the channel average).
    pub fn bad_channels(&self) -> [bool; NUM_CHANNELS] {
        self.rest.as_ref().map_or([false; NUM_CHANNELS], |r| outliers(&r.mean))
    }

    /// The classifier is fitted and in use.
    fn lda_on(&self) -> bool {
        self.use_lda && self.lda.is_some()
    }

    /// The classifier decodes this sample: it is on, and every channel it was
    /// trained on has contact (it knows nothing of a channel going missing;
    /// the patterns just fit around it).
    fn lda_live(&self, usable: &[bool; NUM_CHANNELS]) -> bool {
        self.lda_on() && *usable == self.seen()
    }

    fn featrec(&self, class: usize, kind: u8) -> Option<&FeatRec> {
        self.feats.iter().find(|r| r.class == class && r.kind == kind)
    }

    /// Keeps the covariance for sample `i` of a recording (every FEAT_EVERY-th).
    fn record_feats(&self, i: usize, out: &mut Vec<f32>) {
        if i % FEAT_EVERY != 0 {
            return;
        }
        if self.cov.filled() >= COV_WIN {
            out.extend(self.cov.window().iter().map(|&v| v as f32));
        } else {
            out.extend(std::iter::repeat(f32::NAN).take(NUM_COV));
        }
    }

    /// How sure the classifier is of each movement (0..1): its share against
    /// the best class, and against the best rest. All 1 without a classifier;
    /// all 0 while the window fills.
    fn gate(&self, feat: Option<&[f64; NUM_FEATS]>) -> [f64; G] {
        let Some(lda) = self.lda.as_ref().filter(|_| self.use_lda) else { return [1.0; G] };
        let Some(f) = feat else { return [0.0; G] };
        let mut out = [(0usize, 0.0f64); MAX_LDA_CLASSES];
        let n = lda.model.scores(&with_effort(*f, lda.rest_mlr), &mut out);
        let s = &out[..n];
        let s_max = s.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
        let s_rest = s.iter().filter(|x| x.0 >= G).map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
        let mut gate = [0.0; G];
        for &(c, sc) in s {
            if c < G {
                let vs_rest = if s_rest.is_finite() { 1.0 / (1.0 + (-(sc - s_rest) / LDA_TEMP).exp()) } else { 1.0 };
                gate[c] = ((sc - s_max) / LDA_TEMP).exp() * vs_rest;
            }
        }
        gate
    }

    /// Power of each movement in activation `x`. By the patterns alone (also
    /// when a channel the classifier was trained on has lost contact): their
    /// non-negative fit (with the compounds). With the classifier: each
    /// pattern's own projection, gated (if `gated`) by how sure the classifier
    /// is that it is that movement.
    fn decode(&self, x: &Features, feat: Option<&[f64; NUM_FEATS]>, usable: &[bool; NUM_CHANNELS], gated: bool) -> [f64; G] {
        if !self.lda_live(usable) {
            return self.fit_all(x, usable);
        }
        let gate = if gated { self.gate(feat) } else { [1.0; G] };
        std::array::from_fn(|g| {
            self.templates[g].as_ref().map_or(0.0, |t| (dot(t, x, usable) / dot(t, t, usable).max(1e-9)).max(0.0) * gate[g])
        })
    }

    /// Classifier features of a stored covariance, over the channels seen at rest.
    fn feat_at(&self, fr: Option<&FeatRec>, i: usize) -> Option<[f64; NUM_FEATS]> {
        let seen = self.seen();
        fr.and_then(|r| r.at(i)).map(|c| raw_features(&c, &seen))
    }

    /// How far a movement's power is from its noise floor to its level (1 = at the level).
    fn excess(&self, g: usize) -> f64 {
        let b = self.bands(g);
        (self.power[g] - b.stasis) / (b.level - b.stasis).max(1e-6)
    }

    /// What the faint round watches: with the classifier, the ungated power,
    /// since it was not trained on faint holds.
    fn faint_power(&self, g: usize) -> f64 {
        if self.lda_on() { self.singles[g] } else { self.power[g] }
    }

    /// Fits the classifier: every rest posture, and the hard and half-strength
    /// holds of every movement with a pattern. None until the limp rest and
    /// every such movement have recordings.
    fn fit_lda(&mut self) {
        self.lda = None;
        let seen = self.seen();
        let rest0: Vec<f64> =
            self.featrec(G, 0).into_iter().flat_map(|r| r.rows()).map(|c| raw_features(&c, &seen)[NUM_FEATS - 1]).collect();
        let moves: Vec<usize> = (0..G).filter(|&g| self.templates[g].is_some()).collect();
        if rest0.is_empty() || moves.is_empty() || !moves.iter().all(|&g| self.feats.iter().any(|r| r.class == g && r.kind < 3)) {
            return;
        }
        let rest_mlr = rest0.iter().sum::<f64>() / rest0.len() as f64;
        let samples: Vec<(usize, [f64; NUM_FEATS])> = self
            .feats
            .iter()
            .filter(|r| if r.class >= G { r.kind == 0 } else { self.templates[r.class].is_some() && (r.kind == 1 || r.kind == 2) })
            .flat_map(|r| r.rows().map(move |c| (r.class, with_effort(raw_features(&c, &seen), rest_mlr))))
            .collect();
        self.lda = LdaModel::fit(&samples, LDA_SHRINK).map(|model| Lda { model, rest_mlr });
    }

    /// `z` explained as a non-negative mix of `patterns` (at most NNLS_MAX) over the usable channels.
    fn fit_patterns(patterns: &[&Features], z: &Features, usable: &[bool; NUM_CHANNELS], ridge: f64) -> [f64; NNLS_MAX] {
        let n = patterns.len().min(NNLS_MAX);
        let mut g = [[0.0; NNLS_MAX]; NNLS_MAX];
        let mut h = [0.0; NNLS_MAX];
        for i in 0..n {
            for j in i..n {
                let v = dot(patterns[i], patterns[j], usable);
                g[i][j] = v;
                g[j][i] = v;
            }
            g[i][i] *= 1.0 + ridge;
            h[i] = dot(patterns[i], z, usable);
        }
        nnls(&g[..n], &h[..n])
    }

    /// The calibrated movements' patterns, in a fixed-size list, and which movement each is.
    fn single_patterns(templates: &[Option<Features>; G]) -> ([&Features; NNLS_MAX], [usize; G], usize) {
        let mut patterns = [&NO_PATTERN; NNLS_MAX];
        let mut idx = [0; G];
        let mut n = 0;
        for (g, t) in templates.iter().enumerate() {
            if let Some(t) = t.as_ref().filter(|_| n < NNLS_MAX.min(G)) {
                patterns[n] = t;
                idx[n] = g;
                n += 1;
            }
        }
        (patterns, idx, n)
    }

    /// Power of each calibrated gesture in an activation sample, from the
    /// single-movement patterns and the learned compounds: a compound's weight
    /// is shared between its two movements by its mix.
    fn fit_all(&self, x: &Features, usable: &[bool; NUM_CHANNELS]) -> [f64; G] {
        if self.compounds.is_empty() {
            return Self::fit(&self.templates, x, usable);
        }
        let (mut patterns, idx, singles) = Self::single_patterns(&self.templates);
        let compounds = &self.compounds[..self.compounds.len().min(NNLS_MAX - singles)];
        for (k, c) in compounds.iter().enumerate() {
            patterns[singles + k] = &c.pattern;
        }
        let w = Self::fit_patterns(&patterns[..singles + compounds.len()], x, usable, GESTURE_RIDGE);
        let mut out = [0.0; G];
        for k in 0..singles {
            out[idx[k]] = w[k];
        }
        for (k, c) in compounds.iter().enumerate() {
            let wk = w[singles + k];
            out[c.pair.0] += wk * c.mix;
            out[c.pair.1] += wk * (1.0 - c.mix);
        }
        out
    }

    /// Power of each calibrated gesture in an activation sample, from the single movements alone.
    fn fit(templates: &[Option<Features>; G], z: &Features, usable: &[bool; NUM_CHANNELS]) -> [f64; G] {
        let (patterns, idx, n) = Self::single_patterns(templates);
        let w = Self::fit_patterns(&patterns[..n], z, usable, GESTURE_RIDGE);
        let mut out = [0.0; G];
        for k in 0..n {
            out[idx[k]] = w[k];
        }
        out
    }

    /// How much of each raised posture is in a slowly averaged sample (raised
    /// posture p at index p − 1). The gestures' patterns are in the fit too, so
    /// a held gesture isn't taken for posture.
    fn fit_posture(&self, z_slow: &Features, usable: &[bool; NUM_CHANNELS]) -> [f64; POSTURES] {
        let mut out = [0.0; POSTURES];
        if self.postures.is_empty() {
            return out;
        }
        let mut patterns = [&NO_PATTERN; NNLS_MAX];
        let mut n = 0;
        for p in self.postures.iter().take(POSTURES).chain(self.templates.iter().flatten()).take(NNLS_MAX) {
            patterns[n] = p;
            n += 1;
        }
        let w = Self::fit_patterns(&patterns[..n], z_slow, usable, POSTURE_RIDGE);
        let k = self.postures.len().min(POSTURES);
        out[..k].copy_from_slice(&w[..k]);
        out
    }

    /// `z` with the posture baseline taken out.
    fn without_posture(&self, z: &Features, w: &[f64]) -> Features {
        std::array::from_fn(|c| z[c] - self.postures.iter().zip(w).map(|(p, w)| w * p[c]).sum::<f64>())
    }

    /// Share of each recorded posture (limp first) for posture weights `w`:
    /// the fitted posture signal placed on the path limp → forearm raised → arm
    /// up, shared between the two postures either side of it (see finger.rs).
    fn membership(&self, w: &[f64]) -> [f64; POSTURES] {
        let seen = self.seen();
        let signal: Features = std::array::from_fn(|c| self.postures.iter().zip(w).map(|(p, w)| w * p[c]).sum());
        let n = (self.postures.len() + 1).min(POSTURES);
        let point = |i: usize| if i == 0 { [0.0; NUM_CHANNELS] } else { self.postures[i - 1] };
        let mut m = [0.0; POSTURES];
        if n == 1 {
            m[0] = 1.0;
            return m;
        }
        let (mut best, mut best_seg, mut best_t) = (f64::INFINITY, 0, 0.0);
        for i in 0..n - 1 {
            let (a, b) = (point(i), point(i + 1));
            let d: Features = std::array::from_fn(|c| b[c] - a[c]);
            let rel: Features = std::array::from_fn(|c| signal[c] - a[c]);
            let t = (dot(&rel, &d, &seen) / dot(&d, &d, &seen).max(1e-9)).clamp(0.0, 1.0);
            let off: Features = std::array::from_fn(|c| rel[c] - t * d[c]);
            let dist = dot(&off, &off, &seen);
            if dist < best {
                (best, best_seg, best_t) = (dist, i, t);
            }
        }
        m[best_seg] = 1.0 - best_t;
        m[best_seg + 1] = best_t;
        m
    }

    /// Each gesture's noise floor for the current posture.
    fn blend_stasis(&self) -> [f64; G] {
        if self.stasis_by.is_empty() {
            return self.stasis;
        }
        let m = self.membership(&self.posture_w);
        std::array::from_fn(|g| self.stasis_by.iter().zip(&m).map(|(s, m)| m * s[g]).sum())
    }

    /// Current posture: the share of each recorded posture (limp first).
    pub fn posture(&self) -> Vec<f64> {
        if self.posture_rest.is_empty() {
            return Vec::new();
        }
        let n = (self.postures.len() + 1).min(POSTURES);
        self.membership(&self.posture_w)[..n].to_vec()
    }

    /// Gesture recordings keep the posture they started in, so a strong
    /// contraction can't be mistaken for the arm being raised.
    fn posture_frozen(&self) -> bool {
        self.run.as_ref().is_some_and(|r| {
            r.phase == Phase::Record && matches!(r.step, Step::Hard(_) | Step::Medium(_) | Step::Faint(_))
        })
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
            let z_raw: Features =
                std::array::from_fn(|c| if contact[c] { (env[c] as f64 - rest.mean[c]) / rest.scale[c] } else { 0.0 });
            let usable = self.usable();
            self.feat = (self.cov.filled() >= COV_WIN).then(|| raw_features(&self.cov.window(), &usable));
            let s = 1.0 - (-1.0 / (POSTURE_TAU_S * SAMPLE_RATE) as f64).exp();
            for (slow, &z) in self.z_slow.iter_mut().zip(&z_raw) {
                *slow += (z - *slow) * s;
            }
            if !self.posture_frozen() && self.tick % POSTURE_EVERY == 0 {
                let w = self.fit_posture(&self.z_slow, &usable);
                let n = self.posture_w.len().min(POSTURES);
                self.posture_w[..n].copy_from_slice(&w[..n]);
            }
            self.tick = self.tick.wrapping_add(1);
            self.z_raw = z_raw;
            let lda = self.lda_live(&usable);
            // the classifier knows the raised postures as rests of their own;
            // the patterns alone need the posture taken out first
            self.z = if lda { z_raw } else { self.without_posture(&z_raw, &self.posture_w) };
            self.x = self.z;
            self.stasis = self.blend_stasis();
            let feat = self.feat;
            let raw = self.decode(&self.x, feat.as_ref(), &usable, true);
            let singles = if lda {
                self.decode(&self.x, feat.as_ref(), &usable, false)
            } else {
                Self::fit(&self.templates, &self.x, &usable)
            };
            let alpha = 1.0 - (-1.0 / (POWER_TAU_S * SAMPLE_RATE) as f64).exp();
            for (p, r) in self.power.iter_mut().zip(raw) {
                *p += (r - *p) * alpha;
            }
            for (p, r) in self.singles.iter_mut().zip(singles) {
                *p += (r - *p) * alpha;
            }
            self.update_compound();
            let decodable = self.decodable();
            for g in 0..G {
                if decodable[g] {
                    self.update_active(g);
                } else {
                    self.active[g] = false;
                    self.rising_t[g] = 0.0;
                }
            }
        }
        self.learn_compounds();
        if self.refit_due && !self.active.iter().any(|&a| a) {
            self.refit_due = false;
            self.update_stasis();
            self.update_coupling();
        }
        self.calibration_step();
        self.update_inputs();
    }

    /// Live use only: while a pair is decoded together (both driving), gathers
    /// its activation per unit of what the single-movement fit gives the two;
    /// when the pair ends, the episode becomes or refines its compound pattern.
    fn learn_compounds(&mut self) {
        if self.run.is_some() || self.rest.is_none() || !self.learn_in_use || self.lda_on() {
            self.episode = Episode::default();
            return;
        }
        let pair = self
            .compound
            .filter(|&(a, b)| self.active[a] && self.active[b])
            .map(|(a, b)| (a.min(b), a.max(b)));
        if pair != self.episode.pair {
            self.close_episode();
            self.episode = Episode { pair, ..Default::default() };
        }
        let Some((a, b)) = pair else { return };
        let lost = self.usable() != self.seen();
        let (pa, pb) = (self.singles[a].max(0.0), self.singles[b].max(0.0));
        let ep = &mut self.episode;
        ep.lost |= lost;
        if pa + pb < 1e-6 {
            return;
        }
        // sums over the episode, divided at its end: a ratio of sums, so the lag
        // of the smoothed powers behind the activation averages out
        for (s, x) in ep.sum.iter_mut().zip(&self.x) {
            *s += x;
        }
        ep.mix += pa;
        ep.total += pa + pb;
        ep.n += 1;
    }

    fn close_episode(&mut self) {
        let ep = std::mem::take(&mut self.episode);
        let Some((a, b)) = ep.pair else { return };
        if ep.n < samples(LEARN_MIN_S) || ep.lost || ep.total < 1e-9 || self.axes[a].0 == self.axes[b].0 {
            return;
        }
        let (Some(ta), Some(tb)) = (self.templates[a], self.templates[b]) else { return };
        let seen = self.seen();
        let pattern: Features = ep.sum.map(|s| s / ep.total);
        let mix = ep.mix / ep.total;
        if let Some(k) = self.compounds.iter_mut().find(|k| k.pair == (a, b)) {
            k.n = (k.n + 1.0).min(LEARN_MEMORY);
            for (p, &v) in k.pattern.iter_mut().zip(&pattern) {
                *p += (v - *p) / k.n;
            }
            k.mix += (mix - k.mix) / k.n;
        } else {
            // the fit takes at most NNLS_MAX patterns, the single movements first
            if self.compounds.len() + G >= NNLS_MAX {
                return;
            }
            // only worth a pattern of its own if the two don't simply add up to it
            let w = Self::fit_patterns(&[&ta, &tb], &pattern, &seen, 0.0);
            let fitted: Features = std::array::from_fn(|c| w[0] * ta[c] + w[1] * tb[c]);
            let off: Features = std::array::from_fn(|c| pattern[c] - fitted[c]);
            let residual = (dot(&off, &off, &seen) / dot(&pattern, &pattern, &seen).max(1e-12)).sqrt();
            let like_one = cosine(&pattern, &ta, &seen).max(cosine(&pattern, &tb, &seen)) > COMPOUND_SINGLE_COS;
            if residual < COMPOUND_RESIDUAL || like_one {
                return;
            }
            self.compounds.push(Compound { pair: (a, b), pattern, mix, n: 1.0 });
        }
        self.compound_updates = self.compound_updates.wrapping_add(1);
        self.refit_due = true;
    }

    /// How far `x` reads above what `y`'s crosstalk predicts for it, in standard deviations.
    fn stands_out(&self, x: usize, y: usize) -> f64 {
        let (k, sd) = self.coupling[y][x];
        (self.power[x] - k * self.power[y]) / sd
    }

    /// Decides whether two gestures are being made together. A pair holds while
    /// they are the two strongest and the weaker still stands out from the
    /// stronger one's crosstalk by COMPOUND_EXIT_SIGMAS; a new pair needs the
    /// runner-up at COMPOUND_SIGMAS for COMPOUND_COMMIT_S.
    fn update_compound(&mut self) {
        let mut order: [usize; G] = std::array::from_fn(|g| g);
        order.sort_by(|&a, &b| self.power[b].total_cmp(&self.power[a]));
        let calibrated = |g: &usize| self.templates[*g].is_some();
        // flexing and extending the same joint at once is no compound: the weaker one is crosstalk
        let best = order.iter().copied().find(calibrated);
        let runner = best.and_then(|b| order.iter().copied().filter(calibrated).find(|&g| self.axes[g].0 != self.axes[b].0));
        let (Some(best), Some(runner)) = (best, runner) else {
            self.compound = None;
            self.compound_t = 0.0;
            return;
        };
        if let Some((a, b)) = self.compound {
            let top_two = (a == best && b == runner) || (a == runner && b == best);
            if top_two && self.stands_out(runner, best) >= COMPOUND_EXIT_SIGMAS {
                return;
            }
            self.compound = None;
            self.compound_t = 0.0;
        }
        if self.stands_out(runner, best) >= COMPOUND_SIGMAS {
            self.compound_t += 1.0 / SAMPLE_RATE;
            if self.compound_t >= COMPOUND_COMMIT_S {
                self.compound = Some((best, runner));
            }
        } else {
            self.compound_t = 0.0;
        }
    }

    /// The gestures allowed to drive this sample: the strongest, and its
    /// compound partner if there is one.
    fn decodable(&self) -> [bool; G] {
        let mut ok = [false; G];
        let best = (0..G)
            .filter(|&g| self.templates[g].is_some())
            .max_by(|&a, &b| self.excess(a).total_cmp(&self.excess(b)));
        if let Some(best) = best {
            ok[best] = true;
        }
        if let Some((a, b)) = self.compound {
            ok[a] = true;
            ok[b] = true;
        }
        ok
    }

    /// How much each gesture reads when only another one is made: for each
    /// ordered pair (a, b), b's power against a's over all of a's recordings
    /// (hard, half-strength, faint), as a slope through the origin and the
    /// scatter around it. The powers are smoothed the way the live decoder
    /// smooths them, so the scatter is the one the compound test really sees.
    fn update_coupling(&mut self) {
        let seen = self.seen();
        let alpha = 1.0 - (-1.0 / (POWER_TAU_S * SAMPLE_RATE) as f64).exp();
        let smoothed = |rec: &[Features], fr: Option<&FeatRec>| -> Vec<[f64; G]> {
            let mut p = [0.0; G];
            rec.iter()
                .enumerate()
                .map(|(i, z)| {
                    let w = self.decode(z, self.feat_at(fr, i).as_ref(), &seen, true);
                    for (pg, wg) in p.iter_mut().zip(w) {
                        *pg = if i == 0 { wg } else { *pg + (wg - *pg) * alpha };
                    }
                    p
                })
                .collect()
        };
        let mut c = [[(0.0, COUPLING_MIN_SD); G]; G];
        for a in (0..G).filter(|&a| self.templates[a].is_some()) {
            let d = &self.data[a];
            let fits: Vec<[f64; G]> =
                [(&d.hard_rec, 1), (&d.medium, 2), (&d.faint, 3)]
                    .into_iter()
                    .flat_map(|(rec, k)| smoothed(rec, self.featrec(a, k)))
                    .collect();
            let saa: f64 = fits.iter().map(|w| w[a] * w[a]).sum();
            if fits.len() < 2 || saa < 1e-9 {
                continue;
            }
            for b in (0..G).filter(|&b| b != a) {
                let k = (fits.iter().map(|w| w[a] * w[b]).sum::<f64>() / saa).max(0.0);
                let var = fits.iter().map(|w| (w[b] - k * w[a]).powi(2)).sum::<f64>() / (fits.len() - 1) as f64;
                c[a][b] = (k, var.sqrt().max(COUPLING_MIN_SD));
            }
        }
        self.coupling = c;
    }

    fn update_active(&mut self, g: usize) {
        if self.templates[g].is_none() {
            self.active[g] = false;
            return;
        }
        let p = self.power[g];
        let b = self.bands(g);
        if self.active[g] {
            self.active[g] = p >= b.stasis;
            return;
        }
        self.rising_t[g] = if p > b.rest_exit { self.rising_t[g] + 1.0 / SAMPLE_RATE } else { 0.0 };
        if self.rising_t[g] >= COMMIT_S {
            self.active[g] = true;
            self.rising_t[g] = 0.0;
        }
    }

    /// Speed (0..1, before gain) of an active gesture: from 0 at the buffer to
    /// full at the level.
    fn speed(&self, g: usize) -> f32 {
        let b = self.bands(g);
        ((self.power[g] - b.rest_exit) / (b.level - b.rest_exit).max(1e-6)).clamp(0.0, 1.0) as f32
    }

    fn update_inputs(&mut self) {
        self.inputs = [0.0; NUM_JOINTS];
        let step = self.run.as_ref().map(|r| r.step);
        let moves: [bool; G] = std::array::from_fn(|g| match step {
            None => self.active[g],
            // training: only the gesture being trained moves the arm
            Some(Step::Train(t)) => g == t && self.active[g],
            Some(_) => false,
        });
        for g in (0..G).filter(|&g| moves[g]) {
            let (axis, dir) = self.axes[g];
            self.inputs[axis] = (self.inputs[axis] + dir * self.speed(g) * self.gain).clamp(-1.0, 1.0);
        }
    }

    // ---- runs ---------------------------------------------------------------------

    /// Phase 1 for every gesture (rest in every posture first), then phase 2 if `train_after`.
    pub fn start_calibration(&mut self, train_after: bool) {
        self.start(Step::Rest(0), (0..G).collect(), Vec::new(), train_after);
    }

    /// Phase 1 for one gesture, keeping the rest model (a full run if there is none).
    pub fn start_gesture(&mut self, gesture: usize, train_after: bool) {
        if gesture >= G {
            return;
        }
        if self.rest.is_none() {
            self.start_calibration(train_after);
            return;
        }
        self.start(Step::Hard(gesture), vec![gesture], Vec::new(), train_after);
    }

    /// Phase 2 only: every calibrated gesture.
    pub fn start_training(&mut self) {
        let gestures: Vec<usize> = (0..G).filter(|&g| self.templates[g].is_some()).collect();
        if let Some(&first) = gestures.first() {
            self.start(Step::Train(first), Vec::new(), gestures, false);
        }
    }

    fn start(&mut self, step: Step, calibrate: Vec<usize>, train: Vec<usize>, train_after: bool) {
        self.run = Some(Run {
            step,
            phase: Phase::Prepare,
            ready: false,
            t: 0,
            good: 0,
            stuck: 0,
            rec: Vec::new(),
            feats: Vec::new(),
            holds: 0,
            next: None,
            rests: if step == Step::Rest(0) { POSTURES } else { 0 },
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

    /// The user is in position for the current step: start its countdown.
    pub fn ready(&mut self) {
        if let Some(run) = &mut self.run {
            run.ready = true;
        }
    }

    /// A phase-1 step held before its countdown, until `ready`.
    fn waiting(run: &Run) -> bool {
        run.phase == Phase::Prepare && !run.ready && !matches!(run.step, Step::Train(_))
    }

    #[cfg(test)]
    fn running(&self) -> bool {
        self.run.is_some()
    }

    /// Skip the current step. A raised posture: left out. Hard or half-strength:
    /// drops that gesture from the rest of the calibration. Faint: that gesture
    /// keeps what it learned from the half-strength round. Training: on to the
    /// next gesture. The limp rest can't be skipped, since everything is
    /// measured against it.
    pub fn skip(&mut self) {
        let Some(mut run) = self.run.take() else { return };
        let step = run.step;
        let next = match (step, run.phase) {
            (Step::Rest(0), _) => {
                self.run = Some(run);
                return;
            }
            (Step::Rest(p), _) => Self::after_rest(&run, p),
            // already recorded: just skip the relax
            (Step::Hard(_) | Step::Medium(_) | Step::Faint(_), Phase::Relax) => run.next.take(),
            (Step::Hard(g) | Step::Medium(g), _) => {
                self.data[g] = GestureData::default();
                self.templates[g] = None;
                self.feats.retain(|r| r.class != g);
                self.fit_lda();
                run.holds = 0;
                self.next_calibration_step(&mut run, step)
            }
            (Step::Faint(_), _) => self.next_calibration_step(&mut run, step),
            (Step::Train(g), _) => Self::after_training(&run, g),
        };
        if self.enter(&mut run, next) {
            self.run = Some(run);
        }
    }

    fn duration(run: &Run) -> u32 {
        match (run.step, run.phase) {
            (Step::Train(_), Phase::Prepare) => samples(CUE_S),
            (_, Phase::Prepare) => samples(COUNTDOWN_S),
            (Step::Rest(_), Phase::Record) => samples(REST_S),
            (Step::Hard(_) | Step::Medium(_), Phase::Record) => samples(GESTURE_S) / HOLDS,
            (Step::Faint(_), Phase::Record) => samples(GESTURE_S),
            (Step::Train(_), Phase::Record) => samples(TRIAL_TIMEOUT_S),
            (Step::Train(_), Phase::Relax) => samples(TRAIN_RELAX_S),
            (_, Phase::Relax) => samples(RELAX_S),
        }
    }

    /// Half-strength: projection onto the hard pattern (1 = as hard as the hard one).
    fn of_hard(&self, g: usize) -> f64 {
        let usable = self.usable();
        self.data[g].hard.map_or(0.0, |h| (dot(&h, &self.z, &usable) / dot(&h, &h, &usable).max(1e-9)).max(0.0))
    }

    /// Another gesture reading stronger than `g` (and above its own noise): the
    /// strongest such one, i.e. the gesture actually being made instead.
    fn stronger_gesture(&self, g: usize) -> Option<usize> {
        let p = self.power[g];
        (0..G)
            .filter(|&h| h != g && self.templates[h].is_some() && self.power[h] > self.stasis[h].max(p))
            .max_by(|&a, &b| self.power[a].total_cmp(&self.power[b]))
    }

    /// The channel furthest above its recent level, against its stillness
    /// limit (capped at REST_CEILING_UV if `ceiling`: the limp rest only, since
    /// holding the arm up is a steady contraction of its own).
    fn stillness_recent(&self, ceiling: bool) -> (usize, f64) {
        (0..NUM_CHANNELS)
            .filter(|&c| self.contact[c])
            .map(|c| {
                let limit = self.recent[c] * STILL_RATIO + STILL_MARGIN_UV;
                let limit = if ceiling { limit.min(REST_CEILING_UV) } else { limit };
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

    /// Box for the faint round: from just above noise up to FAINT_TOP.
    fn faint_box(&self, g: usize) -> (f64, f64) {
        let lo = FAINT_MIN * self.stasis[g];
        (lo, FAINT_TOP.max(lo * 1.6).min(1.0))
    }

    /// Meter kind, value, target range and aim for the coach.
    fn meter(&self, run: &Run) -> (Meter, f64, (f64, f64), f64) {
        let none = (f64::NAN, f64::NAN);
        match (run.step, run.phase) {
            (Step::Rest(p), _) => (Meter::Stillness, self.stillness_recent(p == 0).1, none, f64::NAN),
            (Step::Hard(_), Phase::Record) => {
                let a = Self::activity(&self.z, &self.usable()) / HARD_MIN_ACTIVITY;
                (Meter::Activity, a, (1.0, f64::INFINITY), f64::NAN)
            }
            (Step::Medium(g), Phase::Record) => (Meter::OfHard, self.of_hard(g), MEDIUM_TARGET, f64::NAN),
            (Step::Faint(g), Phase::Record) => (Meter::Power, self.faint_power(g), self.faint_box(g), f64::NAN),
            (Step::Train(g), Phase::Relax) => (Meter::Power, self.power[g], none, f64::NAN),
            (Step::Train(g), _) => (Meter::Power, self.power[g], self.trial_band(g), self.bands(g).level),
            _ => (Meter::Stillness, self.stillness_z().1, none, f64::NAN),
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
            (Step::Rest(p), _) => match self.stillness_recent(p == 0) {
                (c, r) if r > 1.0 => Check::Moving(c),
                _ => Check::Clean,
            },
            (Step::Hard(_), Phase::Record) => {
                if Self::activity(&self.z, &usable) >= HARD_MIN_ACTIVITY { Check::Clean } else { Check::TooWeak }
            }
            (Step::Medium(g), Phase::Record) => {
                let h = self.of_hard(g);
                let active = Self::activity(&self.z, &usable) >= MEDIUM_MIN_ACTIVITY;
                if !active || h < MEDIUM_MIN {
                    Check::TooWeak
                } else if h > MEDIUM_MAX {
                    Check::TooStrong
                } else {
                    Check::Clean
                }
            }
            (Step::Faint(g), Phase::Record) => {
                let p = self.faint_power(g);
                let (lo, hi) = self.faint_box(g);
                if p > hi {
                    Check::Fainter
                } else if p < lo {
                    Check::TooWeak
                } else {
                    Check::Clean
                }
            }
            (Step::Train(g), Phase::Record) => {
                let p = self.power[g];
                let (lo, hi) = self.trial_band(g);
                if let Some(other) = self.stronger_gesture(g) {
                    Check::OtherGesture(other)
                } else if p > hi {
                    Check::TooStrong
                } else if p < lo {
                    Check::TooWeak
                } else {
                    Check::Clean
                }
            }
            (Step::Train(g), Phase::Relax) => {
                if self.power[g] < self.stasis[g] { Check::Clean } else { Check::NotRelaxed(g) }
            }
            (_, Phase::Relax) => match self.stillness_z() {
                (c, r) if r > 1.0 => Check::NotRelaxed(c),
                _ => Check::Clean,
            },
        })
    }

    fn calibration_step(&mut self) {
        let Some(mut run) = self.run.take() else { return };
        if Self::waiting(&run) {
            self.run = Some(run);
            return;
        }
        let clean = self.signal_check(&run) == Some(Check::Clean);
        run.t += 1;
        let done = match (run.phase, run.step) {
            (Phase::Prepare, _) => run.t >= Self::duration(&run),
            (Phase::Record, Step::Train(g)) => {
                run.hold = if clean { run.hold + 1 } else { 0 };
                let hit = run.hold >= samples(TRIAL_HOLD_S);
                if hit || run.t >= Self::duration(&run) {
                    self.trial_result(&mut run, g, hit);
                    true
                } else {
                    false
                }
            }
            (Phase::Record, step) => {
                if clean && run.t > samples(SETTLE_S) {
                    run.rec.push(match step {
                        Step::Rest(0) => self.envelope.map(f64::from),
                        Step::Rest(_) => self.z_raw,
                        _ => self.z,
                    });
                    self.record_feats(run.rec.len() - 1, &mut run.feats);
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
            (Phase::Record, Step::Rest(p)) => {
                let next = self.finish_rest(&mut run, p);
                self.enter(&mut run, next)
            }
            (Phase::Record, _) => {
                match run.step {
                    Step::Train(_) => {}
                    // a hard or half-strength recording is HOLDS holds, with a relax between
                    Step::Hard(_) | Step::Medium(_) if run.holds + 1 < HOLDS => {
                        run.holds += 1;
                        run.next = Some(run.step);
                    }
                    _ => {
                        run.holds = 0;
                        run.next = self.finish_recording(&mut run);
                    }
                }
                run.phase = Phase::Relax;
                true
            }
            (Phase::Relax, Step::Train(g)) => {
                if run.trial + 1 < TRIALS {
                    run.trial += 1;
                    run.phase = Phase::Prepare;
                    true
                } else {
                    let next = Self::after_training(&run, g);
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
        if run.holds == 0 {
            run.rec.clear();
            run.feats.clear();
        }
        if keep {
            self.run = Some(run);
        }
    }

    /// Moves the run to `step`'s prepare phase; false when there is none (run over).
    fn enter(&mut self, run: &mut Run, step: Option<Step>) -> bool {
        let Some(step) = step else { return false };
        if let Step::Train(_) = step {
            if run.step.gesture() != step.gesture() || !matches!(run.step, Step::Train(_)) {
                run.trial = 0;
                run.streak = 0;
            }
        }
        // the next hold of the same recording: straight into its countdown, keeping what was recorded
        let again = run.holds > 0 && step == run.step;
        if !again {
            run.holds = 0;
            run.rec.clear();
            run.feats.clear();
        }
        run.step = step;
        run.phase = Phase::Prepare;
        run.ready = again;
        run.t = 0;
        run.good = 0;
        run.stuck = 0;
        run.hold = 0;
        true
    }

    /// A rest recording is done. Limp (p = 0): the reference everything else is
    /// measured against. A raised posture: its pattern and noise.
    fn finish_rest(&mut self, run: &mut Run, p: usize) -> Option<Step> {
        let rec = std::mem::take(&mut run.rec);
        let feats = std::mem::take(&mut run.feats);
        if p == 0 {
            let before = self.bad_channels();
            let n = rec.len().max(1);
            let mean = mean(&rec);
            let scale: Features = std::array::from_fn(|c| {
                let var = rec.iter().map(|x| (x[c] - mean[c]).powi(2)).sum::<f64>() / n as f64;
                var.sqrt().max(MIN_SCALE_UV)
            });
            // contact check: a channel far noisier than the others at rest is left out
            let bad = outliers(&mean);
            let seen: [bool; NUM_CHANNELS] = std::array::from_fn(|c| self.contact[c] && mean[c] > 0.0 && !bad[c]);
            let z = rec.iter().map(|x| std::array::from_fn(|c| (x[c] - mean[c]) / scale[c])).collect();
            self.rest = Some(RestModel { mean, scale, seen });
            // every pattern was measured against the old rest: start over
            self.postures.clear();
            self.posture_rest = vec![z];
            self.posture_w.clear();
            self.z_slow = [0.0; NUM_CHANNELS];
            self.templates = [None; G];
            self.data = std::array::from_fn(|_| GestureData::default());
            self.level = [LEVEL_MAX; G];
            self.report = [GestureReport::default(); G];
            self.compounds.clear();
            self.feats = vec![FeatRec { class: G, kind: 0, data: feats }];
            self.lda = None;
            if seen.iter().filter(|&&s| s).count() < MIN_WORKING {
                self.update_stasis();
                return None;
            }
            // with the channel average taken out, a bad contact spoiled every
            // channel's rest: record it again now that it is left out
            if self.car && (0..NUM_CHANNELS).any(|c| bad[c] && !before[c]) {
                self.update_stasis();
                return Some(Step::Rest(0));
            }
        } else {
            self.postures.push(mean(&rec));
            self.posture_rest.push(rec);
            self.posture_w.push(0.0);
            self.feats.push(FeatRec { class: G + self.posture_rest.len() - 1, kind: 0, data: feats });
        }
        self.fit_lda();
        self.update_stasis();
        Self::after_rest(run, p)
    }

    /// The step after rest posture `p`: the next posture, then the first gesture.
    fn after_rest(run: &Run, p: usize) -> Option<Step> {
        if p + 1 < run.rests {
            Some(Step::Rest(p + 1))
        } else {
            run.calibrate.first().map(|&g| Step::Hard(g))
        }
    }

    /// A gesture recording is done: learn from it and decide the next step.
    fn finish_recording(&mut self, run: &mut Run) -> Option<Step> {
        let rec = std::mem::take(&mut run.rec);
        let feats = std::mem::take(&mut run.feats);
        let (Step::Hard(g) | Step::Medium(g) | Step::Faint(g)) = run.step else { return None };
        let kind = kind_of(run.step);
        // patterns learned in live use were fitted against the old patterns
        self.compounds.clear();
        // a new hard recording starts the movement over
        self.feats.retain(|r| r.class != g || (kind != 1 && r.kind != kind));
        self.feats.push(FeatRec { class: g, kind, data: feats });
        // every recording re-learns every movement's pattern,
        // so a movement recorded early isn't stuck with the patterns there were then
        match run.step {
            Step::Hard(g) => {
                self.data[g] = GestureData { hard: Some(mean(&rec)), hard_rec: rec, ..Default::default() };
                self.relearn();
            }
            Step::Medium(g) => {
                self.data[g].medium = rec;
                self.relearn();
                if self.templates[g].is_none() {
                    self.data[g] = GestureData::default(); // no usable pattern: out of the faint round
                    self.feats.retain(|r| r.class != g);
                    self.relearn();
                }
            }
            Step::Faint(g) => {
                self.data[g].faint = rec;
                self.relearn();
            }
            Step::Rest(_) | Step::Train(_) => return None,
        }
        let step = run.step;
        self.next_calibration_step(run, step)
    }

    /// Phase 1 runs round by round: every gesture hard, then every gesture at
    /// half strength, then every gesture faint. The step after `after` in that
    /// order, or the end of phase 1 (then training, if asked for).
    fn next_calibration_step(&mut self, run: &mut Run, after: Step) -> Option<Step> {
        let pos = |g: usize| run.calibrate.iter().position(|&h| h == g).map_or(0, |p| p + 1);
        let (mut round, mut i) = match after {
            Step::Hard(g) => (0, pos(g)),
            Step::Medium(g) => (1, pos(g)),
            Step::Faint(g) => (2, pos(g)),
            Step::Rest(_) | Step::Train(_) => (0, 0),
        };
        while round < 3 {
            while let Some(&g) = run.calibrate.get(i) {
                let d = &self.data[g];
                let step = match round {
                    0 => Some(Step::Hard(g)),
                    1 => d.hard.is_some().then_some(Step::Medium(g)),
                    _ => (self.templates[g].is_some() && !d.medium.is_empty()).then_some(Step::Faint(g)),
                };
                if step.is_some() {
                    return step;
                }
                i += 1;
            }
            round += 1;
            i = 0;
        }
        self.finalize(&run.calibrate);
        if !run.train_after {
            return None;
        }
        run.train = run.calibrate.iter().copied().filter(|&g| self.templates[g].is_some()).collect();
        run.train.first().map(|&g| Step::Train(g))
    }

    /// Re-learns every movement's pattern from its recordings (each one from
    /// its own alone), then the classifier and the noise floors.
    fn relearn(&mut self) {
        for g in 0..G {
            let (t, mixed) = self.learn_template(g);
            self.templates[g] = t;
            self.data[g].mixed = mixed;
            self.data[g].shapes.clear();
        }
        self.fit_lda();
        self.update_stasis();
    }

    /// Whether sample `x` is a hit for the movement whose pattern points along `dir`.
    fn is_hit(dir: &Features, x: &Features, seen: &[bool; NUM_CHANNELS]) -> bool {
        cosine(x, dir, seen) >= HIT_COS
    }

    /// Movement `g`'s pattern, and whether its faint round was mostly misses.
    /// The half-strength round seeds the direction; then each round's hits
    /// (by their own pattern alone) are averaged, every round counting the
    /// same, so the hard contraction anchors it and stray moments don't pull it. Scaled so that the softest
    /// part of the half-strength hits reads 1. Before there is a half-strength
    /// round: a provisional pattern from the hard one, so the next round can
    /// already tell this movement from the others (scaled so a hard one reads ~2).
    fn learn_template(&self, g: usize) -> (Option<Features>, bool) {
        let seen = self.seen();
        let d = &self.data[g];
        let (hard, medium, faint) = (&d.hard_rec, &d.medium, &d.faint);
        let scaled = |dir: Features, rec: &[Features], p: f64, div: f64| {
            let a = percentile(rec.iter().map(|x| dot(&dir, x, &seen)).collect(), p) / div;
            (a > 1e-6).then(|| dir.map(|v| v * a))
        };
        if medium.is_empty() {
            return (direction(&mean(hard), &seen).and_then(|dir| scaled(dir, hard, CEILING_PERCENTILE, 2.0)), false);
        }
        if hard.is_empty() {
            return (None, false);
        }
        let Some(mut dir) = direction(&mean(medium), &seen) else { return (None, false) };
        let hits = |dir: &Features, rec: &[Features]| -> Vec<Features> {
            rec.iter().copied().filter(|x| Self::is_hit(dir, x, &seen)).collect()
        };
        for _ in 0..2 {
            let shapes: Vec<Features> = [hard, medium, faint]
                .into_iter()
                .filter_map(|rec| {
                    let h = hits(&dir, rec);
                    let enough = !rec.is_empty() && h.len() as f64 >= ROUND_MIN_HITS * rec.len() as f64;
                    enough.then(|| direction(&mean(&h), &seen)).flatten()
                })
                .collect();
            match direction(&mean(&shapes), &seen) {
                Some(next) => dir = next,
                None => break,
            }
        }
        let mixed = !faint.is_empty() && (hits(&dir, faint).len() as f64) < 0.5 * faint.len() as f64;
        let medium_hits = hits(&dir, medium);
        let base = if medium_hits.len() as f64 >= ROUND_MIN_HITS * medium.len() as f64 { &medium_hits } else { medium };
        (scaled(dir, base, CEILING_PERCENTILE, 1.0), mixed)
    }

    /// Gesture powers over a recorded stretch of raw z, worked out the way the
    /// live decoder does: slow posture fit, posture taken out, gestures
    /// fitted. Not smoothed, unlike the live powers: the noise floors come from
    /// these, and the unsmoothed spread is the margin that keeps a rest held
    /// longer than its recording (the arm up, above all) under them.
    fn replay(&self, seq: &[Features], fr: Option<&FeatRec>) -> Vec<[f64; G]> {
        let seen = self.seen();
        let lda = self.lda_on();
        let s = 1.0 - (-1.0 / (POSTURE_TAU_S * SAMPLE_RATE) as f64).exp();
        let mut slow = seq.first().copied().unwrap_or([0.0; NUM_CHANNELS]);
        let mut w = [0.0; POSTURES];
        seq.iter()
            .enumerate()
            .map(|(i, z)| {
                let z = if lda {
                    *z
                } else {
                    for (sl, v) in slow.iter_mut().zip(z) {
                        *sl += (v - *sl) * s;
                    }
                    if i % POSTURE_EVERY as usize == 0 {
                        w = self.fit_posture(&slow, &seen);
                    }
                    self.without_posture(z, &w)
                };
                self.decode(&z, self.feat_at(fr, i).as_ref(), &seen, true)
            })
            .collect()
    }

    /// Each gesture's noise floor in each recorded posture: above its resting
    /// power there, with every template in the fit.
    fn update_stasis(&mut self) {
        self.stasis_by = self
            .posture_rest
            .iter()
            .enumerate()
            .map(|(p, seq)| {
                let fits = self.replay(seq, self.featrec(G + p, 0));
                std::array::from_fn(|g| {
                    let p = percentile(fits.iter().map(|w| w[g]).collect(), STASIS_PERCENTILE);
                    (p + STASIS_MARGIN).max(STASIS_MIN)
                })
            })
            .collect();
        self.stasis = self.blend_stasis();
    }

    fn after_training(run: &Run, g: usize) -> Option<Step> {
        let pos = run.train.iter().position(|&h| h == g)?;
        run.train.get(pos + 1).map(|&h| Step::Train(h))
    }

    /// End of phase 1: each calibrated gesture's faint level (its starting
    /// level) and isolation, measured with all the templates competing.
    fn finalize(&mut self, gestures: &[usize]) {
        self.update_stasis();
        self.update_coupling();
        let seen = self.seen();
        for &g in gestures {
            if self.templates[g].is_none() {
                continue;
            }
            let fr = self.featrec(g, 3);
            let fit = |gated: bool| -> Vec<[f64; G]> {
                self.data[g]
                    .faint
                    .iter()
                    .enumerate()
                    .map(|(i, z)| self.decode(z, self.feat_at(fr, i).as_ref(), &seen, gated))
                    .collect()
            };
            // the level from the patterns' own reading (the classifier never saw a
            // faint hold); isolation as it decodes
            let levels = fit(false);
            let fits = fit(true);
            let faint = if levels.is_empty() { LEVEL_MAX } else { percentile(levels.iter().map(|w| w[g]).collect(), 0.5) };
            let on_top = fits
                .iter()
                .filter(|w| w[g] > self.stasis[g] && (0..G).all(|h| w[h] <= w[g]))
                .count();
            let (lo, hi) = self.level_limits(g);
            self.level[g] = faint.clamp(lo, hi);
            self.report[g] = GestureReport { faint, isolation: on_top as f64 / fits.len().max(1) as f64, ..Default::default() };
        }
    }

    /// A training trial ended: tally it and step the level. Two hits in a row
    /// lower it (lighter intention will do), a miss raises it.
    fn trial_result(&mut self, run: &mut Run, g: usize, hit: bool) {
        self.report[g].trials += 1;
        let (lo, hi) = self.level_limits(g);
        if hit {
            self.report[g].hits += 1;
            run.streak += 1;
            if run.streak >= 2 {
                self.level[g] = (self.level[g] * LEVEL_DOWN).clamp(lo, hi);
                run.streak = 0;
            }
        } else {
            run.streak = 0;
            self.level[g] = (self.level[g] * LEVEL_UP).clamp(lo, hi);
        }
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
        let (meter_kind, meter, target, aim) = self.meter(run);
        // segments: each rest posture, each calibration round, each gesture's training
        let rest = run.rests;
        let rounds_total = if run.calibrate.is_empty() { 0 } else { 3 };
        let trains = if run.train_after { run.calibrate.len() } else { run.train.len() };
        let segment = match run.step {
            Step::Rest(p) => p,
            Step::Hard(_) => rest,
            Step::Medium(_) => rest + 1,
            Step::Faint(_) => rest + 2,
            Step::Train(g) => rest + rounds_total + run.train.iter().position(|&h| h == g).unwrap_or(0),
        };
        let (round, rounds) = match run.step {
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
            aim,
            segment,
            segments: rest + rounds_total + trains,
            round,
            rounds,
            waiting: Self::waiting(run),
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

    /// Envelope share of each gesture on each electrode site (pronator,
    /// supinator, triceps, biceps, wrist extensors, wrist flexors, finger
    /// flexors, finger extensors), with realistic overlap: rotate out also uses
    /// the biceps, grip braces the wrist with the extensors, and the forearm
    /// flexors share a lot.
    const PATTERNS: [[f64; NUM_CHANNELS]; NUM_GESTURES] = [
        [0.9, 0.05, 0.0, 0.1, 0.05, 0.35, 0.2, 0.0],
        [0.05, 0.85, 0.05, 0.45, 0.35, 0.0, 0.0, 0.15],
        [0.0, 0.05, 0.9, 0.05, 0.1, 0.0, 0.0, 0.05],
        [0.15, 0.25, 0.05, 0.9, 0.05, 0.1, 0.05, 0.0],
        [0.0, 0.3, 0.05, 0.0, 0.9, 0.05, 0.05, 0.45],
        [0.3, 0.0, 0.0, 0.05, 0.1, 0.85, 0.55, 0.05],
        [0.15, 0.0, 0.0, 0.05, 0.35, 0.5, 0.9, 0.6],
    ];
    /// Extra resting tone (µV) on each site in each posture: limp, forearm raised, arm up.
    const TONE: [[f64; NUM_CHANNELS]; POSTURES] = [
        [0.0; NUM_CHANNELS],
        [0.6, 0.9, 0.9, 1.5, 0.9, 0.6, 0.45, 0.6],
        [1.8, 1.5, 2.7, 2.4, 1.5, 1.2, 0.9, 1.2],
    ];

    /// Digit array, sites in channel order: flexors (thumb side → little-finger
    /// side), extensors (same order), biceps, triceps. Each finger leans on its
    /// own site but spills onto its neighbours, and flexing braces the
    /// extensors a little (and the other way round).
    const DIGIT_PATTERNS: [[f64; NUM_CHANNELS]; NUM_DIGITS] = [
        [0.85, 0.35, 0.05, 0.15, 0.05, 0.0, 0.0, 0.0],
        [0.1, 0.05, 0.0, 0.9, 0.35, 0.05, 0.0, 0.0],
        [0.35, 0.9, 0.35, 0.05, 0.15, 0.05, 0.0, 0.0],
        [0.05, 0.1, 0.05, 0.35, 0.9, 0.35, 0.0, 0.0],
        [0.05, 0.35, 0.9, 0.0, 0.05, 0.15, 0.0, 0.0],
        [0.0, 0.05, 0.1, 0.05, 0.35, 0.9, 0.0, 0.0],
        [0.05, 0.1, 0.05, 0.05, 0.0, 0.0, 0.9, 0.1],
        [0.0, 0.0, 0.0, 0.05, 0.05, 0.1, 0.15, 0.9],
    ];

    /// A simulated user who follows the coach by watching its bar.
    struct User<const G: usize> {
        rng: u32,
        patterns: &'static [[f64; NUM_CHANNELS]; G],
        drive: [f64; G],
        /// Which site is wired to each channel (identity = in the order above).
        wiring: [usize; NUM_CHANNELS],
        posture: usize,
        /// Training: the user can't hold lighter than this power, whatever the target.
        floor: f64,
        /// Two movements that, made together, also fire this site by this much
        /// of the weaker one's drive: a compound that isn't the sum of its parts.
        synergy: Option<((usize, usize), usize, f64)>,
    }

    impl User<NUM_GESTURES> {
        fn new(seed: u32) -> Self {
            Self::with(seed, &PATTERNS)
        }
    }

    impl User<NUM_DIGITS> {
        fn digits(seed: u32) -> Self {
            Self::with(seed, &DIGIT_PATTERNS)
        }
    }

    impl<const G: usize> User<G> {
        fn with(seed: u32, patterns: &'static [[f64; NUM_CHANNELS]; G]) -> Self {
            Self {
                rng: seed,
                patterns,
                drive: [0.0; G],
                wiring: std::array::from_fn(|c| c),
                posture: 0,
                floor: 0.0,
                synergy: None,
            }
        }

        fn step(&mut self, gc: &mut GestureControl<G>) {
            let env = std::array::from_fn(|c| {
                let site = self.wiring[c];
                let base = 3.0 * (1.0 + 0.12 * noise(&mut self.rng));
                let tone = TONE[self.posture][site] * (1.0 + 0.25 * noise(&mut self.rng));
                let mut sig: f64 = (0..G).map(|g| self.drive[g] * self.patterns[g][site]).sum();
                if let Some(((a, b), s, k)) = self.synergy {
                    if s == site {
                        sig += k * self.drive[a].min(self.drive[b]);
                    }
                }
                (base + tone + 40.0 * sig * (1.0 + 0.15 * noise(&mut self.rng))) as f32
            });
            gc.process(&env, &[true; NUM_CHANNELS]);
        }

        fn steer(&mut self, g: usize, aim: f64, reading: f64, guess: f64) {
            let d = if self.drive[g] == 0.0 { guess } else { self.drive[g] };
            self.drive = [0.0; G];
            self.drive[g] = (d + 0.003 * (aim - reading)).clamp(0.001, 1.2);
        }

        fn follow(&mut self, gc: &mut GestureControl<G>) {
            if gc.status().is_some_and(|s| s.waiting) {
                gc.ready();
            }
            let st = gc.status();
            self.posture = match st.map(|s| s.step) {
                Some(Step::Rest(p)) => p,
                _ => 0,
            };
            match st {
                Some(st) if st.phase == Phase::Record && !matches!(st.step, Step::Rest(_)) => {
                    let g = st.step.gesture().unwrap();
                    if st.meter_kind == Meter::Activity {
                        self.drive = [0.0; G];
                        self.drive[g] = 1.0;
                    } else {
                        let aim = if st.aim.is_finite() { st.aim.max(self.floor) } else { (st.target.0 + st.target.1) / 2.0 };
                        let guess = if st.meter_kind == Meter::OfHard { aim } else { aim * 0.45 };
                        self.steer(g, aim, st.meter, guess);
                    }
                }
                _ => self.drive = [0.0; G],
            }
            self.step(gc);
        }

        /// Hold gesture `g` at power `aim` for a while; the axis inputs at the end.
        fn hold(&mut self, gc: &mut GestureControl<G>, g: usize, aim: f64, seconds: f32) -> [f32; NUM_JOINTS] {
            for _ in 0..samples(seconds) {
                if aim <= 0.0 {
                    self.drive = [0.0; G];
                } else {
                    self.steer(g, aim, gc.powers()[g], aim * 0.2);
                }
                self.step(gc);
            }
            gc.axis_inputs()
        }
    }

    fn calibrated(u: User<NUM_GESTURES>, train: bool) -> (GestureControl<NUM_GESTURES>, User<NUM_GESTURES>) {
        calibrated_with(GestureControl::muscle(), u, train)
    }

    fn calibrated_with<const G: usize>(mut gc: GestureControl<G>, mut u: User<G>, train: bool) -> (GestureControl<G>, User<G>) {
        for _ in 0..samples(2.0) {
            u.step(&mut gc);
        }
        gc.start_calibration(train);
        for _ in 0..samples(1200.0) {
            if !gc.running() {
                break;
            }
            u.follow(&mut gc);
        }
        assert!(!gc.running(), "run never finished: {:?}", gc.status());
        (gc, u)
    }

    /// Each gesture held at its level drives only its own axis, in its direction.
    fn assert_each_gesture_drives_its_axis<const G: usize>(gc: &mut GestureControl<G>, u: &mut User<G>) {
        for g in 0..G {
            u.hold(gc, g, 0.0, 1.0);
            let level = gc.bands(g).level;
            let inputs = u.hold(gc, g, level * 1.2, 1.5);
            let (axis, dir) = gc.axes()[g];
            assert!(inputs[axis] * dir > 0.5, "gesture {g}: axis {axis} input {inputs:?}");
            for (a, &v) in inputs.iter().enumerate() {
                if a != axis {
                    assert_eq!(v, 0.0, "gesture {g} leaked onto axis {a}: {inputs:?}");
                }
            }
        }
        assert_eq!(u.hold(gc, 0, 0.0, 1.5), [0.0; NUM_JOINTS], "relaxed holds everything");
    }

    #[test]
    fn calibration_learns_every_gesture_and_each_drives_only_its_axis() {
        let (mut gc, mut u) = calibrated(User::new(11), false);
        assert!(gc.calibrated());
        for r in gc.report() {
            assert!(r.calibrated && r.isolation > 0.6, "{r:?}");
        }
        assert_each_gesture_drives_its_axis(&mut gc, &mut u);
    }

    #[test]
    fn electrodes_wired_in_any_order_work_the_same() {
        let mut u = User::new(12);
        u.wiring = [5, 2, 7, 0, 3, 6, 1, 4];
        let (mut gc, mut u) = calibrated(u, false);
        assert!(gc.calibrated());
        assert_each_gesture_drives_its_axis(&mut gc, &mut u);
    }

    #[test]
    fn a_raised_arm_is_not_read_as_a_gesture() {
        let (mut gc, mut u) = calibrated(User::new(13), false);
        for posture in [1, 2, 1, 0] {
            u.posture = posture;
            for _ in 0..samples(3.0) {
                u.step(&mut gc);
                assert_eq!(gc.axis_inputs(), [0.0; NUM_JOINTS], "posture {posture} moved the arm");
            }
            let m = gc.posture();
            assert!(m.iter().all(|&v| v <= m[posture]), "posture {posture} recognised: {m:?}");
        }
        // and a gesture still works with the arm held up
        u.posture = 2;
        let level = gc.bands(3).level;
        let inputs = u.hold(&mut gc, 3, level * 1.2, 1.5);
        assert!(inputs[1] < -0.5, "bicep curl reaches in with the arm up: {inputs:?}");
    }

    #[test]
    fn training_lowers_the_level_for_a_consistent_user() {
        let (gc0, _) = calibrated(User::new(14), false);
        let (gc, _) = calibrated(User::new(14), true);
        let before = gc0.report();
        let after = gc.report();
        for g in 0..NUM_GESTURES {
            assert_eq!(after[g].trials, TRIALS as u32, "gesture {g} trained");
            assert!(after[g].hits >= TRIALS as u32 / 2, "gesture {g}: {} hits", after[g].hits);
            assert!(after[g].level < before[g].level, "gesture {g}: level {} → {}", before[g].level, after[g].level);
        }
    }

    #[test]
    fn training_raises_the_level_for_a_user_who_cant_go_that_light() {
        let (untrained, _) = calibrated(User::new(15), false);
        let mut u = User::new(15);
        u.floor = 0.5; // never holds below half the ceiling
        let (gc, _) = calibrated(u, true);
        for (g, r) in gc.report().iter().enumerate() {
            let before = untrained.report()[g].level;
            // it rises until what they hold fits the trial band (within a step)
            assert!(r.level > before, "gesture {g}: level {before} → {}", r.level);
            assert!(TRIAL_HI * r.level >= 0.5 * LEVEL_DOWN, "gesture {g}: level {} still too low", r.level);
        }
    }

    #[test]
    fn only_the_trained_gesture_moves_the_arm_during_training() {
        let mut gc = GestureControl::muscle();
        let mut u = User::new(16);
        for _ in 0..samples(2.0) {
            u.step(&mut gc);
        }
        gc.start_calibration(true);
        let mut seen_training = false;
        for _ in 0..samples(1200.0) {
            let Some(st) = gc.status() else { break };
            u.follow(&mut gc);
            let inputs = gc.axis_inputs();
            match st.step {
                Step::Train(g) => {
                    seen_training = true;
                    let (axis, _) = AXES[g];
                    assert!(inputs.iter().enumerate().all(|(a, &v)| a == axis || v == 0.0), "{st:?} {inputs:?}");
                }
                _ => assert_eq!(inputs, [0.0; NUM_JOINTS], "phase 1 must hold the arm: {st:?}"),
            }
        }
        assert!(seen_training);
    }

    #[test]
    fn saved_calibration_decodes_the_same_after_restore() {
        let (gc, mut u) = calibrated(User::new(17), true);
        let bytes = bincode::serialize(&gc.calibration()).unwrap();
        let mut restored = GestureControl::muscle();
        restored.restore(bincode::deserialize(&bytes).unwrap());
        let mut gc = gc;
        gc.reset_state();
        for g in [4, 6, 1] {
            for _ in 0..samples(1.0) {
                u.drive = [0.0; NUM_GESTURES];
                u.drive[g] = 0.5;
                let mut a = User { rng: u.rng, ..User::new(0) };
                a.drive = u.drive;
                let mut b = User { rng: u.rng, ..User::new(0) };
                b.drive = u.drive;
                a.step(&mut gc);
                b.step(&mut restored);
                u.rng = a.rng;
                assert_eq!(gc.axis_inputs(), restored.axis_inputs());
            }
        }
    }

    #[test]
    fn each_step_holds_until_ready_then_counts_down_and_records() {
        let mut gc = GestureControl::muscle();
        let mut u = User::new(19);
        gc.start_calibration(false);
        for _ in 0..samples(20.0) {
            u.step(&mut gc);
        }
        let st = gc.status().unwrap();
        assert!(st.waiting && st.step == Step::Rest(0) && st.phase == Phase::Prepare, "held: {st:?}");
        gc.ready();
        for _ in 0..samples(COUNTDOWN_S) + 1 {
            u.step(&mut gc);
        }
        let st = gc.status().unwrap();
        assert_eq!((st.step, st.phase, st.waiting), (Step::Rest(0), Phase::Record, false));
        // ten clean seconds (plus the settling at the start), then the next posture waits again
        for _ in 0..samples(REST_S + SETTLE_S) + 2 {
            u.step(&mut gc);
        }
        let st = gc.status().unwrap();
        assert_eq!((st.step, st.waiting), (Step::Rest(1), true), "{st:?}");
    }

    /// Hold these gestures together at a fixed drive; the axis inputs at the end,
    /// and the most gestures that were active at once along the way.
    fn hold_together(gc: &mut GestureControl<NUM_GESTURES>, u: &mut User<NUM_GESTURES>, gestures: &[usize], drive: f64, seconds: f32) -> ([f32; NUM_JOINTS], usize) {
        let mut most = 0;
        for _ in 0..samples(seconds) {
            u.drive = [0.0; NUM_GESTURES];
            for &g in gestures {
                u.drive[g] = drive;
            }
            u.step(gc);
            most = most.max(gc.active().iter().filter(|&&a| a).count());
        }
        (gc.axis_inputs(), most)
    }

    #[test]
    fn two_gestures_at_once_drive_both_axes() {
        let (mut gc, mut u) = calibrated(User::new(20), false);
        // tricep extend (reach out) and wrist curl (elbow −) together
        let (inputs, _) = hold_together(&mut gc, &mut u, &[2, 5], 0.4, 1.5);
        assert!(inputs[1] > 0.3 && inputs[2] < -0.3, "both should move: {inputs:?}");
        assert_eq!((inputs[0], inputs[3]), (0.0, 0.0), "nothing else: {inputs:?}");
    }

    #[test]
    fn never_more_than_two_gestures_and_crosstalk_is_not_a_second() {
        let (mut gc, mut u) = calibrated(User::new(21), false);
        // three at once: only the two strongest can count
        let (_, most) = hold_together(&mut gc, &mut u, &[0, 2, 5], 0.4, 1.5);
        assert!(most <= 2, "{most} gestures active at once");
        // one at a time, strong: its crosstalk never becomes a second gesture
        for g in 0..NUM_GESTURES {
            hold_together(&mut gc, &mut u, &[], 0.0, 1.0);
            let (_, most) = hold_together(&mut gc, &mut u, &[g], 0.8, 1.5);
            assert_eq!(most, 1, "gesture {g} alone read as a compound");
        }
    }

    #[test]
    fn a_compound_needs_a_moment_to_join_and_holds_through_a_dip() {
        let mut gc = GestureControl::muscle();
        // two movements on different joints (reach out, elbow −)
        let (a, b) = (2, 5);
        gc.templates[a] = Some([1.0; NUM_CHANNELS]);
        gc.templates[b] = Some([1.0; NUM_CHANNELS]);
        // b reads 0.2 × a when only a is made, give or take 0.05
        gc.coupling[a][b] = (0.2, 0.05);
        gc.coupling[b][a] = (0.2, 0.05);
        let hold = |gc: &mut GestureControl<NUM_GESTURES>, p0: f64, p1: f64, s: f32| {
            for _ in 0..samples(s) {
                gc.power = [0.0; NUM_GESTURES];
                gc.power[a] = p0;
                gc.power[b] = p1;
                gc.update_compound();
            }
            gc.decodable()[b]
        };
        let line = |sigmas: f64| 0.2 + 0.05 * sigmas;
        assert!(!hold(&mut gc, 1.0, line(1.5), 1.0), "within 2σ of its crosstalk: not a compound");
        assert!(!hold(&mut gc, 1.0, line(2.5), 0.05), "a blip shorter than the commit doesn't join");
        assert!(!hold(&mut gc, 1.0, line(0.0), 0.2));
        assert!(hold(&mut gc, 1.0, line(2.5), 0.2), "held past 2σ: joins");
        assert!(hold(&mut gc, 1.0, line(1.5), 1.0), "dips to 1.5σ: still together");
        // the two swap which is stronger: still the same pair
        assert!(hold(&mut gc, 0.9, 1.0, 0.5), "swapped order keeps the pair");
        assert!(!hold(&mut gc, 1.0, line(0.5), 0.1), "under 1σ: leaves");
    }

    #[test]
    fn redoing_one_gesture_keeps_the_rest() {
        let (mut gc, mut u) = calibrated(User::new(18), false);
        let before: Vec<_> = gc.data.iter().map(|d| (d.hard_rec.clone(), d.medium.clone(), d.faint.clone())).collect();
        gc.start_gesture(2, false);
        for _ in 0..samples(300.0) {
            if !gc.running() {
                break;
            }
            u.follow(&mut gc);
        }
        assert!(!gc.running());
        assert!(gc.calibrated());
        // the others keep their recordings (their patterns are re-learned from them)
        for g in (0..NUM_GESTURES).filter(|&g| g != 2) {
            let d = &gc.data[g];
            assert!((d.hard_rec.clone(), d.medium.clone(), d.faint.clone()) == before[g], "gesture {g} untouched");
        }
        assert_each_gesture_drives_its_axis(&mut gc, &mut u);
    }

    #[test]
    fn digit_array_learns_every_movement_and_each_drives_only_its_axis() {
        let (mut gc, mut u) = calibrated_with(GestureControl::digits(), User::digits(31), false);
        assert!(gc.calibrated());
        for (g, r) in gc.report().iter().enumerate() {
            assert!(r.calibrated && r.isolation > 0.6, "movement {g}: {r:?}");
        }
        assert_each_gesture_drives_its_axis(&mut gc, &mut u);
    }

    #[test]
    fn digit_array_flex_and_extend_drive_opposite_ways() {
        let (mut gc, mut u) = calibrated_with(GestureControl::digits(), User::digits(32), false);
        // thumb: gripper close then open; index: tip down then up; ring + little: base − then +
        for (flex, ext, axis) in [(0, 1, 3), (2, 3, 2), (4, 5, 0)] {
            u.hold(&mut gc, flex, 0.0, 1.0);
            let aim = gc.bands(flex).level * 1.2;
            let a = u.hold(&mut gc, flex, aim, 1.5)[axis];
            u.hold(&mut gc, ext, 0.0, 1.0);
            let aim = gc.bands(ext).level * 1.2;
            let b = u.hold(&mut gc, ext, aim, 1.5)[axis];
            assert!(a * b < 0.0 && a.abs() > 0.5 && b.abs() > 0.5, "axis {axis}: flex {a}, extend {b}");
        }
        assert_eq!(DIGIT_AXES[0], (3, -1.0), "thumb flex closes the gripper");
    }

    #[test]
    fn digit_calibration_survives_save_and_restore() {
        let (gc, _) = calibrated_with(GestureControl::digits(), User::digits(33), false);
        let bytes = bincode::serialize(&gc.calibration()).unwrap();
        let mut restored = GestureControl::digits();
        restored.restore(bincode::deserialize(&bytes).unwrap());
        assert!(restored.calibrated());
        assert_eq!(restored.templates, gc.templates);
    }

    #[test]
    fn channel_activation_reads_zero_at_rest_and_stays_in_range() {
        let (mut gc, mut u) = calibrated(User::new(40), false);
        u.hold(&mut gc, 0, 0.0, 1.0);
        let rest = gc.channel_activation().unwrap();
        assert!(rest.iter().all(|&a| a < 0.3), "rest reads near 0: {rest:?}");
        // far past any calibration: still within 0..1
        u.drive = [1.2; NUM_GESTURES];
        u.step(&mut gc);
        assert!(gc.channel_activation().unwrap().iter().all(|&a| (0.0..=1.0).contains(&a)));
    }

    #[test]
    fn misses_during_calibration_dont_shape_the_patterns() {
        let (mut gc, mut u) = calibrated(User::new(41), false);
        let before = gc.templates;
        // random misses: spells where one or two channels fire on their own,
        // as much as a third again of each half-strength and faint recording
        let mut rng = 41u32;
        let mut miss = || -> Features {
            let mut z = [0.0; NUM_CHANNELS];
            for _ in 0..2 {
                let c = ((noise(&mut rng) + 1.0) / 2.0 * NUM_CHANNELS as f64) as usize % NUM_CHANNELS;
                z[c] += 25.0 * (1.5 + noise(&mut rng));
            }
            z
        };
        for d in gc.data.iter_mut() {
            for rec in [&mut d.medium, &mut d.faint] {
                let n = rec.len() / 3;
                for k in 0..n {
                    let m = miss();
                    rec.insert((k * 3) % rec.len(), m);
                }
            }
        }
        gc.relearn();
        gc.update_coupling();
        let seen = gc.seen();
        for g in 0..NUM_GESTURES {
            let (a, b) = (gc.templates[g].unwrap(), before[g].unwrap());
            let cos = cosine(&a, &b, &seen);
            assert!(cos > 0.995, "movement {g}: pattern {cos:.4} of the clean one");
        }
        assert_each_gesture_drives_its_axis(&mut gc, &mut u);
    }

    #[test]
    fn a_compound_that_isnt_the_sum_of_its_parts_is_learned_in_use() {
        let mut u = User::new(42);
        // tricep extend + wrist curl together also brace the wrist extensors (site 4)
        u.synergy = Some(((2, 5), 4, 0.6));
        let (mut gc, mut u) = calibrated(u, false);
        assert!(gc.compounds().is_empty());
        for _ in 0..4 {
            hold_together(&mut gc, &mut u, &[], 0.0, 1.0);
            hold_together(&mut gc, &mut u, &[2, 5], 0.4, 1.5);
        }
        hold_together(&mut gc, &mut u, &[], 0.0, 1.0);
        let learned = gc.compounds();
        assert!(learned.iter().any(|&(a, b, _)| (a, b) == (2, 5)), "learned {learned:?}");
        assert!(learned.iter().all(|&(a, b, _)| (a, b) == (2, 5)), "only that pair: {learned:?}");
        // it decodes as both, cleanly
        let (inputs, most) = hold_together(&mut gc, &mut u, &[2, 5], 0.4, 1.5);
        assert!(inputs[1] > 0.3 && inputs[2] < -0.3, "both should move: {inputs:?}");
        assert_eq!((inputs[0], inputs[3], most), (0.0, 0.0, 2), "nothing else: {inputs:?}");
        // and each alone is still just itself
        for g in [2, 5] {
            hold_together(&mut gc, &mut u, &[], 0.0, 1.0);
            let (_, most) = hold_together(&mut gc, &mut u, &[g], 0.6, 1.5);
            assert_eq!(most, 1, "movement {g} alone read as a compound");
        }
        // it survives a save
        let bytes = bincode::serialize(&gc.calibration()).unwrap();
        let mut restored = GestureControl::muscle();
        restored.restore(bincode::deserialize(&bytes).unwrap());
        assert_eq!(restored.compounds(), gc.compounds());
    }

    #[test]
    fn additive_compounds_need_no_pattern_of_their_own() {
        let (mut gc, mut u) = calibrated(User::new(43), false);
        for _ in 0..3 {
            hold_together(&mut gc, &mut u, &[], 0.0, 1.0);
            hold_together(&mut gc, &mut u, &[2, 5], 0.4, 1.5);
        }
        hold_together(&mut gc, &mut u, &[], 0.0, 1.0);
        assert!(gc.compounds().is_empty(), "{:?}", gc.compounds());
    }

    #[test]
    fn a_v5_calibration_still_loads() {
        let (gc, _) = calibrated(User::new(44), false);
        let c = gc.calibration();
        let v5 = bincode::serialize(&(&c.rest, &c.postures, &c.posture_rest, &c.data, &c.templates, &c.level, &c.report)).unwrap();
        let mut restored = GestureControl::muscle();
        restored.restore(bincode::deserialize::<CalibrationV5<NUM_GESTURES>>(&v5).unwrap().into());
        assert_eq!(restored.templates, gc.templates);
    }

    #[test]
    fn a_v6_calibration_still_loads() {
        let (gc, _) = calibrated(User::new(45), false);
        let c = gc.calibration();
        let v6 = bincode::serialize(&(&c.rest, &c.postures, &c.posture_rest, &c.data, &c.templates, &c.level, &c.report, &c.compounds))
            .unwrap();
        let mut restored = GestureControl::muscle();
        restored.restore(bincode::deserialize::<CalibrationV6<NUM_GESTURES>>(&v6).unwrap().into());
        assert_eq!(restored.templates, gc.templates);
    }

    #[test]
    fn calibration_saves_arrays_as_plain_tuples() {
        // the layout muscle-pair calibrations were saved in before the control was generic
        let (gc, _) = calibrated(User::new(34), false);
        let generic = bincode::serialize(&gc.calibration()).unwrap();
        let plain = bincode::serialize(&(&gc.rest, &gc.postures, &gc.posture_rest, &gc.data, &gc.templates, &gc.level, &gc.report, &gc.compounds, &gc.feats)).unwrap();
        assert_eq!(generic, plain);
    }
}

#[cfg(test)]
#[path = "replay_tests.rs"]
mod replay;
