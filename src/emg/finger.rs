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
//!
//! Arm posture changes the resting signal: holding the forearm up tenses the
//! flexors under the grid a little. Calibration therefore records rest in
//! three postures (limp on the table, forearm raised with the elbow down, arm
//! held up unsupported). Each raised posture becomes a pattern of its own. A
//! slow fit of posture and finger patterns says which posture the arm is in;
//! that posture's baseline is taken out before the finger powers are measured,
//! and each finger's noise floor blends between the floors of the postures.

use serde::{Deserialize, Serialize};

use super::cyton::{NUM_CHANNELS, SAMPLE_RATE};
use super::decoder::{nnls, Features};
use crate::arm::NUM_JOINTS;

pub const NUM_FINGERS: usize = 4;
/// Rest postures recorded in calibration: limp on the table (the reference),
/// forearm raised with the elbow down, arm held up unsupported.
pub const POSTURES: usize = 3;
/// Posture is judged on an envelope averaged this long, so a finger pulse
/// doesn't read as a change of posture.
const POSTURE_TAU_S: f32 = 0.5;
/// Time to move into a new posture before its recording starts.
const POSTURE_PREPARE_S: f32 = 4.0;
/// Fit ridges, relative to each pattern's size. The finger fit needs a little
/// to keep overlapping fingers apart. The posture fit only needs the summed
/// baseline right, and the raised postures are often close to scaled copies of
/// each other, which a ridge would shrink.
const FINGER_RIDGE: f64 = 0.02;
const POSTURE_RIDGE: f64 = 0.001;

/// Smoothing of the finger powers.
const POWER_TAU_S: f32 = 0.06;
/// The stasis band sits just above resting noise: this percentile of resting
/// power plus a small margin, and never narrower than STASIS_MIN.
const STASIS_PERCENTILE: f64 = 0.99;
const STASIS_MARGIN: f64 = 0.02;
const STASIS_MIN: f64 = 0.04;
/// The ceiling sits at this percentile of the medium flex: tension past the
/// softest part of a real flex counts as real movement.
const CEILING_PERCENTILE: f64 = 0.2;
/// Faint rounds: the box runs from just above noise (FAINT_MIN × stasis) up to
/// FAINT_FIRST_TOP for the first round, then FAINT_SHRINK × the previous
/// round's level, so every round asks for fainter tension. One round is
/// enough to learn what each finger looks like at low effort.
pub const MAX_FAINT_ROUNDS: usize = 1;
const FAINT_MIN: f64 = 1.3;
const FAINT_FIRST_TOP: f64 = 0.3;
const FAINT_SHRINK: f64 = 0.7;
/// Another round only if its box would still be this many times its bottom…
const FAINT_ROOM: f64 = 1.6;
/// …and the last round came in below this fraction of the one before.
const PROGRESS_RATIO: f64 = 0.85;
/// Command level (the faint, − target): at most this, at least this multiple of stasis.
const LEVEL_MAX: f64 = 0.6;
const LEVEL_MIN_ROOM: f64 = 2.2;
/// Leaving rest needs this fraction of the way from stasis to the level: the
/// large buffer that keeps resting noise from twitching the joint.
const REST_BUFFER: f64 = 0.4;
/// Hysteresis around the −/+ split, as a fraction of split − stasis.
const SPLIT_HYSTERESIS: f64 = 0.1;
/// A faint level only drives − after holding this long, so rising to + doesn't
/// start with a − twitch…
const COMMIT_S: f32 = 0.15;
/// …and only once it has settled: while power is still climbing faster than
/// this (as a fraction of split − stasis, measured as how far it runs ahead of
/// a slower average) the wait starts over, so easing up to + over a second
/// doesn't commit to − on the way. A steady faint hold commits after ~0.3 s.
const RISE_TOL: f64 = 0.15;
const TREND_TAU_S: f32 = 0.15;
/// Firmer (+) needs this long above its threshold, so overshooting on the way
/// to a faint level doesn't flick the joint the other way.
const PLUS_COMMIT_S: f32 = 0.1;
/// Simple fixed bands (the default for now): rest below ¼ of the ceiling,
/// − from ¼ to ½, + above ½ with no upper cap, and no adaptation. The small
/// gaps are hysteresis so a level sitting on a boundary doesn't flicker.
const SIMPLE_BANDS: Bands = Bands {
    stasis: 0.2,
    rest_exit: 0.25,
    level: 0.375,
    split: 0.5,
    plus_enter: 0.5,
    plus_exit: 0.45,
    plus_target: 0.7,
};
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
/// The split stays at least this fraction of the gap inside the two targets.
const SPLIT_MARGIN: f64 = 0.2;
/// Phase 2 teaches the decoder where this person's faint and firm attempts
/// really sit. After each trial, that direction's anchor (the raw power the
/// decoder maps onto the band's target) moves part of the way to the level
/// they held in its last second. Half the way the first time, then less each
/// time (½, ⅓, ¼, ⅕ … with MEET_DECAY = ½), so it settles on their average
/// instead of chasing a user who aims relative to what they see.
const MEET: f64 = 0.5;
const MEET_DECAY: f64 = 0.5;
/// Wobble filter: a finger's power is smoothed longer the more fast jitter
/// the person has relative to the width of a band, up to WOBBLE_TAU_MAX_S.
/// Only jitter faster than JITTER_WINDOW_S counts: slow drift can't be
/// smoothed away, only lagged.
const WOBBLE_GAIN: f64 = 4.0;
const WOBBLE_TAU_MAX_S: f64 = 0.2;
const JITTER_WINDOW_S: f32 = 0.25;
/// The remap never puts the rest line below this multiple of the noise floor,
/// so resting noise can't be stretched into a command.
const REMAP_NOISE_ROOM: f64 = 1.3;
/// …but faint and firmer always stay apart by at least this many of the
/// user's own standard deviations (faint + firm), and at least GAP_MIN_FRAC
/// of the faint level, so the split between them never becomes ambiguous.
const SEP_SIGMAS: f64 = 2.0;
const GAP_MIN_FRAC: f64 = 0.6;
const GAP_MIN: f64 = 0.03;
/// The firmer target stays well under the ceiling.
const PLUS_MAX: f64 = 0.9;

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

/// Starting firmer target for a faint level: as far above the split as the
/// faint one is below it, with the split at twice the faint level (or closer
/// to the ceiling for high levels).
fn default_plus(level: f64) -> f64 {
    let split = (2.0 * level).min(level + 0.35 * (1.0 - level));
    (2.0 * split - level).min(PLUS_MAX)
}

/// Unit vector along `x` over the usable channels; None if it is ~zero.
fn direction(x: &Features, usable: &[bool; NUM_CHANNELS]) -> Option<Features> {
    let n = dot(x, x, usable).sqrt();
    (n > 1e-9).then(|| std::array::from_fn(|c| if usable[c] { x[c] / n } else { 0.0 }))
}

/// One step of a calibration or training run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    /// Whole hand relaxed in posture p: 0 sets the resting level and noise of
    /// every channel, the others record how a raised arm shifts them.
    Rest(usize),
    /// Flex this finger hard (it moves): the top of its range.
    Hard(usize),
    /// About half as hard (it still moves): sets the ceiling.
    Medium(usize),
    /// Faint round k: tension only, fainter than the round before.
    Faint(usize, usize),
    /// Training: hit this finger's faint and firmer bands.
    Train(usize),
}

impl Step {
    pub fn finger(self) -> Option<usize> {
        match self {
            Step::Rest(_) => None,
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
    /// Faint round: above the top of the box (but not yet a real movement).
    Fainter,
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
    /// The firmer (+) target: as far above the split as the faint one is below it.
    pub plus_target: f64,
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
    /// Range to aim for on the meter (NaN when there is none). Training: the
    /// edges of the band the trial asks for.
    pub target: (f64, f64),
    /// Training: the exact level to aim for inside that band (NaN otherwise).
    pub aim: f64,
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
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
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
    /// Current faint (−) and firmer (+) targets.
    pub level: f64,
    pub firm: f64,
    pub stasis: f64,
    /// Where this person's faint and firm attempts average, in raw power
    /// (0 = not measured yet): what the decoder compensates for.
    pub held: [f64; 2],
    /// Faint rounds that looked more like another finger (not used to refine this one).
    pub mixed: u32,
    /// Smoothing applied to this finger to filter out the person's wobble.
    pub smoothing_s: f64,
}

/// Per-finger calibration recordings.
#[derive(Clone, Default, Serialize, Deserialize)]
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
    /// No more faint rounds for this finger (it stopped getting fainter, or was skipped).
    faint_done: bool,
    /// Faint rounds whose pattern looked more like another finger, and so
    /// weren't used to refine this one.
    mixed: u32,
}

/// The limp rest recording: the reference every z-score is measured from.
#[derive(Clone, Serialize, Deserialize)]
struct RestModel {
    mean: Features,
    scale: Features,
    /// Channel had contact during the rest recording.
    seen: [bool; NUM_CHANNELS],
}

struct Run {
    step: Step,
    phase: Phase,
    /// Steps entered before this one (the first gets a longer prepare).
    steps_done: usize,
    t: u32,
    good: u32,
    stuck: u32,
    /// Clean samples of the current Record phase: envelopes for the limp rest,
    /// raw z for the other postures, posture-corrected z for the fingers.
    rec: Vec<Features>,
    /// Decided when a recording finishes, entered after its relax phase.
    next: Option<Step>,
    /// Rest postures in this run (all of them, or none for a single finger / training).
    rests: usize,
    /// Fingers to calibrate in phase 1, and to train in phase 2.
    calibrate: Vec<usize>,
    train: Vec<usize>,
    train_after: bool,
    /// Training: trial number, samples on target, and the finger's power through the trial.
    trial: usize,
    hold: u32,
    trial_powers: Vec<f64>,
}

pub struct FingerControl {
    rest: Option<RestModel>,
    /// Pattern of each raised posture (posture p at index p − 1), in z units.
    postures: Vec<Features>,
    /// Raw z samples of each posture's rest recording (index = posture).
    posture_rest: Vec<Vec<Features>>,
    /// Each finger's noise floor in each recorded posture.
    stasis_by: Vec<[f64; NUM_FINGERS]>,
    /// Envelope z-scores averaged over POSTURE_TAU_S, and the posture weights fitted on them.
    z_slow: Features,
    posture_w: Vec<f64>,
    z_raw: Features,
    data: [FingerData; NUM_FINGERS],
    templates: [Option<Features>; NUM_FINGERS],
    /// Current noise floor of each finger: the posture floors, blended by posture.
    stasis: [f64; NUM_FINGERS],
    /// Faint (−) and firmer (+) targets.
    level: [f64; NUM_FINGERS],
    plus_level: [f64; NUM_FINGERS],
    /// The user's own standard deviation while holding faint and firmer (0 = not measured yet).
    wobble: [[f64; 2]; NUM_FINGERS],
    /// The fast part of it: RMS around a JITTER_WINDOW_S moving average.
    jitter: [[f64; 2]; NUM_FINGERS],
    /// What the user actually holds for faint and for firmer (0 = not measured yet).
    habit: [[f64; 2]; NUM_FINGERS],
    /// Training trials measured so far for faint and for firmer (running means,
    /// and the shrinking step size, are based on them).
    measured: [[u32; 2]; NUM_FINGERS],
    /// Raw power the decoder maps onto the faint and the firm target, learned
    /// in training (NaN = not learned: no remapping).
    anchor: [[f64; 2]; NUM_FINGERS],
    report: [FingerReport; NUM_FINGERS],
    /// Joint each finger drives.
    joint_of: [Option<usize>; NUM_FINGERS],
    pub speed: f32,
    /// Fixed ¼ / ½ bands with no ceiling (true), or the adaptive bands with
    /// the real-movement ceiling and phase 2 fine-tuning (false).
    pub simple_bands: bool,
    envelope: [f32; NUM_CHANNELS],
    contact: [bool; NUM_CHANNELS],
    recent: [f32; NUM_CHANNELS],
    z: Features,
    /// Finger power straight from the fit, lightly smoothed (POWER_TAU_S): what
    /// training measures the person's levels and wobble on.
    raw_power: [f64; NUM_FINGERS],
    /// The same, smoothed per finger according to the person's wobble.
    filtered: [f64; NUM_FINGERS],
    /// What drives the arm and the meters: `filtered` remapped so this person's
    /// faint and firm land on the band targets.
    power: [f64; NUM_FINGERS],
    /// Slower average of each power, to tell a level that is still rising.
    trend: [f64; NUM_FINGERS],
    band: [Band; NUM_FINGERS],
    band_t: [f32; NUM_FINGERS],
    /// Time each finger has been above its + threshold.
    above_t: [f32; NUM_FINGERS],
    run: Option<Run>,
    inputs: [f32; NUM_JOINTS],
}

/// Everything calibration and training learned, to save and restore it. The
/// joint each finger drives, the speed and the band style are panel settings
/// and are saved by the page.
#[derive(Serialize, Deserialize)]
pub struct Calibration {
    rest: Option<RestModel>,
    postures: Vec<Features>,
    posture_rest: Vec<Vec<Features>>,
    stasis_by: Vec<[f64; NUM_FINGERS]>,
    data: [FingerData; NUM_FINGERS],
    templates: [Option<Features>; NUM_FINGERS],
    stasis: [f64; NUM_FINGERS],
    level: [f64; NUM_FINGERS],
    plus_level: [f64; NUM_FINGERS],
    wobble: [[f64; 2]; NUM_FINGERS],
    jitter: [[f64; 2]; NUM_FINGERS],
    habit: [[f64; 2]; NUM_FINGERS],
    measured: [[u32; 2]; NUM_FINGERS],
    anchor: [[f64; 2]; NUM_FINGERS],
    report: [FingerReport; NUM_FINGERS],
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
            postures: Vec::new(),
            posture_rest: Vec::new(),
            stasis_by: Vec::new(),
            z_slow: [0.0; NUM_CHANNELS],
            posture_w: Vec::new(),
            z_raw: [0.0; NUM_CHANNELS],
            data: Default::default(),
            templates: [None; NUM_FINGERS],
            stasis: [STASIS_MIN; NUM_FINGERS],
            level: [LEVEL_MAX; NUM_FINGERS],
            plus_level: [default_plus(LEVEL_MAX); NUM_FINGERS],
            wobble: [[0.0; 2]; NUM_FINGERS],
            jitter: [[0.0; 2]; NUM_FINGERS],
            habit: [[0.0; 2]; NUM_FINGERS],
            measured: [[0; 2]; NUM_FINGERS],
            anchor: [[f64::NAN; 2]; NUM_FINGERS],
            report: [FingerReport::default(); NUM_FINGERS],
            joint_of: std::array::from_fn(|f| Some(f % NUM_JOINTS)),
            speed: DEFAULT_SPEED,
            simple_bands: true,
            envelope: [0.0; NUM_CHANNELS],
            contact: [true; NUM_CHANNELS],
            recent: [0.0; NUM_CHANNELS],
            z: [0.0; NUM_CHANNELS],
            raw_power: [0.0; NUM_FINGERS],
            filtered: [0.0; NUM_FINGERS],
            power: [0.0; NUM_FINGERS],
            trend: [0.0; NUM_FINGERS],
            band: [Band::Rest; NUM_FINGERS],
            band_t: [0.0; NUM_FINGERS],
            above_t: [0.0; NUM_FINGERS],
            run: None,
            inputs: [0.0; NUM_JOINTS],
        }
    }

    pub fn calibration(&self) -> Calibration {
        Calibration {
            rest: self.rest.clone(),
            postures: self.postures.clone(),
            posture_rest: self.posture_rest.clone(),
            stasis_by: self.stasis_by.clone(),
            data: self.data.clone(),
            templates: self.templates,
            stasis: self.stasis,
            level: self.level,
            plus_level: self.plus_level,
            wobble: self.wobble,
            jitter: self.jitter,
            habit: self.habit,
            measured: self.measured,
            anchor: self.anchor,
            report: self.report,
        }
    }

    pub fn restore(&mut self, cal: Calibration) {
        self.run = None;
        self.posture_w = vec![0.0; cal.postures.len()];
        self.rest = cal.rest;
        self.postures = cal.postures;
        self.posture_rest = cal.posture_rest;
        self.stasis_by = cal.stasis_by;
        self.data = cal.data;
        self.templates = cal.templates;
        self.stasis = cal.stasis;
        self.level = cal.level;
        self.plus_level = cal.plus_level;
        self.wobble = cal.wobble;
        self.jitter = cal.jitter;
        self.habit = cal.habit;
        self.measured = cal.measured;
        self.anchor = cal.anchor;
        self.report = cal.report;
        self.reset_state();
    }

    /// Clears the live state after a reconnect (keeps everything learned).
    pub fn reset_state(&mut self) {
        self.recent = [0.0; NUM_CHANNELS];
        self.raw_power = [0.0; NUM_FINGERS];
        self.filtered = [0.0; NUM_FINGERS];
        self.power = [0.0; NUM_FINGERS];
        self.trend = [0.0; NUM_FINGERS];
        self.z_slow = [0.0; NUM_CHANNELS];
        self.posture_w.iter_mut().for_each(|w| *w = 0.0);
        self.band = [Band::Rest; NUM_FINGERS];
        self.band_t = [0.0; NUM_FINGERS];
        self.above_t = [0.0; NUM_FINGERS];
        self.inputs = [0.0; NUM_JOINTS];
    }

    /// Simple ¼ / ½ bands with no ceiling, or the adaptive ones. Clears any
    /// "real movement" hold left over from the adaptive bands.
    pub fn set_simple_bands(&mut self, on: bool) {
        self.simple_bands = on;
        for (b, t) in self.band.iter_mut().zip(&mut self.band_t) {
            if *b == Band::Over {
                *b = Band::Rest;
                *t = 0.0;
            }
        }
    }

    /// The joint a finger drives, or None to leave that finger out of the control.
    pub fn set_joint(&mut self, finger: usize, joint: Option<usize>) {
        if finger < NUM_FINGERS && joint.map_or(true, |j| j < NUM_JOINTS) {
            self.joint_of[finger] = joint;
        }
    }

    pub fn joints(&self) -> [Option<usize>; NUM_FINGERS] {
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
        std::array::from_fn(|f| {
            let b = self.bands(f);
            FingerReport {
                calibrated: self.templates[f].is_some(),
                level: b.level,
                firm: b.plus_target,
                stasis: self.stasis[f],
                held: self.habit[f],
                mixed: self.data[f].mixed,
                smoothing_s: self.smoothing_tau(f),
                ..self.report[f]
            }
        })
    }

    pub fn joint_inputs(&self) -> [f32; NUM_JOINTS] {
        self.inputs
    }

    pub fn bands(&self, f: usize) -> Bands {
        if self.simple_bands {
            return SIMPLE_BANDS;
        }
        let stasis = self.stasis[f];
        let (lo, hi) = self.level_limits(f);
        let level = self.level[f].clamp(lo, hi);
        let gap = self.gap(f, level);
        let plus_target = self.plus_level[f].clamp(level + gap, PLUS_MAX.max(level + gap));
        // The split goes where the user's own faint and firmer are told apart
        // best: equally many of their standard deviations from each. Until both
        // are measured, midway between the targets. Either way inside the targets.
        let [held_faint, held_firm] = self.habit[f];
        let [wob_faint, wob_firm] = self.wobble[f];
        let mid = (level + plus_target) / 2.0;
        let split = if held_faint > 0.0 && held_firm > held_faint {
            let w = if wob_faint + wob_firm > 1e-9 { wob_faint / (wob_faint + wob_firm) } else { 0.5 };
            held_faint + w * (held_firm - held_faint)
        } else {
            mid
        };
        let margin = SPLIT_MARGIN * (plus_target - level);
        let split = split.clamp(level + margin, plus_target - margin);
        let h = SPLIT_HYSTERESIS * (split - stasis) / 2.0;
        Bands {
            stasis,
            rest_exit: stasis + REST_BUFFER * (level - stasis),
            level,
            split,
            plus_enter: split + h,
            plus_exit: split - h,
            plus_target,
        }
    }

    /// The learned anchors (raw powers mapped onto the faint and firm targets),
    /// kept at least the person's own wobble apart so the remap can never make
    /// their faint and firm ambiguous. None until training has measured any.
    fn anchors(&self, f: usize) -> Option<(f64, f64)> {
        let [af, ap] = self.anchor[f];
        if af.is_nan() && ap.is_nan() {
            return None;
        }
        let b = self.bands(f);
        let af = if af.is_nan() { b.level } else { af }.max(1e-3);
        let ap = if ap.is_nan() { b.plus_target } else { ap };
        let [sf, sp] = self.wobble[f];
        let sep = (SEP_SIGMAS * (sf + sp)).max(GAP_MIN_FRAC * af).max(GAP_MIN);
        if ap - af >= sep {
            return Some((af, ap));
        }
        let lo = ((af + ap) / 2.0 - sep / 2.0).max(1e-3);
        Some((lo, lo + sep))
    }

    /// Raw (filtered) power → the power the bands see. Piecewise linear
    /// through (0, 0), (faint anchor, faint target), (firm anchor, firm target),
    /// then on at the same ratio with no cap. Without anchors it's the identity.
    /// Below the faint anchor the slope is limited so the rest line stays at
    /// least REMAP_NOISE_ROOM × the noise floor in raw terms.
    fn remap(&self, f: usize, raw: f64) -> f64 {
        let Some((af, ap)) = self.anchors(f) else { return raw };
        let b = self.bands(f);
        let k0 = (b.level / af).min(b.rest_exit / (REMAP_NOISE_ROOM * self.stasis[f]).max(1e-6));
        let yf = k0 * af;
        if raw <= af {
            raw * k0
        } else if raw <= ap {
            yf + (raw - af) * (b.plus_target - yf) / (ap - af)
        } else {
            b.plus_target + (raw - ap) * (b.plus_target / ap)
        }
    }

    /// Smoothing for a finger's power: longer the more the person wobbles
    /// (measured in training) relative to the width of a band.
    fn smoothing_tau(&self, f: usize) -> f64 {
        let base = POWER_TAU_S as f64;
        if self.measured[f] == [0, 0] {
            return base;
        }
        let b = self.bands(f);
        let half_band = ((b.split - b.rest_exit) / 2.0).max(1e-3);
        let rel = (0..2)
            .filter(|&k| self.measured[f][k] > 0)
            .map(|k| {
                let (h, s) = (self.habit[f][k], self.jitter[f][k]);
                (self.remap(f, h + s) - self.remap(f, (h - s).max(0.0))) / 2.0 / half_band
            })
            .fold(0.0, f64::max);
        (base * (1.0 + WOBBLE_GAIN * rel)).min(WOBBLE_TAU_MAX_S)
    }

    /// Range the faint target may take. Both targets have to fit between the
    /// noise floor and PLUS_MAX: the faint one at least a little above the
    /// floor, and never so high that it leaves the firmer one less than 40%
    /// of that room. A noisy posture or a weak finger squeezes both, rather
    /// than pushing the + band past the ceiling.
    fn level_limits(&self, f: usize) -> (f64, f64) {
        let stasis = self.stasis[f];
        let room = (PLUS_MAX - stasis).max(0.05);
        let lo = (stasis * LEVEL_MIN_ROOM).min(stasis + 0.3 * room);
        let hi = (stasis + 0.6 * room).min(LEVEL_MAX).max(lo);
        (lo, hi)
    }

    /// Distance kept between the faint and firmer targets: wide enough that
    /// the user's own wobble on each can't cross the split, but never more
    /// than the room left under PLUS_MAX.
    fn gap(&self, f: usize, level: f64) -> f64 {
        let [faint, firm] = self.wobble[f];
        let want = (SEP_SIGMAS * (faint + firm)).max(GAP_MIN_FRAC * level).max(GAP_MIN);
        want.min(PLUS_MAX - level)
    }

    /// Keeps the targets inside their limits and apart, moving only the one a
    /// trial just adjusted (`moved`: 0 faint, 1 firmer), so fixing one doesn't
    /// shove the other and set up a back-and-forth.
    fn fit_targets(&mut self, f: usize, moved: usize) {
        let (lo, hi) = self.level_limits(f);
        self.level[f] = self.level[f].clamp(lo, hi);
        let gap = self.gap(f, self.level[f]);
        if moved == 0 {
            if self.plus_level[f] - self.level[f] < gap {
                self.level[f] = (self.plus_level[f] - gap).clamp(lo, hi);
            }
            let gap = self.gap(f, self.level[f]);
            self.plus_level[f] = self.plus_level[f].clamp(self.level[f] + gap, PLUS_MAX.max(self.level[f] + gap));
        } else {
            self.plus_level[f] = self.plus_level[f].clamp(self.level[f] + gap, PLUS_MAX);
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

    /// `z` explained as a non-negative mix of `patterns`, fitted over the usable
    /// channels. `ridge` (relative to each pattern's own size) keeps similar
    /// patterns from trading wild, opposite weights, at the cost of shrinking them.
    fn fit_patterns(patterns: &[&Features], z: &Features, usable: &[bool; NUM_CHANNELS], ridge: f64) -> Vec<f64> {
        let n = patterns.len();
        if n == 0 {
            return Vec::new();
        }
        let mut g: Vec<Vec<f64>> = patterns.iter().map(|a| patterns.iter().map(|b| dot(a, b, usable)).collect()).collect();
        for (i, row) in g.iter_mut().enumerate() {
            row[i] *= 1.0 + ridge;
        }
        let h: Vec<f64> = patterns.iter().map(|a| dot(a, z, usable)).collect();
        nnls(&g, &h)
    }

    /// Power of each calibrated finger in a (posture-corrected) sample.
    fn fit(templates: &[Option<Features>; NUM_FINGERS], z: &Features, usable: &[bool; NUM_CHANNELS]) -> [f64; NUM_FINGERS] {
        let idx: Vec<usize> = (0..NUM_FINGERS).filter(|&f| templates[f].is_some()).collect();
        let patterns: Vec<&Features> = idx.iter().map(|&f| templates[f].as_ref().unwrap()).collect();
        let mut out = [0.0; NUM_FINGERS];
        for (k, w) in Self::fit_patterns(&patterns, z, usable, FINGER_RIDGE).into_iter().enumerate() {
            out[idx[k]] = w;
        }
        out
    }

    /// How much of each raised posture is in a slowly averaged sample. The
    /// fingers are in the fit too, so a held finger isn't taken for posture.
    fn fit_posture(&self, z_slow: &Features, usable: &[bool; NUM_CHANNELS]) -> Vec<f64> {
        if self.postures.is_empty() {
            return Vec::new();
        }
        let patterns: Vec<&Features> = self.postures.iter().chain(self.templates.iter().flatten()).collect();
        let mut w = Self::fit_patterns(&patterns, z_slow, usable, POSTURE_RIDGE);
        w.truncate(self.postures.len());
        w
    }

    /// `z` with the posture baseline taken out.
    fn without_posture(&self, z: &Features, w: &[f64]) -> Features {
        std::array::from_fn(|c| z[c] - self.postures.iter().zip(w).map(|(p, w)| w * p[c]).sum::<f64>())
    }

    /// Share of each recorded posture (limp first) for posture weights `w`.
    ///
    /// The postures lie on a path, limp (no extra tone) → forearm raised → arm
    /// up. The fitted posture signal is placed at the nearest point on that
    /// path and shared between the two postures either side of it. That works
    /// even when one raised pattern is close to a stronger copy of the other,
    /// where the individual weights alone can't say which posture it is.
    fn membership(&self, w: &[f64]) -> Vec<f64> {
        let seen = self.seen();
        let signal: Features = std::array::from_fn(|c| self.postures.iter().zip(w).map(|(p, w)| w * p[c]).sum());
        let points: Vec<Features> = std::iter::once([0.0; NUM_CHANNELS]).chain(self.postures.iter().copied()).collect();
        let mut m = vec![0.0; points.len()];
        if points.len() == 1 {
            m[0] = 1.0;
            return m;
        }
        let (mut best, mut best_seg, mut best_t) = (f64::INFINITY, 0, 0.0);
        for i in 0..points.len() - 1 {
            let d: Features = std::array::from_fn(|c| points[i + 1][c] - points[i][c]);
            let rel: Features = std::array::from_fn(|c| signal[c] - points[i][c]);
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

    /// Each finger's noise floor for the current posture.
    fn blend_stasis(&self) -> [f64; NUM_FINGERS] {
        if self.stasis_by.is_empty() {
            return self.stasis;
        }
        let m = self.membership(&self.posture_w);
        std::array::from_fn(|f| self.stasis_by.iter().zip(&m).map(|(s, m)| m * s[f]).sum())
    }

    /// Current posture: the share of each recorded posture (limp first).
    pub fn posture(&self) -> Vec<f64> {
        if self.posture_rest.is_empty() {
            return Vec::new();
        }
        self.membership(&self.posture_w)
    }

    /// The finger recordings keep the posture they started in, so a strong
    /// flex can't be mistaken for the arm being raised.
    fn posture_frozen(&self) -> bool {
        self.run.as_ref().is_some_and(|r| {
            r.phase == Phase::Record && matches!(r.step, Step::Hard(_) | Step::Medium(_) | Step::Faint(..))
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
            let g = 1.0 - (-1.0 / (POSTURE_TAU_S * SAMPLE_RATE) as f64).exp();
            for (s, &z) in self.z_slow.iter_mut().zip(&z_raw) {
                *s += (z - *s) * g;
            }
            if !self.posture_frozen() {
                self.posture_w = self.fit_posture(&self.z_slow, &usable);
            }
            self.z_raw = z_raw;
            self.z = self.without_posture(&z_raw, &self.posture_w);
            self.stasis = self.blend_stasis();
            let raw = Self::fit(&self.templates, &self.z, &usable);
            let alpha = 1.0 - (-1.0 / (POWER_TAU_S * SAMPLE_RATE) as f64).exp();
            let beta = 1.0 - (-1.0 / (TREND_TAU_S * SAMPLE_RATE) as f64).exp();
            for f in 0..NUM_FINGERS {
                self.raw_power[f] += (raw[f] - self.raw_power[f]) * alpha;
                let a = 1.0 - (-1.0 / (self.smoothing_tau(f) * SAMPLE_RATE as f64)).exp();
                self.filtered[f] += (raw[f] - self.filtered[f]) * a;
                self.power[f] = self.remap(f, self.filtered[f]);
                self.trend[f] += (self.power[f] - self.trend[f]) * beta;
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
        let above = p > b.plus_enter;
        self.above_t[f] = if above { self.above_t[f] + dt } else { 0.0 };
        let plus_ready = self.above_t[f] >= PLUS_COMMIT_S;
        // simple bands have no ceiling: everything over ½ is +
        let (next, t) = if p > 1.0 && !self.simple_bands {
            (Band::Over, 0.0)
        } else {
            match self.band[f] {
                Band::Over => {
                    let t = if p > OVER_EXIT { 0.0 } else { t + dt };
                    if t >= OVER_HOLD_S { (Band::Rest, 0.0) } else { (Band::Over, t) }
                }
                Band::Rest if plus_ready => (Band::Plus, 0.0),
                Band::Rest if p > b.rest_exit => (Band::Pending, 0.0),
                Band::Rest => (Band::Rest, 0.0),
                _ if p < b.stasis => (Band::Rest, 0.0),
                Band::Pending | Band::Minus if plus_ready => (Band::Plus, 0.0),
                // above + but not for long, or still climbing (maybe on its way to +): keep waiting
                Band::Pending if above || p - self.trend[f] > RISE_TOL * (b.split - b.stasis) => (Band::Pending, 0.0),
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
            if let Some(j) = self.joint_of[f] {
                self.inputs[j] = (self.inputs[j] + v).clamp(-1.0, 1.0);
            }
        }
    }

    // ---- runs ---------------------------------------------------------------------

    /// Phase 1 for all four fingers (rest in every posture first), then phase 2 if `train_after`.
    pub fn start_calibration(&mut self, train_after: bool) {
        self.start(Step::Rest(0), (0..NUM_FINGERS).collect(), Vec::new(), train_after);
    }

    /// Phase 1 for one finger, keeping the rest model (a full run if there is none).
    pub fn start_finger(&mut self, finger: usize, train_after: bool) {
        if finger >= NUM_FINGERS {
            return;
        }
        let first = if self.rest.is_some() { Step::Hard(finger) } else { Step::Rest(0) };
        self.start(first, vec![finger], Vec::new(), train_after);
    }

    /// Phase 2 only: every calibrated finger that drives a joint.
    pub fn start_training(&mut self) {
        let fingers: Vec<usize> =
            (0..NUM_FINGERS).filter(|&f| self.templates[f].is_some() && self.joint_of[f].is_some()).collect();
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
            rests: if step == Step::Rest(0) { POSTURES } else { 0 },
            calibrate,
            train,
            train_after,
            trial: 0,
            hold: 0,
            trial_powers: Vec::new(),
        });
    }

    pub fn cancel(&mut self) {
        self.run = None;
    }

    #[cfg(test)]
    fn running(&self) -> bool {
        self.run.is_some()
    }

    /// Skip the current step. A raised posture: left out. Hard or medium flex:
    /// drops that finger from the rest of the calibration. A faint round:
    /// that finger stops at its faintest level so far. Training: on to the
    /// next finger. The limp rest can't be skipped, since everything is
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
            (Step::Hard(_) | Step::Medium(_) | Step::Faint(..), Phase::Relax) => run.next.take(),
            (Step::Hard(f) | Step::Medium(f), _) => {
                self.data[f] = FingerData::default();
                self.templates[f] = None;
                self.next_calibration_step(&mut run, step)
            }
            (Step::Faint(f, _), _) => {
                self.data[f].faint_done = true;
                self.next_calibration_step(&mut run, step)
            }
            (Step::Train(f), _) => Self::after_training(&run, f),
        };
        if self.enter(&mut run, next) {
            self.run = Some(run);
        }
    }

    fn duration(run: &Run) -> u32 {
        match (run.step, run.phase) {
            (Step::Train(_), Phase::Prepare) => samples(CUE_S),
            (Step::Rest(p), Phase::Prepare) if p > 0 => samples(POSTURE_PREPARE_S),
            // the first finger after the raised postures: time to put the arm back down
            (Step::Hard(_), Phase::Prepare) if run.rests > 0 && run.calibrate.first() == run.step.finger().as_ref() => {
                samples(POSTURE_PREPARE_S)
            }
            (_, Phase::Prepare) => samples(if run.steps_done == 0 { PREPARE_FIRST_S } else { PREPARE_S }),
            (Step::Rest(_), Phase::Record) => samples(REST_S),
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

    /// Another finger reading stronger than `f` (and above its own noise): the
    /// strongest such one, i.e. the finger actually being used instead.
    fn stronger_finger(&self, f: usize) -> Option<usize> {
        let p = self.power[f];
        (0..NUM_FINGERS)
            .filter(|&g| g != f && self.templates[g].is_some() && self.power[g] > self.stasis[g].max(p))
            .max_by(|&a, &b| self.power[a].total_cmp(&self.power[b]))
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

    /// Box for finger `f`'s next faint round: from just above noise up to a
    /// little below the last round's level.
    fn faint_box(&self, f: usize) -> (f64, f64) {
        let lo = FAINT_MIN * self.stasis[f];
        let top = self.data[f].faint_levels.last().map_or(FAINT_FIRST_TOP, |&l| FAINT_SHRINK * l);
        (lo, top.max(lo * FAINT_ROOM).min(1.0))
    }

    /// Meter kind, value, target range and exact aim for the coach.
    fn meter(&self, run: &Run) -> (Meter, f64, (f64, f64), f64) {
        let none = (f64::NAN, f64::NAN);
        match (run.step, run.phase) {
            (Step::Rest(_), _) => (Meter::Stillness, self.stillness_recent().1, none, f64::NAN),
            (Step::Hard(_), Phase::Record) => {
                let a = Self::activity(&self.z, &self.usable()) / HARD_MIN_ACTIVITY;
                (Meter::Activity, a, (1.0, f64::INFINITY), f64::NAN)
            }
            (Step::Medium(f), Phase::Record) => (Meter::OfHard, self.of_hard(f), MEDIUM_TARGET, f64::NAN),
            (Step::Faint(f, _), Phase::Record) => (Meter::Power, self.power[f], self.faint_box(f), f64::NAN),
            (Step::Train(f), Phase::Relax) => (Meter::Power, self.power[f], none, f64::NAN),
            (Step::Train(f), _) => {
                let b = self.bands(f);
                let (target, aim) = match Self::goal(run.trial) {
                    Band::Minus => ((b.rest_exit, b.plus_exit), b.level),
                    _ => ((b.plus_enter, 1.0), b.plus_target),
                };
                (Meter::Power, self.power[f], target, aim)
            }
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
            (Step::Rest(_), _) => match self.stillness_recent() {
                (c, r) if r > 1.0 => Check::Moving(c),
                _ => Check::Clean,
            },
            (Step::Hard(_), Phase::Record) => {
                if Self::activity(&self.z, &usable) >= HARD_MIN_ACTIVITY { Check::Clean } else { Check::TooWeak }
            }
            (Step::Medium(f), Phase::Record) => {
                let h = self.of_hard(f);
                let active = Self::activity(&self.z, &usable) >= MEDIUM_MIN_ACTIVITY;
                // every finger's hard pattern is known by now: is it really this one?
                if let Some(g) = self.stronger_finger(f).filter(|_| active) {
                    Check::OtherFinger(g)
                } else if !active || h < MEDIUM_MIN {
                    Check::TooWeak
                } else if h > MEDIUM_MAX {
                    Check::TooStrong
                } else {
                    Check::Clean
                }
            }
            (Step::Faint(f, _), Phase::Record) => {
                let p = self.power[f];
                let (lo, hi) = self.faint_box(f);
                let other = self.stronger_finger(f);
                // simple bands have no ceiling: nothing is rejected as a real movement
                if p > 1.0 && !self.simple_bands {
                    Check::TooStrong
                } else if let Some(g) = other {
                    // the wrong finger: say which, before anything about the level
                    Check::OtherFinger(g)
                } else if p > hi {
                    Check::Fainter
                } else if p < lo {
                    Check::TooWeak
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
                run.trial_powers.push(self.raw_power[f]);
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
                    run.rec.push(match step {
                        Step::Rest(0) => self.envelope.map(f64::from),
                        Step::Rest(_) => self.z_raw,
                        _ => self.z,
                    });
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

    /// A rest recording is done. Limp (p = 0): the reference everything else is
    /// measured against. A raised posture: its pattern and noise.
    fn finish_rest(&mut self, run: &mut Run, p: usize) -> Option<Step> {
        let rec = std::mem::take(&mut run.rec);
        if p == 0 {
            let n = rec.len().max(1);
            let mean = mean(&rec);
            let scale: Features = std::array::from_fn(|c| {
                let var = rec.iter().map(|x| (x[c] - mean[c]).powi(2)).sum::<f64>() / n as f64;
                var.sqrt().max(MIN_SCALE_UV)
            });
            let seen: [bool; NUM_CHANNELS] = std::array::from_fn(|c| self.contact[c] && mean[c] > 0.0);
            let z = rec.iter().map(|x| std::array::from_fn(|c| (x[c] - mean[c]) / scale[c])).collect();
            self.rest = Some(RestModel { mean, scale, seen });
            // every pattern was measured against the old rest: start over
            self.postures.clear();
            self.posture_rest = vec![z];
            self.posture_w.clear();
            self.z_slow = [0.0; NUM_CHANNELS];
            self.templates = [None; NUM_FINGERS];
            self.data = Default::default();
            self.level = [LEVEL_MAX; NUM_FINGERS];
            self.plus_level = [default_plus(LEVEL_MAX); NUM_FINGERS];
            self.wobble = [[0.0; 2]; NUM_FINGERS];
            self.jitter = [[0.0; 2]; NUM_FINGERS];
            self.habit = [[0.0; 2]; NUM_FINGERS];
            self.measured = [[0; 2]; NUM_FINGERS];
            self.anchor = [[f64::NAN; 2]; NUM_FINGERS];
            self.report = [FingerReport::default(); NUM_FINGERS];
            if seen.iter().filter(|&&s| s).count() < MIN_WORKING {
                self.update_stasis();
                return None;
            }
        } else {
            // a posture recorded out of order (one was skipped) still lands in its own slot
            self.postures.push(mean(&rec));
            self.posture_rest.push(rec);
            self.posture_w.push(0.0);
        }
        self.update_stasis();
        Self::after_rest(run, p)
    }

    /// The step after rest posture `p`: the next posture, then the first finger.
    fn after_rest(run: &Run, p: usize) -> Option<Step> {
        if p + 1 < run.rests {
            Some(Step::Rest(p + 1))
        } else {
            run.calibrate.first().map(|&f| Step::Hard(f))
        }
    }

    /// A finger recording is done: learn from it and decide the next step.
    fn finish_recording(&mut self, run: &mut Run) -> Option<Step> {
        let seen = self.seen();
        let rec = std::mem::take(&mut run.rec);
        match run.step {
            Step::Hard(f) => {
                let hard = mean(&rec);
                self.data[f] = FingerData { hard: Some(hard), ..Default::default() };
                // a provisional pattern straight away, so the half-strength round can
                // already tell this finger from the others (scaled so a hard flex reads ~2)
                self.templates[f] = direction(&hard, &seen).and_then(|dir| {
                    let a = percentile(rec.iter().map(|z| dot(&dir, z, &seen)).collect(), CEILING_PERCENTILE) / 2.0;
                    (a > 1e-6).then(|| dir.map(|v| v * a))
                });
                // a new pattern: what training learned about this finger no longer applies
                self.anchor[f] = [f64::NAN; 2];
                self.measured[f] = [0; 2];
                self.habit[f] = [0.0; 2];
                self.wobble[f] = [0.0; 2];
                self.jitter[f] = [0.0; 2];
                self.update_stasis();
            }
            Step::Medium(f) => {
                let d = &mut self.data[f];
                d.shapes = direction(&mean(&rec), &seen).into_iter().collect();
                d.medium = rec;
                self.rebuild_template(f);
                if self.templates[f].is_none() {
                    self.data[f] = FingerData::default(); // no usable pattern: out of the faint rounds
                }
            }
            Step::Faint(f, k) => {
                let level = percentile(rec.iter().map(|z| Self::fit(&self.templates, z, &seen)[f]).collect(), 0.5);
                let prev = self.data[f].faint_levels.iter().copied().fold(f64::INFINITY, f64::min);
                // only refine this finger with the round if it really looks like this finger
                let shape = direction(&mean(&rec), &seen).filter(|s| self.closest_finger(s, &seen) == Some(f));
                let d = &mut self.data[f];
                match shape {
                    Some(s) => d.shapes.push(s),
                    None => d.mixed += 1,
                }
                d.faint_levels.push(level);
                if level < prev {
                    d.faintest = rec;
                }
                self.rebuild_template(f);
                // another round only while there is room above the noise and it keeps getting fainter
                let lo = FAINT_MIN * self.stasis[f];
                let room = k + 1 < MAX_FAINT_ROUNDS && FAINT_SHRINK * level >= lo * FAINT_ROOM;
                if !(room && level < prev * PROGRESS_RATIO) {
                    self.data[f].faint_done = true;
                }
            }
            Step::Rest(_) | Step::Train(_) => return None,
        }
        let step = run.step;
        self.next_calibration_step(run, step)
    }

    /// The finger whose pattern points most the same way as `shape`.
    fn closest_finger(&self, shape: &Features, seen: &[bool; NUM_CHANNELS]) -> Option<usize> {
        (0..NUM_FINGERS)
            .filter_map(|g| {
                let t = self.templates[g].as_ref()?;
                let cos = dot(shape, t, seen) / dot(t, t, seen).sqrt().max(1e-9);
                Some((g, cos))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(g, _)| g)
    }

    /// Phase 1 runs round by round: every finger's hard flex, then every
    /// finger's half-strength flex, then each faint round for every finger
    /// still getting fainter. The step after `after` in that order, or the end
    /// of phase 1 (then training, if asked for).
    fn next_calibration_step(&mut self, run: &mut Run, after: Step) -> Option<Step> {
        let pos = |f: usize| run.calibrate.iter().position(|&g| g == f).map_or(0, |p| p + 1);
        let (mut round, mut i) = match after {
            Step::Hard(f) => (0, pos(f)),
            Step::Medium(f) => (1, pos(f)),
            Step::Faint(f, k) => (2 + k, pos(f)),
            Step::Rest(_) | Step::Train(_) => (0, 0),
        };
        while round < 2 + MAX_FAINT_ROUNDS {
            while let Some(&g) = run.calibrate.get(i) {
                let d = &self.data[g];
                let step = match round {
                    0 => Some(Step::Hard(g)),
                    1 => d.hard.is_some().then_some(Step::Medium(g)),
                    r => (self.templates[g].is_some() && !d.faint_done && d.faint_levels.len() == r - 2)
                        .then_some(Step::Faint(g, r - 2)),
                };
                if step.is_some() {
                    return step;
                }
                i += 1;
            }
            round += 1;
            i = 0;
        }
        self.finalize();
        if !run.train_after {
            return None;
        }
        // a finger that's off is still calibrated (the fit needs its pattern) but has nothing to practise on
        run.train = run.calibrate.iter().copied().filter(|&g| self.templates[g].is_some() && self.joint_of[g].is_some()).collect();
        run.train.first().map(|&g| Step::Train(g))
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

    /// Finger powers over a recorded stretch of raw z, worked out the way the
    /// live decoder does: slow posture fit, posture taken out, fingers fitted.
    fn replay(&self, seq: &[Features]) -> Vec<[f64; NUM_FINGERS]> {
        let seen = self.seen();
        let g = 1.0 - (-1.0 / (POSTURE_TAU_S * SAMPLE_RATE) as f64).exp();
        let mut slow = seq.first().copied().unwrap_or([0.0; NUM_CHANNELS]);
        seq.iter()
            .map(|z| {
                for (s, v) in slow.iter_mut().zip(z) {
                    *s += (v - *s) * g;
                }
                let w = self.fit_posture(&slow, &seen);
                Self::fit(&self.templates, &self.without_posture(z, &w), &seen)
            })
            .collect()
    }

    /// Each finger's noise floor in each recorded posture: above its resting
    /// power there, with every template in the fit.
    fn update_stasis(&mut self) {
        self.stasis_by = self
            .posture_rest
            .iter()
            .map(|seq| {
                let fits = self.replay(seq);
                std::array::from_fn(|f| {
                    let p = percentile(fits.iter().map(|w| w[f]).collect(), STASIS_PERCENTILE);
                    (p + STASIS_MARGIN).max(STASIS_MIN)
                })
            })
            .collect();
        self.stasis = self.blend_stasis();
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
            let (lo, hi) = self.level_limits(f);
            self.level[f] = faint.clamp(lo, hi);
            self.plus_level[f] = default_plus(self.level[f]);
            self.wobble[f] = [0.0; 2];
            self.jitter[f] = [0.0; 2];
            self.habit[f] = [0.0; 2];
            self.measured[f] = [0; 2];
            self.anchor[f] = [f64::NAN; 2];
            self.fit_targets(f, 1);
            self.report[f] = FingerReport {
                faint,
                rounds: self.data[f].faint_levels.len(),
                isolation: on_top as f64 / fits.len().max(1) as f64,
                ..FingerReport::default()
            };
        }
    }

    /// A training trial ended: tally it, then learn from what the person did.
    ///
    /// The targets stay put; the decoder adapts. It notes where this person's
    /// attempt in this direction really sat over the trial's last second (raw
    /// power) and how much it wobbled. Their typical level and wobble are
    /// running means over every trial. The direction's anchor, the raw power
    /// mapped onto the target, moves part of the way to what they held, by a
    /// step that shrinks trial by trial, so it settles on their average.
    fn trial_result(&mut self, run: &mut Run, f: usize, hit: bool) {
        self.report[f].trials += 1;
        if hit {
            self.report[f].hits += 1;
        }
        let n = samples(TRIAL_HOLD_S) as usize;
        let stasis = self.stasis[f];
        let capped = !self.simple_bands;
        let last = &run.trial_powers[run.trial_powers.len().saturating_sub(n)..];
        let tail: Vec<f64> = last.iter().copied().filter(|&p| p > stasis && !(capped && p > 1.0)).collect();
        // fast jitter: RMS around a short moving average of that last second
        let w = samples(JITTER_WINDOW_S) as usize;
        let jitter = if last.len() > w {
            let res: Vec<f64> = (w..last.len())
                .map(|i| last[i] - last[i - w..i].iter().sum::<f64>() / w as f64)
                .collect();
            (res.iter().map(|r| r * r).sum::<f64>() / res.len() as f64).sqrt()
        } else {
            0.0
        };
        run.trial_powers.clear();
        // only when they were really trying for most of that second (not resting or moving)
        if tail.len() < n / 2 {
            return;
        }
        let held = percentile(tail.clone(), 0.5);
        let sigma = (percentile(tail.clone(), 0.75) - percentile(tail, 0.25)) / 1.35;
        let k = usize::from(Self::goal(run.trial) != Band::Minus);
        let b = self.bands(f);
        let default = if k == 0 { b.level } else { b.plus_target };
        self.measured[f][k] += 1;
        let count = self.measured[f][k] as f64;
        self.habit[f][k] += (held - self.habit[f][k]) / count;
        self.wobble[f][k] += (sigma - self.wobble[f][k]) / count;
        self.jitter[f][k] += (jitter - self.jitter[f][k]) / count;
        let step = MEET / (1.0 + (count - 1.0) * MEET_DECAY);
        let anchor = &mut self.anchor[f][k];
        if anchor.is_nan() {
            *anchor = default;
        }
        *anchor += step * (held - *anchor);
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
        // segments: each rest posture, each calibration round, each finger's training
        let rest = run.rests;
        let rounds_total = if run.calibrate.is_empty() { 0 } else { 2 + MAX_FAINT_ROUNDS };
        let trains = if run.train_after { run.calibrate.len() } else { run.train.len() };
        let segment = match run.step {
            Step::Rest(p) => p,
            Step::Hard(_) => rest,
            Step::Medium(_) => rest + 1,
            Step::Faint(_, k) => rest + 2 + k,
            Step::Train(f) => rest + rounds_total + run.train.iter().position(|&g| g == f).unwrap_or(0),
        };
        let (round, rounds) = match run.step {
            Step::Faint(_, k) => (k, MAX_FAINT_ROUNDS),
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

    /// A simulated user: tension per finger (1 = a hard flex), steered towards
    /// whatever the coach asks for by watching its bar, like a person would.
    /// Extra resting tone (µV of envelope) on each grid site in each posture:
    /// limp, forearm raised, arm held up. Raised postures are noisier too.
    const TONE: [[f64; NUM_CHANNELS]; POSTURES] = [
        [0.0; NUM_CHANNELS],
        [1.5, 0.6, 0.3, 0.9, 1.2, 0.45, 0.3, 0.6],
        [2.7, 1.5, 1.2, 2.4, 2.1, 1.2, 1.05, 1.8],
    ];

    struct User {
        rng: u32,
        drive: [f64; NUM_FINGERS],
        off: Option<usize>,
        /// Which grid site is wired to each channel (identity = wired as drawn).
        wiring: [usize; NUM_CHANNELS],
        posture: usize,
        /// A user with habits of their own: in training they produce these raw
        /// powers [faint, firmer] per finger, whatever the screen says.
        habit: Option<[[f64; 2]; NUM_FINGERS]>,
        /// A user who aims relative to the target: [faint, firmer] multipliers.
        relative: Option<[f64; 2]>,
        /// A ~2.5 Hz tremor on every finger's tension (fraction of it).
        tremor: f64,
        t: u64,
    }

    impl User {
        fn new(seed: u32) -> Self {
            Self {
                rng: seed,
                drive: [0.0; NUM_FINGERS],
                off: None,
                wiring: std::array::from_fn(|c| c),
                posture: 0,
                habit: None,
                relative: None,
                tremor: 0.0,
                t: 0,
            }
        }

        fn step(&mut self, fc: &mut FingerControl) {
            self.t += 1;
            let shake = 1.0 + self.tremor * (2.0 * std::f64::consts::PI * 2.5 * self.t as f64 / SAMPLE_RATE as f64).sin();
            let env = std::array::from_fn(|c| {
                let base = 3.0 * (1.0 + 0.12 * noise(&mut self.rng));
                let site = self.wiring[c];
                let tone = TONE[self.posture][site] * (1.0 + 0.25 * noise(&mut self.rng));
                let sig: f64 = (0..NUM_FINGERS).map(|f| self.drive[f] * shake * PATTERNS[f][site]).sum();
                let v = base + tone + 40.0 * sig * (1.0 + 0.15 * noise(&mut self.rng));
                if Some(c) == self.off { 0.0 } else { v as f32 }
            });
            let contact = std::array::from_fn(|c| Some(c) != self.off);
            fc.process(&env, &contact);
        }

        /// Nudge finger `f`'s tension so `reading` approaches `aim`; everything else relaxes.
        fn steer(&mut self, f: usize, aim: f64, reading: f64, guess: f64) {
            let d = if self.drive[f] == 0.0 { guess } else { self.drive[f] };
            self.drive = [0.0; NUM_FINGERS];
            self.drive[f] = (d + 0.003 * (aim - reading)).clamp(0.001, 1.2);
        }

        /// One sample of following the coach: the cued posture (limp for the
        /// fingers), and the cued finger's tension while recording.
        fn follow(&mut self, fc: &mut FingerControl) {
            let st = fc.status();
            self.posture = match st.map(|s| s.step) {
                Some(Step::Rest(p)) => p,
                _ => 0,
            };
            match st {
                Some(st) if st.phase == Phase::Record && !matches!(st.step, Step::Rest(_)) => {
                    let f = st.step.finger().unwrap();
                    if st.meter_kind == Meter::Activity {
                        self.drive = [0.0; NUM_FINGERS];
                        self.drive[f] = 1.0;
                    } else {
                        let habit = match (self.habit, st.goal) {
                            (Some(h), Some(Band::Minus)) => Some(h[f][0]),
                            (Some(h), Some(_)) => Some(h[f][1]),
                            _ => None,
                        };
                        let scale = match (self.relative, st.goal) {
                            (Some(r), Some(Band::Minus)) => r[0],
                            (Some(r), Some(_)) => r[1],
                            _ => 1.0,
                        };
                        let aim = habit.unwrap_or(if st.aim.is_finite() {
                            st.aim * scale
                        } else {
                            st.target.0 + 0.45 * (st.target.1 - st.target.0)
                        });
                        let guess = if st.meter_kind == Meter::OfHard { aim } else { aim * 0.45 };
                        // a habit is what they actually do, whatever the screen shows
                        let reading = if habit.is_some() { fc.raw_power[f] } else { st.meter };
                        self.steer(f, aim, reading, guess);
                    }
                }
                _ => self.drive = [0.0; NUM_FINGERS],
            }
            self.step(fc);
        }

        /// Hold finger `f` at power `aim` (no coach: steering by the decoded
        /// power, easing in from rest). Returns the final joint inputs and the
        /// lowest and highest input each joint saw on the way.
        fn hold(&mut self, fc: &mut FingerControl, f: usize, aim: f64, seconds: f32) -> [f32; NUM_JOINTS] {
            self.hold_range(fc, f, aim, seconds).0
        }

        fn hold_range(
            &mut self,
            fc: &mut FingerControl,
            f: usize,
            aim: f64,
            seconds: f32,
        ) -> ([f32; NUM_JOINTS], [f32; NUM_JOINTS], [f32; NUM_JOINTS]) {
            let (mut lo, mut hi) = ([0.0f32; NUM_JOINTS], [0.0f32; NUM_JOINTS]);
            for _ in 0..samples(seconds) {
                if aim <= 0.0 {
                    self.drive = [0.0; NUM_FINGERS];
                } else {
                    self.steer(f, aim, fc.powers()[f], aim * 0.2);
                }
                self.step(fc);
                for (j, &v) in fc.joint_inputs().iter().enumerate() {
                    lo[j] = lo[j].min(v);
                    hi[j] = hi[j].max(v);
                }
            }
            (fc.joint_inputs(), lo, hi)
        }
    }

    /// A controller using the adaptive bands (ceiling, fine-tuning), which most tests exercise.
    fn adaptive() -> FingerControl {
        let mut fc = FingerControl::new();
        fc.simple_bands = false;
        fc
    }

    #[test]
    fn simple_bands_quarter_and_half_with_no_ceiling() {
        let mut fc = FingerControl::new();
        assert!(fc.simple_bands, "simple bands are the default for now");
        fc.templates[0] = Some([1.0; NUM_CHANNELS]);
        let beta = 1.0 - (-1.0 / (TREND_TAU_S * SAMPLE_RATE) as f64).exp();
        let hold = |fc: &mut FingerControl, p: f64, s: f32| {
            for _ in 0..samples(s) {
                fc.power[0] = p;
                fc.trend[0] += (p - fc.trend[0]) * beta;
                fc.update_band(0);
                fc.update_inputs();
            }
            (fc.band[0], fc.inputs[0])
        };
        assert_eq!(hold(&mut fc, 0.2, 1.0), (Band::Rest, 0.0), "below ¼ holds");
        let (band, v) = hold(&mut fc, 0.35, 1.0);
        assert!(band == Band::Minus && v < 0.0, "¼ to ½ is −");
        let (band, v) = hold(&mut fc, 0.7, 1.0);
        assert!(band == Band::Plus && v > 0.0, "over ½ is +");
        let (band, v) = hold(&mut fc, 2.5, 1.0);
        assert!(band == Band::Plus && v > 0.0, "no ceiling: far over 1 is still +, got {band:?}");
        assert_eq!(hold(&mut fc, 0.0, 1.0), (Band::Rest, 0.0));
        // whatever the noise floor or training does, the bands stay put
        fc.stasis[0] = 0.4;
        fc.level[0] = 0.05;
        let b = fc.bands(0);
        assert_eq!((b.rest_exit, b.plus_enter), (0.25, 0.5));
    }

    fn calibrated(seed: u32, train: bool) -> (FingerControl, User) {
        calibrated_with(User::new(seed), train)
    }

    fn calibrated_with(u: User, train: bool) -> (FingerControl, User) {
        calibrated_in(adaptive(), u, train)
    }

    /// Calibrated with the simple ¼ / ½ bands (the default).
    fn calibrated_simple(u: User) -> (FingerControl, User) {
        calibrated_in(FingerControl::new(), u, false)
    }

    fn calibrated_in(mut fc: FingerControl, mut u: User, train: bool) -> (FingerControl, User) {
        for _ in 0..samples(2.0) {
            u.step(&mut fc);
        }
        fc.start_calibration(train);
        for _ in 0..samples(900.0) {
            if !fc.running() {
                break;
            }
            u.follow(&mut fc);
        }
        assert!(!fc.running(), "run never finished: {:?}", fc.status());
        (fc, u)
    }

    #[test]
    fn bands_are_ordered_with_room_between_them() {
        let mut fc = adaptive();
        // (noise floor, faint level, firm level, wobble): including a noisy raised arm and a
        // shaky user, where the + band used to get pushed past the ceiling
        let cases = [
            (0.1, 0.6, 0.88, 0.0),
            (0.1, 0.25, 0.5, 0.0),
            (0.1, 0.15, 0.3, 0.0),
            (0.2, 0.3, 0.6, 0.0),
            (0.35, 0.6, 0.88, 0.0),
            (0.45, 0.2, 0.3, 0.0),
            (0.3, 0.4, 0.5, 0.15),
            (0.6, 0.6, 0.6, 0.2),
        ];
        for (stasis, level, firm, wobble) in cases {
            fc.stasis[0] = stasis;
            fc.level[0] = level;
            fc.plus_level[0] = firm;
            fc.wobble[0] = [wobble; 2];
            let b = fc.bands(0);
            assert!(b.stasis < b.rest_exit && b.rest_exit < b.level && b.level < b.plus_exit, "{b:?}");
            assert!(b.plus_exit < b.split && b.split < b.plus_enter && b.plus_enter < b.plus_target, "{b:?}");
            assert!(b.plus_target <= PLUS_MAX + 1e-9, "the + target must stay under the ceiling: {b:?}");
            assert!(1.0 - b.plus_enter >= 0.08, "the + band must keep real width: {b:?}");
        }
    }

    #[test]
    fn band_controller_rest_faint_firmer_and_ceiling() {
        let mut fc = adaptive();
        fc.templates[0] = Some([1.0; NUM_CHANNELS]);
        fc.stasis[0] = 0.1;
        fc.level[0] = 0.3;
        let b = fc.bands(0);
        let beta = 1.0 - (-1.0 / (TREND_TAU_S * SAMPLE_RATE) as f64).exp();
        let hold = |fc: &mut FingerControl, p: f64, s: f32| {
            let mut lowest = f32::INFINITY;
            for _ in 0..samples(s) {
                fc.power[0] = p;
                fc.trend[0] += (p - fc.trend[0]) * beta;
                fc.update_band(0);
                fc.update_inputs();
                lowest = lowest.min(fc.inputs[0]);
            }
            lowest
        };
        hold(&mut fc, 0.08, 1.0);
        assert_eq!((fc.band[0], fc.inputs[0]), (Band::Rest, 0.0), "resting noise holds");
        hold(&mut fc, b.level, 0.1);
        assert_eq!((fc.band[0], fc.inputs[0]), (Band::Pending, 0.0), "not committed yet");
        hold(&mut fc, b.level, 0.3);
        assert!(fc.band[0] == Band::Minus && fc.inputs[0] < 0.0, "faint = −");
        hold(&mut fc, 0.02, 0.1);
        assert_eq!(fc.band[0], Band::Rest);

        // overshooting past + for a moment on the way to a faint hold: no + flick
        let mut highest = f32::NEG_INFINITY;
        for (p, s) in [(b.level, 0.05), (b.plus_enter * 1.2, 0.06), (b.level, 0.6)] {
            for _ in 0..samples(s) {
                fc.power[0] = p;
                fc.trend[0] += (p - fc.trend[0]) * beta;
                fc.update_band(0);
                fc.update_inputs();
                highest = highest.max(fc.inputs[0]);
            }
        }
        assert!(highest <= 0.0, "a brief overshoot must not drive +");
        assert_eq!(fc.band[0], Band::Minus, "and the faint hold still commits");
        hold(&mut fc, 0.02, 0.1);

        // a firmer press passes through the faint band on its way up: no − twitch,
        // whether it rises quickly or eases up over a second
        for ramp_s in [0.08, 1.0] {
            let mut lowest = f32::INFINITY;
            let steps = 50;
            for i in 1..=steps {
                lowest = lowest.min(hold(&mut fc, b.plus_target * i as f64 / steps as f64, ramp_s / steps as f32));
            }
            lowest = lowest.min(hold(&mut fc, b.plus_target, 0.5));
            assert!(fc.band[0] == Band::Plus && fc.inputs[0] > 0.0, "firmer = + (ramp {ramp_s} s)");
            assert!(lowest >= 0.0, "never drove − (ramp {ramp_s} s)");
            hold(&mut fc, 0.02, 0.5);
            assert_eq!(fc.band[0], Band::Rest);
        }

        // past the ceiling: a real movement, everything holds for a moment after
        hold(&mut fc, 0.8, 0.5);
        hold(&mut fc, 1.2, 0.05);
        assert_eq!((fc.band[0], fc.inputs[0]), (Band::Over, 0.0));
        hold(&mut fc, 0.8, 0.3);
        assert_eq!(fc.band[0], Band::Over, "still holding just after");
        hold(&mut fc, 0.8, 0.3);
        assert_eq!(fc.band[0], Band::Plus);
    }

    #[test]
    fn phase_one_and_two_with_a_simulated_user() {
        let (fc, _) = calibrated(7, true);
        for (f, r) in fc.report().iter().enumerate() {
            println!("finger {f}: {r:?}");
            assert!(r.calibrated, "finger {f}");
            assert_eq!(r.rounds, 1, "finger {f}: one faint round: {r:?}");
            assert!(r.faint < 0.25, "finger {f} should get well down towards the noise: {r:?}");
            assert_eq!(r.mixed, 0, "finger {f}: the faint round was the right finger: {r:?}");
            assert!(r.isolation > 0.7, "finger {f} mixed up with others: {r:?}");
            assert!(r.stasis >= STASIS_MIN);
            assert_eq!(r.trials, TRIALS as u32, "finger {f}");
            assert!(r.hits as usize >= TRIALS - 2, "finger {f} training: {r:?}");
            assert!(r.level <= r.faint.max(r.stasis * LEVEL_MIN_ROOM) * 1.1, "consistent hits don't raise the level: {r:?}");
            assert!(r.firm > r.level, "{r:?}");
        }
    }

    #[test]
    fn each_finger_drives_only_its_joint_both_ways() {
        let (mut fc, mut u) = calibrated(11, false);
        assert_eq!(u.hold(&mut fc, 0, 0.0, 1.0), [0.0; NUM_JOINTS], "relaxed hand holds still");
        for f in 0..NUM_FINGERS {
            let b = fc.bands(f);
            let j = fc.joints()[f].unwrap();
            let faint = u.hold(&mut fc, f, b.level, 1.5);
            assert!(faint[j] < 0.0, "finger {f} faint: {faint:?} {b:?} power {:?}", fc.powers());
            assert!(faint.iter().enumerate().all(|(k, &v)| k == j || v == 0.0), "finger {f} faint: {faint:?}");
            u.hold(&mut fc, f, 0.0, 1.0);
            // easing up to firmer from rest passes through the faint band without a − twitch
            let (firm, lowest, _) = u.hold_range(&mut fc, f, b.plus_target, 2.0);
            assert!(firm[j] > 0.0, "finger {f} firmer: {firm:?} {b:?} power {:?}", fc.powers());
            assert!(firm.iter().enumerate().all(|(k, &v)| k == j || v == 0.0), "finger {f} firmer: {firm:?}");
            assert!(lowest[j] >= 0.0, "finger {f}: easing up to firmer twitched −");
            u.hold(&mut fc, f, 0.0, 1.0);
        }
        // a real flex trips the ceiling: everything holds
        u.drive = [0.0, 0.0, 0.8, 0.0];
        for _ in 0..samples(0.5) {
            u.step(&mut fc);
        }
        assert!(fc.band_states().contains(&Band::Over));
        assert_eq!(fc.joint_inputs(), [0.0; NUM_JOINTS]);
    }

    /// Runs `sessions` phase-2 sessions; returns the hits per finger in each.
    fn train_sessions(fc: &mut FingerControl, u: &mut User, sessions: usize) -> Vec<[u32; NUM_FINGERS]> {
        let mut out = Vec::new();
        for _ in 0..sessions {
            let before = fc.report().map(|r| r.hits);
            fc.start_training();
            for _ in 0..samples(600.0) {
                if !fc.running() {
                    break;
                }
                u.follow(fc);
            }
            assert!(!fc.running());
            let after = fc.report().map(|r| r.hits);
            out.push(std::array::from_fn(|f| after[f] - before[f]));
        }
        out
    }

    #[test]
    fn a_real_hand_movement_never_locks_the_arm() {
        // simple bands: a strong movement is just a strong +, held as long as it lasts,
        // even when the whole hand joins in (which looks a lot like a raised arm's tone)
        let (mut fc, mut u) = calibrated_simple(User::new(43));
        let j = fc.joints()[2].unwrap();
        for (label, drive) in [("middle flexed hard", [0.0, 0.0, 1.0, 0.0]), ("whole hand", [0.6, 0.6, 1.0, 0.6])] {
            u.drive = drive;
            let mut stalled = 0;
            for i in 0..samples(4.0) {
                u.step(&mut fc);
                if i > samples(0.5) && fc.joint_inputs()[j] <= 0.0 {
                    stalled += 1;
                }
            }
            assert_eq!(stalled, 0, "{label}: the elbow stopped for {stalled} samples; posture {:?}, powers {:?}", fc.posture(), fc.powers());
            u.drive = [0.0; NUM_FINGERS];
            for _ in 0..samples(2.0) {
                u.step(&mut fc);
            }
        }
    }

    #[test]
    fn decoder_learns_where_the_person_averages() {
        // this person's faint really sits at 0.18 raw (under the ¼ rest line) and
        // their firm at 0.42 (under ½, so it reads as faint): fixed bands alone fail them
        let habit = [0.18, 0.42];
        let mut u = User::new(29);
        u.habit = Some([habit; NUM_FINGERS]);
        let (mut fc, mut u) = calibrated_simple(u);
        for f in 0..NUM_FINGERS {
            assert!(fc.remap(f, habit[0]) < 0.25 && fc.remap(f, habit[1]) < 0.5, "untrained: identity");
        }
        let hits = train_sessions(&mut fc, &mut u, 3);
        for f in 0..NUM_FINGERS {
            let (faint, firm) = (fc.remap(f, habit[0]), fc.remap(f, habit[1]));
            assert!(faint > 0.27 && faint < 0.45, "finger {f}: their faint should now read as −: {faint}");
            assert!(firm > 0.55, "finger {f}: their firm should now read as +: {firm}");
            assert!(hits[2][f] >= 7 && hits[2][f] > hits[0][f], "finger {f}: hits per session {hits:?}");
            assert_eq!(fc.bands(f).plus_enter, 0.5, "the bands themselves stay put");
        }
    }

    #[test]
    fn compensation_settles_instead_of_chasing() {
        // the worst case: someone who always aims 30% over the faint target and
        // 20% under the firm one, wherever the screen puts it
        let mut u = User::new(37);
        u.relative = Some([1.3, 0.8]);
        let (mut fc, mut u) = calibrated_simple(u);
        let f = 2;
        // largest relative move of either anchor per trial, in each of three sessions
        let mut moves: Vec<f64> = Vec::new();
        for _ in 0..3 {
            fc.start_training();
            let mut largest: f64 = 0.0;
            let mut trials = fc.report()[f].trials;
            let mut last = fc.anchors(f);
            for _ in 0..samples(600.0) {
                if !fc.running() {
                    break;
                }
                u.follow(&mut fc);
                if fc.report()[f].trials != trials {
                    trials = fc.report()[f].trials;
                    let now = fc.anchors(f);
                    if let (Some(a), Some(b)) = (last, now) {
                        largest = largest.max((b.0 - a.0).abs() / a.0).max((b.1 - a.1).abs() / a.1);
                    }
                    last = now;
                }
            }
            moves.push(largest);
        }
        println!("largest relative anchor move per trial, sessions 1-3: {moves:?}");
        assert!(moves[2] < 0.5 * moves[0], "steps should shrink session by session: {moves:?}");
        assert!(moves[2] < 0.12, "by the third session the decoder should barely move: {moves:?}");
    }

    #[test]
    fn compensation_never_makes_faint_and_firm_ambiguous() {
        // this person does the same for both: the decoder must not map that one
        // level onto both targets
        let mut u = User::new(31);
        u.habit = Some([[0.3, 0.3]; NUM_FINGERS]);
        let (mut fc, mut u) = calibrated_simple(u);
        train_sessions(&mut fc, &mut u, 2);
        for f in 0..NUM_FINGERS {
            let (af, ap) = fc.anchors(f).expect("trained");
            assert!(ap - af >= GAP_MIN_FRAC * af - 1e-9, "finger {f}: anchors {af} / {ap}");
            assert!(fc.remap(f, af) < fc.remap(f, ap), "the remap stays increasing");
            // and resting noise can't be stretched over the rest line
            assert!(fc.remap(f, REMAP_NOISE_ROOM * fc.stasis[f]) <= 0.25 + 1e-9);
        }
    }

    #[test]
    fn a_shaky_person_gets_more_smoothing() {
        let smoothing = |tremor: f64| {
            let mut u = User::new(41);
            u.tremor = tremor;
            let (mut fc, mut u) = calibrated_simple(u);
            train_sessions(&mut fc, &mut u, 1);
            fc.report().iter().map(|r| r.smoothing_s).sum::<f64>() / NUM_FINGERS as f64
        };
        let (steady, shaky) = (smoothing(0.0), smoothing(0.5));
        println!("mean smoothing: steady {:.0} ms, shaky {:.0} ms", steady * 1000.0, shaky * 1000.0);
        assert!(shaky > steady * 1.2, "steady {steady}, shaky {shaky}");
        assert!(shaky <= WOBBLE_TAU_MAX_S + 1e-9);
    }

    #[test]
    fn a_raised_arm_is_not_read_as_finger_tension() {
        let (mut fc, mut u) = calibrated(19, false);
        let floors = fc.stasis_by.clone();
        assert_eq!(floors.len(), POSTURES);
        for f in 0..NUM_FINGERS {
            assert!(floors[2][f] >= floors[0][f], "the unsupported arm is noisier: {floors:?}");
        }
        for posture in [1, 2] {
            u.posture = posture;
            u.hold(&mut fc, 0, 0.0, 1.5); // settling into the posture
            let (still, lo, hi) = u.hold_range(&mut fc, 0, 0.0, 3.0);
            assert_eq!(still, [0.0; NUM_JOINTS], "posture {posture}: relaxed must hold still");
            assert!(lo.iter().chain(&hi).all(|&v| v == 0.0), "posture {posture} twitched: {lo:?} {hi:?}");
            let m = fc.posture();
            assert!(m[posture] > 0.6, "posture {posture} recognised: {m:?}");
            // and the fingers still work with the arm up
            for f in [1, 3] {
                let b = fc.bands(f);
                let j = fc.joints()[f].unwrap();
                let faint = u.hold(&mut fc, f, b.level, 1.5);
                assert!(faint[j] < 0.0, "posture {posture}, finger {f} faint: {faint:?} {b:?} {:?}", fc.powers());
                let firm = u.hold(&mut fc, f, b.plus_target, 2.0);
                assert!(firm[j] > 0.0, "posture {posture}, finger {f} firmer: {firm:?} {b:?} {:?}", fc.powers());
                u.hold(&mut fc, f, 0.0, 1.0);
            }
        }
    }

    #[test]
    fn without_posture_calibration_a_raised_arm_would_drive_the_joints() {
        // the same raised arm, but calibration only knew the limp rest: shows what the postures fix
        let mut fc = adaptive();
        let mut u = User::new(19);
        for _ in 0..samples(2.0) {
            u.step(&mut fc);
        }
        fc.start_calibration(false);
        for _ in 0..samples(900.0) {
            let Some(st) = fc.status() else { break };
            if matches!(st.step, Step::Rest(p) if p > 0) {
                fc.skip();
                continue;
            }
            u.follow(&mut fc);
        }
        u.posture = 2;
        let (_, lo, hi) = u.hold_range(&mut fc, 0, 0.0, 3.0);
        assert!(lo.iter().chain(&hi).any(|&v| v != 0.0), "expected the raised arm to leak into the fingers");
    }

    #[test]
    fn electrodes_wired_in_any_order_work_the_same() {
        // all eight leads plugged into the wrong channels, across both rows
        let mut u = User::new(17);
        u.wiring = [5, 2, 7, 0, 3, 6, 1, 4];
        let (mut fc, mut u) = calibrated_with(u, false);
        for (f, r) in fc.report().iter().enumerate() {
            assert!(r.calibrated && r.isolation > 0.7 && r.faint < 0.25, "finger {f}: {r:?}");
        }
        for f in 0..NUM_FINGERS {
            let b = fc.bands(f);
            let j = fc.joints()[f].unwrap();
            let faint = u.hold(&mut fc, f, (b.rest_exit + b.plus_exit) / 2.0, 1.5);
            assert!(faint[j] < 0.0 && faint.iter().enumerate().all(|(k, &v)| k == j || v == 0.0), "finger {f}: {faint:?}");
            let firm = u.hold(&mut fc, f, (b.plus_enter + 1.0) / 2.0, 1.5);
            assert!(firm[j] > 0.0 && firm.iter().enumerate().all(|(k, &v)| k == j || v == 0.0), "finger {f}: {firm:?}");
            u.hold(&mut fc, f, 0.0, 1.0);
        }
    }

    #[test]
    fn skipping_stops_faint_rounds_or_drops_a_finger() {
        let mut fc = adaptive();
        let mut u = User::new(5);
        for _ in 0..samples(2.0) {
            u.step(&mut fc);
        }
        fc.start_calibration(false);
        let mut skipped_faint = false;
        let mut skipped_hard = false;
        for _ in 0..samples(900.0) {
            let Some(st) = fc.status() else { break };
            if !skipped_hard && st.step == Step::Hard(1) {
                // the index finger is dropped from every later round
                fc.skip();
                skipped_hard = true;
                assert_eq!(fc.status().unwrap().step, Step::Hard(2));
            } else if !skipped_faint && st.step == Step::Faint(0, 0) && st.phase == Phase::Record {
                fc.skip();
                skipped_faint = true;
                assert_eq!(fc.status().unwrap().step, Step::Faint(2, 0), "on to the next finger still in");
            }
            u.follow(&mut fc);
        }
        assert!(skipped_faint && skipped_hard && !fc.running());
        let r = fc.report();
        assert_eq!(r.map(|r| r.calibrated), [true, false, true, true]);
        assert_eq!(r[0].rounds, 0);
        assert_eq!(r[2].rounds, 1);
    }

    #[test]
    fn calibration_goes_round_by_round() {
        let mut fc = adaptive();
        let mut u = User::new(47);
        for _ in 0..samples(2.0) {
            u.step(&mut fc);
        }
        fc.start_calibration(false);
        let mut order: Vec<Step> = Vec::new();
        for _ in 0..samples(900.0) {
            let Some(st) = fc.status() else { break };
            if !matches!(st.step, Step::Rest(_)) && order.last() != Some(&st.step) {
                order.push(st.step);
            }
            u.follow(&mut fc);
        }
        let fingers = || 0..NUM_FINGERS;
        let expected: Vec<Step> = fingers()
            .map(Step::Hard)
            .chain(fingers().map(Step::Medium))
            .chain(fingers().map(|f| Step::Faint(f, 0)))
            .collect();
        assert_eq!(order, expected, "every finger strong, then half, then faint");
    }

    #[test]
    fn using_the_wrong_finger_in_a_low_round_is_caught_and_not_recorded() {
        let mut fc = adaptive();
        let mut u = User::new(53);
        for _ in 0..samples(2.0) {
            u.step(&mut fc);
        }
        fc.start_calibration(false);
        // follow the coach until the index finger's half-strength flex starts
        for _ in 0..samples(900.0) {
            let st = fc.status().expect("still running");
            if st.step == Step::Medium(1) && st.phase == Phase::Record {
                break;
            }
            u.follow(&mut fc);
        }
        // …but flex the middle finger instead, about half as hard
        for _ in 0..samples(2.0) {
            u.drive = [0.0, 0.0, 0.5, 0.0];
            u.step(&mut fc);
        }
        let st = fc.status().unwrap();
        assert_eq!(st.check, Some(Check::OtherFinger(2)), "{st:?}");
        assert_eq!(st.phase_progress, 0.0, "nothing recorded while it's the wrong finger");
        // the faint round catches it too, once it gets there
        for _ in 0..samples(900.0) {
            let st = fc.status().expect("still running");
            if st.step == Step::Faint(1, 0) && st.phase == Phase::Record {
                break;
            }
            u.follow(&mut fc);
        }
        for _ in 0..samples(2.0) {
            u.steer(2, 0.15, fc.powers()[2], 0.05);
            u.step(&mut fc);
        }
        let st = fc.status().unwrap();
        assert_eq!(st.check, Some(Check::OtherFinger(2)), "{st:?}");
        assert!(st.phase_progress < 0.05, "{st:?}");
    }

    #[test]
    fn an_electrode_coming_off_is_left_out_of_the_fit() {
        let (mut fc, mut u) = calibrated(13, false);
        u.off = Some(3); // the ring finger's strongest site
        let b = fc.bands(3);
        let firm = u.hold(&mut fc, 3, (b.plus_enter + 1.0) / 2.0, 2.0);
        assert!(firm[fc.joints()[3].unwrap()] > 0.0, "ring still decoded from the other sites: {firm:?} {:?}", fc.powers());
        assert_eq!(u.hold(&mut fc, 3, 0.0, 1.0), [0.0; NUM_JOINTS]);
    }

    #[test]
    fn saved_calibration_decodes_the_same_after_restore() {
        let (mut fc, _) = calibrated(7, true);
        let bytes = bincode::serialize(&fc.calibration()).unwrap();
        let mut restored = adaptive();
        restored.restore(bincode::deserialize(&bytes).unwrap());
        fc.reset_state();
        assert!(restored.trained());

        let mut rng = 99;
        for t in 0..samples(12.0) {
            // each finger in turn: rest, faint, then firm
            let f = (t / samples(1.0)) as usize % NUM_FINGERS;
            let drive = [0.0, 0.15, 0.4][(t / samples(4.0)) as usize % 3];
            let env = std::array::from_fn(|c| (3.0 * (1.0 + 0.12 * noise(&mut rng)) + 40.0 * drive * PATTERNS[f][c]) as f32);
            fc.process(&env, &[true; NUM_CHANNELS]);
            restored.process(&env, &[true; NUM_CHANNELS]);
            assert_eq!(fc.powers(), restored.powers());
            assert_eq!(fc.joint_inputs(), restored.joint_inputs());
        }
    }

    #[test]
    fn a_finger_turned_off_drives_nothing_and_is_not_trained() {
        let (mut fc, mut u) = calibrated_simple(User::new(11));
        let firm = (fc.bands(1).plus_enter + 1.0) / 2.0;
        assert!(u.hold(&mut fc, 1, firm, 2.0)[fc.joints()[1].unwrap()] > 0.0);
        u.hold(&mut fc, 1, 0.0, 1.0);

        fc.set_joint(1, None);
        assert_eq!(u.hold(&mut fc, 1, firm, 2.0), [0.0; NUM_JOINTS]);
        // the other fingers still work
        let firm0 = (fc.bands(0).plus_enter + 1.0) / 2.0;
        assert!(u.hold(&mut fc, 0, firm0, 2.0)[fc.joints()[0].unwrap()] > 0.0);

        fc.start_training();
        let mut trained = Vec::new();
        while let Some(st) = fc.status() {
            if !trained.contains(&st.step.finger()) {
                trained.push(st.step.finger());
            }
            u.follow(&mut fc);
        }
        assert!(!trained.contains(&Some(1)), "trained {trained:?}");
        assert!(trained.contains(&Some(0)));
    }
}
