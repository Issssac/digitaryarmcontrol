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
//!   hard (which sets the ceiling), re-learning the pattern from both rounds.
//! - Phase 2 has the user hold each gesture at its level, with only that
//!   gesture's axis moving. The level starts at LEVEL_MAX; two hits in a row
//!   lower it a step and a miss raises it, so it settles on the lightest
//!   intention they can hold reliably.

use serde::{Deserialize, Serialize};

use super::cyton::{NUM_CHANNELS, SAMPLE_RATE};
use super::decoder::{nnls, Features};
use crate::arm::NUM_JOINTS;

pub const NUM_GESTURES: usize = 7;
/// Control axis (base, reach, elbow, gripper) and direction each gesture
/// drives: wrist rotate in / out → base + / −, tricep extend / bicep curl →
/// reach out / in, wrist back / curl → elbow + / −, grip → gripper closed.
pub const AXES: [(usize, f32); NUM_GESTURES] =
    [(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0), (3, -1.0)];
/// Rest postures recorded in calibration: limp on the table (the reference),
/// forearm raised with the elbow down, arm held up unsupported.
pub const POSTURES: usize = 3;
/// Posture is judged on an envelope averaged this long, so a gesture pulse
/// doesn't read as a change of posture.
const POSTURE_TAU_S: f32 = 0.5;
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
/// Floor on a coupling's standard deviation, so a very clean calibration can't
/// make every flicker of the runner-up count as a compound.
const COUPLING_MIN_SD: f64 = 0.02;

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
    /// Training: hold this gesture at its level.
    Train(usize),
}

impl Step {
    pub fn gesture(self) -> Option<usize> {
        match self {
            Step::Rest(_) => None,
            Step::Hard(g) | Step::Medium(g) | Step::Train(g) => Some(g),
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
    /// Share of the half-strength round in which this gesture came out on top and above stasis.
    pub isolation: f64,
    pub hits: u32,
    pub trials: u32,
    /// Current level (full speed) and noise floor.
    pub level: f64,
    pub stasis: f64,
}

/// Per-gesture calibration recordings.
#[derive(Clone, Default, Serialize, Deserialize)]
struct GestureData {
    hard: Option<Features>,
    /// z samples of the hard contraction (for the coupling between gestures).
    hard_rec: Vec<Features>,
    /// z samples of the half-strength contraction (they define the ceiling).
    medium: Vec<Features>,
    /// Unit patterns from the half-strength round.
    shapes: Vec<Features>,
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
    /// Phase 1: the user said they're ready for this step, so the countdown runs.
    ready: bool,
    t: u32,
    good: u32,
    stuck: u32,
    /// Clean samples of the current Record phase: envelopes for the limp rest,
    /// raw z for the other postures, posture-corrected z for the gestures.
    rec: Vec<Features>,
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

pub struct GestureControl {
    rest: Option<RestModel>,
    /// Pattern of each raised posture (posture p at index p − 1), in z units.
    postures: Vec<Features>,
    /// Raw z samples of each posture's rest recording (index = posture).
    posture_rest: Vec<Vec<Features>>,
    /// Each gesture's noise floor in each recorded posture.
    stasis_by: Vec<[f64; NUM_GESTURES]>,
    z_slow: Features,
    posture_w: Vec<f64>,
    z_raw: Features,
    data: [GestureData; NUM_GESTURES],
    templates: [Option<Features>; NUM_GESTURES],
    /// Current noise floor of each gesture: the posture floors, blended by posture.
    stasis: [f64; NUM_GESTURES],
    /// Power at which each gesture drives its axis at full speed.
    level: [f64; NUM_GESTURES],
    report: [GestureReport; NUM_GESTURES],
    pub gain: f32,
    envelope: [f32; NUM_CHANNELS],
    contact: [bool; NUM_CHANNELS],
    recent: [f32; NUM_CHANNELS],
    z: Features,
    power: [f64; NUM_GESTURES],
    /// Driving its axis (with hysteresis down to stasis).
    active: [bool; NUM_GESTURES],
    /// Time each gesture has been past its buffer while not yet active.
    rising_t: [f32; NUM_GESTURES],
    /// coupling[a][b]: b's power when only a is made, as (slope on a's power,
    /// standard deviation around it). Worked out from the recordings, not saved.
    coupling: [[(f64, f64); NUM_GESTURES]; NUM_GESTURES],
    /// The two gestures currently decoded together, if any.
    compound: Option<(usize, usize)>,
    /// Time the runner-up has stood out from the strongest one's crosstalk.
    compound_t: f32,
    run: Option<Run>,
    inputs: [f32; NUM_JOINTS],
}

/// Everything calibration and training learned, to save and restore it.
#[derive(Serialize, Deserialize)]
pub struct Calibration {
    rest: Option<RestModel>,
    postures: Vec<Features>,
    posture_rest: Vec<Vec<Features>>,
    data: [GestureData; NUM_GESTURES],
    templates: [Option<Features>; NUM_GESTURES],
    level: [f64; NUM_GESTURES],
    report: [GestureReport; NUM_GESTURES],
}

impl Default for GestureControl {
    fn default() -> Self {
        Self::new()
    }
}

impl GestureControl {
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
            templates: [None; NUM_GESTURES],
            stasis: [STASIS_MIN; NUM_GESTURES],
            level: [LEVEL_MAX; NUM_GESTURES],
            report: [GestureReport::default(); NUM_GESTURES],
            gain: DEFAULT_GAIN,
            envelope: [0.0; NUM_CHANNELS],
            contact: [true; NUM_CHANNELS],
            recent: [0.0; NUM_CHANNELS],
            z: [0.0; NUM_CHANNELS],
            power: [0.0; NUM_GESTURES],
            active: [false; NUM_GESTURES],
            rising_t: [0.0; NUM_GESTURES],
            coupling: [[(0.0, COUPLING_MIN_SD); NUM_GESTURES]; NUM_GESTURES],
            compound: None,
            compound_t: 0.0,
            run: None,
            inputs: [0.0; NUM_JOINTS],
        }
    }

    pub fn calibration(&self) -> Calibration {
        Calibration {
            rest: self.rest.clone(),
            postures: self.postures.clone(),
            posture_rest: self.posture_rest.clone(),
            data: self.data.clone(),
            templates: self.templates,
            level: self.level,
            report: self.report,
        }
    }

    pub fn restore(&mut self, cal: Calibration) {
        self.run = None;
        self.posture_w = vec![0.0; cal.postures.len()];
        self.rest = cal.rest;
        self.postures = cal.postures;
        self.posture_rest = cal.posture_rest;
        self.data = cal.data;
        self.templates = cal.templates;
        self.level = cal.level;
        self.report = cal.report;
        self.update_stasis();
        self.update_coupling();
        self.reset_state();
    }

    /// Clears the live state after a reconnect (keeps everything learned).
    pub fn reset_state(&mut self) {
        self.recent = [0.0; NUM_CHANNELS];
        self.power = [0.0; NUM_GESTURES];
        self.z_slow = [0.0; NUM_CHANNELS];
        self.posture_w.iter_mut().for_each(|w| *w = 0.0);
        self.active = [false; NUM_GESTURES];
        self.rising_t = [0.0; NUM_GESTURES];
        self.compound = None;
        self.compound_t = 0.0;
        self.inputs = [0.0; NUM_JOINTS];
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
        AXES.iter().enumerate().any(|(g, &(a, _))| a == axis && self.templates[g].is_some())
    }

    pub fn powers(&self) -> [f64; NUM_GESTURES] {
        self.power
    }

    pub fn active(&self) -> [bool; NUM_GESTURES] {
        self.active
    }

    /// Axis commands (-1..1): base, reach, elbow, gripper.
    pub fn axis_inputs(&self) -> [f32; NUM_JOINTS] {
        self.inputs
    }

    pub fn report(&self) -> [GestureReport; NUM_GESTURES] {
        std::array::from_fn(|g| GestureReport {
            calibrated: self.templates[g].is_some(),
            level: self.bands(g).level,
            stasis: self.stasis[g],
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
            (0..NUM_GESTURES)
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

    /// Channels that take part in the fit: seen at rest and in contact now.
    fn usable(&self) -> [bool; NUM_CHANNELS] {
        let seen = self.seen();
        std::array::from_fn(|c| seen[c] && self.contact[c])
    }

    fn seen(&self) -> [bool; NUM_CHANNELS] {
        self.rest.as_ref().map_or([false; NUM_CHANNELS], |r| r.seen)
    }

    /// `z` explained as a non-negative mix of `patterns` over the usable channels.
    fn fit_patterns(patterns: &[&Features], z: &Features, usable: &[bool; NUM_CHANNELS], ridge: f64) -> Vec<f64> {
        if patterns.is_empty() {
            return Vec::new();
        }
        let mut g: Vec<Vec<f64>> = patterns.iter().map(|a| patterns.iter().map(|b| dot(a, b, usable)).collect()).collect();
        for (i, row) in g.iter_mut().enumerate() {
            row[i] *= 1.0 + ridge;
        }
        let h: Vec<f64> = patterns.iter().map(|a| dot(a, z, usable)).collect();
        nnls(&g, &h)
    }

    /// Power of each calibrated gesture in a (posture-corrected) sample.
    fn fit(templates: &[Option<Features>; NUM_GESTURES], z: &Features, usable: &[bool; NUM_CHANNELS]) -> [f64; NUM_GESTURES] {
        let idx: Vec<usize> = (0..NUM_GESTURES).filter(|&g| templates[g].is_some()).collect();
        let patterns: Vec<&Features> = idx.iter().map(|&g| templates[g].as_ref().unwrap()).collect();
        let mut out = [0.0; NUM_GESTURES];
        for (k, w) in Self::fit_patterns(&patterns, z, usable, GESTURE_RIDGE).into_iter().enumerate() {
            out[idx[k]] = w;
        }
        out
    }

    /// How much of each raised posture is in a slowly averaged sample. The
    /// gestures are in the fit too, so a held gesture isn't taken for posture.
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

    /// Share of each recorded posture (limp first) for posture weights `w`:
    /// the fitted posture signal placed on the path limp → forearm raised → arm
    /// up, shared between the two postures either side of it (see finger.rs).
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

    /// Each gesture's noise floor for the current posture.
    fn blend_stasis(&self) -> [f64; NUM_GESTURES] {
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
        self.membership(&self.posture_w)
    }

    /// Gesture recordings keep the posture they started in, so a strong
    /// contraction can't be mistaken for the arm being raised.
    fn posture_frozen(&self) -> bool {
        self.run.as_ref().is_some_and(|r| {
            r.phase == Phase::Record && matches!(r.step, Step::Hard(_) | Step::Medium(_))
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
            let s = 1.0 - (-1.0 / (POSTURE_TAU_S * SAMPLE_RATE) as f64).exp();
            for (slow, &z) in self.z_slow.iter_mut().zip(&z_raw) {
                *slow += (z - *slow) * s;
            }
            if !self.posture_frozen() {
                self.posture_w = self.fit_posture(&self.z_slow, &usable);
            }
            self.z_raw = z_raw;
            self.z = self.without_posture(&z_raw, &self.posture_w);
            self.stasis = self.blend_stasis();
            let raw = Self::fit(&self.templates, &self.z, &usable);
            let alpha = 1.0 - (-1.0 / (POWER_TAU_S * SAMPLE_RATE) as f64).exp();
            for (p, r) in self.power.iter_mut().zip(raw) {
                *p += (r - *p) * alpha;
            }
            self.update_compound();
            let decodable = self.decodable();
            for g in 0..NUM_GESTURES {
                if decodable[g] {
                    self.update_active(g);
                } else {
                    self.active[g] = false;
                    self.rising_t[g] = 0.0;
                }
            }
        }
        self.calibration_step();
        self.update_inputs();
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
        let mut order: Vec<usize> = (0..NUM_GESTURES).filter(|&g| self.templates[g].is_some()).collect();
        order.sort_by(|&a, &b| self.power[b].total_cmp(&self.power[a]));
        let (Some(&best), Some(&runner)) = (order.first(), order.get(1)) else {
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
    fn decodable(&self) -> [bool; NUM_GESTURES] {
        let mut ok = [false; NUM_GESTURES];
        let best = (0..NUM_GESTURES)
            .filter(|&g| self.templates[g].is_some())
            .max_by(|&a, &b| self.power[a].total_cmp(&self.power[b]));
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
    /// (hard, half-strength), as a slope through the origin and the
    /// scatter around it. The powers are smoothed the way the live decoder
    /// smooths them, so the scatter is the one the compound test really sees.
    fn update_coupling(&mut self) {
        let seen = self.seen();
        let alpha = 1.0 - (-1.0 / (POWER_TAU_S * SAMPLE_RATE) as f64).exp();
        let smoothed = |rec: &[Features]| -> Vec<[f64; NUM_GESTURES]> {
            let mut p = [0.0; NUM_GESTURES];
            rec.iter()
                .enumerate()
                .map(|(i, z)| {
                    let w = Self::fit(&self.templates, z, &seen);
                    for (pg, wg) in p.iter_mut().zip(w) {
                        *pg = if i == 0 { wg } else { *pg + (wg - *pg) * alpha };
                    }
                    p
                })
                .collect()
        };
        let mut c = [[(0.0, COUPLING_MIN_SD); NUM_GESTURES]; NUM_GESTURES];
        for a in (0..NUM_GESTURES).filter(|&a| self.templates[a].is_some()) {
            let d = &self.data[a];
            let fits: Vec<[f64; NUM_GESTURES]> =
                [&d.hard_rec, &d.medium].into_iter().flat_map(|rec| smoothed(rec)).collect();
            let saa: f64 = fits.iter().map(|w| w[a] * w[a]).sum();
            if fits.len() < 2 || saa < 1e-9 {
                continue;
            }
            for b in (0..NUM_GESTURES).filter(|&b| b != a) {
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
        let gestures: Vec<usize> = match self.run.as_ref().map(|r| r.step) {
            None => (0..NUM_GESTURES).collect(),
            // training: only the gesture being trained moves the arm
            Some(Step::Train(g)) => vec![g],
            Some(_) => Vec::new(),
        };
        for g in gestures.into_iter().filter(|&g| self.active[g]) {
            let (axis, dir) = AXES[g];
            self.inputs[axis] = (self.inputs[axis] + dir * self.speed(g) * self.gain).clamp(-1.0, 1.0);
        }
    }

    // ---- runs ---------------------------------------------------------------------

    /// Phase 1 for every gesture (rest in every posture first), then phase 2 if `train_after`.
    pub fn start_calibration(&mut self, train_after: bool) {
        self.start(Step::Rest(0), (0..NUM_GESTURES).collect(), Vec::new(), train_after);
    }

    /// Phase 1 for one gesture, keeping the rest model (a full run if there is none).
    pub fn start_gesture(&mut self, gesture: usize, train_after: bool) {
        if gesture >= NUM_GESTURES {
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
        let gestures: Vec<usize> = (0..NUM_GESTURES).filter(|&g| self.templates[g].is_some()).collect();
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
    /// drops that gesture from the rest of the calibration. Training: on to the
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
            (Step::Hard(_) | Step::Medium(_), Phase::Relax) => run.next.take(),
            (Step::Hard(g) | Step::Medium(g), _) => {
                self.data[g] = GestureData::default();
                self.templates[g] = None;
                self.next_calibration_step(&mut run, step)
            }
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
            (Step::Hard(_) | Step::Medium(_), Phase::Record) => samples(GESTURE_S),
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
        (0..NUM_GESTURES)
            .filter(|&h| h != g && self.templates[h].is_some() && self.power[h] > self.stasis[h].max(p))
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

    /// Meter kind, value, target range and aim for the coach.
    fn meter(&self, run: &Run) -> (Meter, f64, (f64, f64), f64) {
        let none = (f64::NAN, f64::NAN);
        match (run.step, run.phase) {
            (Step::Rest(_), _) => (Meter::Stillness, self.stillness_recent().1, none, f64::NAN),
            (Step::Hard(_), Phase::Record) => {
                let a = Self::activity(&self.z, &self.usable()) / HARD_MIN_ACTIVITY;
                (Meter::Activity, a, (1.0, f64::INFINITY), f64::NAN)
            }
            (Step::Medium(g), Phase::Record) => (Meter::OfHard, self.of_hard(g), MEDIUM_TARGET, f64::NAN),
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
            (Step::Rest(_), _) => match self.stillness_recent() {
                (c, r) if r > 1.0 => Check::Moving(c),
                _ => Check::Clean,
            },
            (Step::Hard(_), Phase::Record) => {
                if Self::activity(&self.z, &usable) >= HARD_MIN_ACTIVITY { Check::Clean } else { Check::TooWeak }
            }
            (Step::Medium(g), Phase::Record) => {
                let h = self.of_hard(g);
                let active = Self::activity(&self.z, &usable) >= MEDIUM_MIN_ACTIVITY;
                if let Some(other) = self.stronger_gesture(g).filter(|_| active) {
                    Check::OtherGesture(other)
                } else if !active || h < MEDIUM_MIN {
                    Check::TooWeak
                } else if h > MEDIUM_MAX {
                    Check::TooStrong
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
        run.rec.clear();
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
        run.step = step;
        run.phase = Phase::Prepare;
        run.ready = false;
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
            self.templates = [None; NUM_GESTURES];
            self.data = Default::default();
            self.level = [LEVEL_MAX; NUM_GESTURES];
            self.report = [GestureReport::default(); NUM_GESTURES];
            if seen.iter().filter(|&&s| s).count() < MIN_WORKING {
                self.update_stasis();
                return None;
            }
        } else {
            self.postures.push(mean(&rec));
            self.posture_rest.push(rec);
            self.posture_w.push(0.0);
        }
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
        let seen = self.seen();
        let rec = std::mem::take(&mut run.rec);
        match run.step {
            Step::Hard(g) => {
                let hard = mean(&rec);
                self.data[g] = GestureData { hard: Some(hard), hard_rec: rec.clone(), ..Default::default() };
                // a provisional pattern straight away, so the half-strength round can
                // already tell this gesture from the others (scaled so a hard one reads ~2)
                self.templates[g] = direction(&hard, &seen).and_then(|dir| {
                    let a = percentile(rec.iter().map(|z| dot(&dir, z, &seen)).collect(), CEILING_PERCENTILE) / 2.0;
                    (a > 1e-6).then(|| dir.map(|v| v * a))
                });
                self.update_stasis();
            }
            Step::Medium(g) => {
                let d = &mut self.data[g];
                d.shapes = direction(&mean(&rec), &seen).into_iter().collect();
                d.medium = rec;
                self.rebuild_template(g);
                if self.templates[g].is_none() {
                    self.data[g] = GestureData::default(); // no usable pattern
                }
            }
            Step::Rest(_) | Step::Train(_) => return None,
        }
        let step = run.step;
        self.next_calibration_step(run, step)
    }

    /// Phase 1 runs round by round: every gesture hard, then every gesture at
    /// half strength. The step after `after` in that
    /// order, or the end of phase 1 (then training, if asked for).
    fn next_calibration_step(&mut self, run: &mut Run, after: Step) -> Option<Step> {
        let pos = |g: usize| run.calibrate.iter().position(|&h| h == g).map_or(0, |p| p + 1);
        let (mut round, mut i) = match after {
            Step::Hard(g) => (0, pos(g)),
            Step::Medium(g) => (1, pos(g)),
            Step::Rest(_) | Step::Train(_) => (0, 0),
        };
        while round < 2 {
            while let Some(&g) = run.calibrate.get(i) {
                let d = &self.data[g];
                let step = match round {
                    0 => Some(Step::Hard(g)),
                    _ => d.hard.is_some().then_some(Step::Medium(g)),
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

    /// The gesture's pattern is the average of its unit patterns, scaled so that the softest part of the half-strength one reads 1.
    fn rebuild_template(&mut self, g: usize) {
        let seen = self.seen();
        let d = &self.data[g];
        self.templates[g] = direction(&mean(&d.shapes), &seen).and_then(|dir| {
            let a = percentile(d.medium.iter().map(|z| dot(&dir, z, &seen)).collect(), CEILING_PERCENTILE);
            (a > 1e-6).then(|| dir.map(|v| v * a))
        });
        self.update_stasis();
    }

    /// Gesture powers over a recorded stretch of raw z, worked out the way the
    /// live decoder does: slow posture fit, posture taken out, gestures fitted.
    fn replay(&self, seq: &[Features]) -> Vec<[f64; NUM_GESTURES]> {
        let seen = self.seen();
        let s = 1.0 - (-1.0 / (POSTURE_TAU_S * SAMPLE_RATE) as f64).exp();
        let mut slow = seq.first().copied().unwrap_or([0.0; NUM_CHANNELS]);
        seq.iter()
            .map(|z| {
                for (sl, v) in slow.iter_mut().zip(z) {
                    *sl += (v - *sl) * s;
                }
                let w = self.fit_posture(&slow, &seen);
                Self::fit(&self.templates, &self.without_posture(z, &w), &seen)
            })
            .collect()
    }

    /// Each gesture's noise floor in each recorded posture: above its resting
    /// power there, with every template in the fit.
    fn update_stasis(&mut self) {
        self.stasis_by = self
            .posture_rest
            .iter()
            .map(|seq| {
                let fits = self.replay(seq);
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

    /// End of phase 1: each calibrated gesture starts training at LEVEL_MAX,
    /// and its isolation is measured on the half-strength round with all the
    /// templates competing.
    fn finalize(&mut self, gestures: &[usize]) {
        self.update_stasis();
        self.update_coupling();
        let seen = self.seen();
        for &g in gestures {
            if self.templates[g].is_none() {
                continue;
            }
            let fits: Vec<[f64; NUM_GESTURES]> = self.data[g].medium.iter().map(|z| Self::fit(&self.templates, z, &seen)).collect();
            let on_top = fits
                .iter()
                .filter(|w| w[g] > self.stasis[g] && (0..NUM_GESTURES).all(|h| w[h] <= w[g]))
                .count();
            let (lo, hi) = self.level_limits(g);
            self.level[g] = LEVEL_MAX.clamp(lo, hi);
            self.report[g] = GestureReport { isolation: on_top as f64 / fits.len().max(1) as f64, ..Default::default() };
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
        let rounds_total = if run.calibrate.is_empty() { 0 } else { 2 };
        let trains = if run.train_after { run.calibrate.len() } else { run.train.len() };
        let segment = match run.step {
            Step::Rest(p) => p,
            Step::Hard(_) => rest,
            Step::Medium(_) => rest + 1,
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

    /// A simulated user who follows the coach by watching its bar.
    struct User {
        rng: u32,
        drive: [f64; NUM_GESTURES],
        /// Which site is wired to each channel (identity = in the order above).
        wiring: [usize; NUM_CHANNELS],
        posture: usize,
        /// Training: the user can't hold lighter than this power, whatever the target.
        floor: f64,
    }

    impl User {
        fn new(seed: u32) -> Self {
            Self { rng: seed, drive: [0.0; NUM_GESTURES], wiring: std::array::from_fn(|c| c), posture: 0, floor: 0.0 }
        }

        fn step(&mut self, gc: &mut GestureControl) {
            let env = std::array::from_fn(|c| {
                let site = self.wiring[c];
                let base = 3.0 * (1.0 + 0.12 * noise(&mut self.rng));
                let tone = TONE[self.posture][site] * (1.0 + 0.25 * noise(&mut self.rng));
                let sig: f64 = (0..NUM_GESTURES).map(|g| self.drive[g] * PATTERNS[g][site]).sum();
                (base + tone + 40.0 * sig * (1.0 + 0.15 * noise(&mut self.rng))) as f32
            });
            gc.process(&env, &[true; NUM_CHANNELS]);
        }

        fn steer(&mut self, g: usize, aim: f64, reading: f64, guess: f64) {
            let d = if self.drive[g] == 0.0 { guess } else { self.drive[g] };
            self.drive = [0.0; NUM_GESTURES];
            self.drive[g] = (d + 0.003 * (aim - reading)).clamp(0.001, 1.2);
        }

        fn follow(&mut self, gc: &mut GestureControl) {
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
                        self.drive = [0.0; NUM_GESTURES];
                        self.drive[g] = 1.0;
                    } else {
                        let aim = if st.aim.is_finite() { st.aim.max(self.floor) } else { (st.target.0 + st.target.1) / 2.0 };
                        let guess = if st.meter_kind == Meter::OfHard { aim } else { aim * 0.45 };
                        self.steer(g, aim, st.meter, guess);
                    }
                }
                _ => self.drive = [0.0; NUM_GESTURES],
            }
            self.step(gc);
        }

        /// Hold gesture `g` at power `aim` for a while; the axis inputs at the end.
        fn hold(&mut self, gc: &mut GestureControl, g: usize, aim: f64, seconds: f32) -> [f32; NUM_JOINTS] {
            for _ in 0..samples(seconds) {
                if aim <= 0.0 {
                    self.drive = [0.0; NUM_GESTURES];
                } else {
                    self.steer(g, aim, gc.powers()[g], aim * 0.2);
                }
                self.step(gc);
            }
            gc.axis_inputs()
        }
    }

    fn calibrated(mut u: User, train: bool) -> (GestureControl, User) {
        let mut gc = GestureControl::new();
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
    fn assert_each_gesture_drives_its_axis(gc: &mut GestureControl, u: &mut User) {
        for g in 0..NUM_GESTURES {
            u.hold(gc, g, 0.0, 1.0);
            let level = gc.bands(g).level;
            let inputs = u.hold(gc, g, level * 1.2, 1.5);
            let (axis, dir) = AXES[g];
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
    fn training_keeps_the_level_where_a_user_who_cant_go_that_light_can_hold_it() {
        let mut u = User::new(15);
        u.floor = 0.5; // never holds below half the ceiling
        let (gc, _) = calibrated(u, true);
        for (g, r) in gc.report().iter().enumerate() {
            // it doesn't step below what they can hold (within a step)
            assert!(TRIAL_HI * r.level >= 0.5 * LEVEL_DOWN, "gesture {g}: level {} too low", r.level);
        }
    }

    #[test]
    fn only_the_trained_gesture_moves_the_arm_during_training() {
        let mut gc = GestureControl::new();
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
        let mut restored = GestureControl::new();
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
        let mut gc = GestureControl::new();
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
    fn hold_together(gc: &mut GestureControl, u: &mut User, gestures: &[usize], drive: f64, seconds: f32) -> ([f32; NUM_JOINTS], usize) {
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
        let mut gc = GestureControl::new();
        gc.templates[0] = Some([1.0; NUM_CHANNELS]);
        gc.templates[1] = Some([1.0; NUM_CHANNELS]);
        // gesture 1 reads 0.2 × gesture 0 when only 0 is made, give or take 0.05
        gc.coupling[0][1] = (0.2, 0.05);
        gc.coupling[1][0] = (0.2, 0.05);
        let hold = |gc: &mut GestureControl, p0: f64, p1: f64, s: f32| {
            for _ in 0..samples(s) {
                gc.power = [0.0; NUM_GESTURES];
                gc.power[0] = p0;
                gc.power[1] = p1;
                gc.update_compound();
            }
            gc.decodable()[1]
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
        let before = gc.templates;
        gc.start_gesture(2, false);
        for _ in 0..samples(300.0) {
            if !gc.running() {
                break;
            }
            u.follow(&mut gc);
        }
        assert!(!gc.running());
        assert!(gc.calibrated());
        for g in (0..NUM_GESTURES).filter(|&g| g != 2) {
            assert_eq!(gc.templates[g], before[g], "gesture {g} untouched");
        }
    }
}
