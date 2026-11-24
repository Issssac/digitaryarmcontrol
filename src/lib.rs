//! WASM entry point. `App` is the only type exposed to JavaScript.

mod arm;
mod emg;
mod reach;
mod render;
mod world;

use glam::Vec3;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use arm::{deg, Arm, JOINTS, NUM_JOINTS};
use emg::{
    cyton::CytonParser,
    gesture::{self, GestureControl, NUM_DIGITS, NUM_GESTURES},
    pipeline::EmgPipeline,
};
use reach::Reach;
use render::{camera::OrbitCamera, Renderer};
use serde::{Deserialize, Serialize};
use world::World;

const TRAIL_MAX: usize = 1500;
const TRAIL_MIN_STEP: f32 = 0.003;
/// Once no joint has had a command for this long, the arm starts drifting back home.
const DRIFT_DELAY_S: f32 = 1.5;

/// Manually, holding the gripper's + or − button moves it toward open or closed,
/// and letting go stops it where it is. Under muscle-pair EMG, once the grip
/// gesture is trained, it closes while you grip and opens when you let go. On
/// the digit array the thumb drives it like any other axis: flexing closes it,
/// extending opens it, at a speed that grows with the tension, and relaxing
/// leaves it where it is.
const GRIPPER: usize = 3;
const GRIPPER_OPEN: f32 = deg(60.0);
const GRIPPER_CLOSED: f32 = 0.0;
/// Velocity command per radian of gripper error, and the error treated as arrived.
const GRIPPER_GAIN: f32 = 6.0;
const GRIPPER_TOLERANCE: f32 = deg(0.3);

/// Bumped whenever a saved calibration's layout changes, so an old file is
/// refused instead of misread. 2: muscle pairs became gestures, not channels.
/// 3: gestures calibrate like the fingers (rest postures, rounds, training).
/// 4: gestures keep their hard-round samples (for the compound coupling).
/// 5: the finger grid became the digit array (eight bipolar pairs). A v4 file
/// still gives back its muscle-pair calibration, which is saved the same way.
const CALIBRATION_VERSION: u32 = 5;

/// Both schemes' calibrations, as saved by `export_calibration`.
#[derive(Serialize, Deserialize)]
struct SavedCalibration {
    version: u32,
    muscle: gesture::Calibration<NUM_GESTURES>,
    digit: gesture::Calibration<NUM_DIGITS>,
}

/// The start of a v4 file: its muscle-pair calibration (the finger grid's after it is dropped).
#[derive(Deserialize)]
struct SavedCalibrationV4 {
    #[allow(dead_code)]
    version: u32,
    muscle: gesture::Calibration<NUM_GESTURES>,
}

/// Runs `$e` with `$c` bound to the control of the current scheme (muscle pairs or the digit array).
macro_rules! active {
    ($self:ident, $c:ident => $e:expr) => {
        if $self.digit_mode {
            let $c = &$self.digits;
            $e
        } else {
            let $c = &$self.gestures;
            $e
        }
    };
    (mut $self:ident, $c:ident => $e:expr) => {
        if $self.digit_mode {
            let $c = &mut $self.digits;
            $e
        } else {
            let $c = &mut $self.gestures;
            $e
        }
    };
}

#[wasm_bindgen]
pub struct App {
    renderer: Renderer,
    camera: OrbitCamera,
    arm: Arm,
    world: World,
    manual: [f32; NUM_JOINTS],
    use_emg: bool,
    parser: CytonParser,
    emg: EmgPipeline,
    /// Muscle-pair control (when `digit_mode` is off).
    gestures: GestureControl<NUM_GESTURES>,
    /// Digit-array control (eight bipolar pairs) when `digit_mode` is on.
    digits: GestureControl<NUM_DIGITS>,
    digit_mode: bool,
    drift: bool,
    /// Seconds since any joint last had a command.
    idle: f32,
    /// Angle the gripper is moving to (radians).
    gripper_target: f32,
    /// The joints were set from outside (the real arm) for this frame: no inputs, no drift.
    tracked: bool,
    trail: Vec<Vec3>,
    /// Turns the reach axis into shoulder + elbow motion.
    reach: Reach,
}

#[wasm_bindgen]
impl App {
    #[wasm_bindgen(constructor)]
    pub fn new(canvas: HtmlCanvasElement) -> Result<App, JsValue> {
        console_error_panic_hook::set_once();
        Ok(App {
            renderer: Renderer::new(canvas)?,
            camera: OrbitCamera::default(),
            arm: Arm::new(),
            world: World::new(),
            manual: [0.0; NUM_JOINTS],
            use_emg: false,
            parser: CytonParser::new(),
            emg: EmgPipeline::new(),
            gestures: GestureControl::muscle(),
            digits: GestureControl::digits(),
            digit_mode: false,
            drift: true,
            idle: 0.0,
            tracked: false,
            gripper_target: GRIPPER_CLOSED,
            trail: Vec::new(),
            reach: Reach::default(),
        })
    }

    // ---- view -------------------------------------------------------------

    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.camera.orbit(dx, dy);
    }

    pub fn zoom(&mut self, delta: f32) {
        self.camera.zoom(delta);
    }

    /// Advance the simulation by `dt` seconds and draw a frame.
    pub fn frame(&mut self, dt: f32) {
        let dt = dt.clamp(0.0, 0.1);
        if std::mem::take(&mut self.tracked) {
            self.idle = 0.0;
        } else {
            self.drive(dt);
        }
        self.world.step(self.arm.frames().tool, self.arm.angle(3));

        let ee = self.arm.end_effector();
        if self.trail.last().map_or(true, |p| p.distance(ee) > TRAIL_MIN_STEP) {
            if self.trail.len() >= TRAIL_MAX {
                self.trail.remove(0);
            }
            self.trail.push(ee);
        }

        self.renderer.render(&self.camera, &self.arm, &self.world, &self.trail);
    }

    /// Integrate this frame's joint commands (manual or EMG, plus idle drift).
    fn drive(&mut self, dt: f32) {
        // Each control decides itself what moves while it calibrates: nothing in
        // phase 1, the trained movement's axis in phase 2. Both drive axes (base,
        // reach, elbow, gripper), not joints.
        let axes = active!(self, c => c.axis_inputs());
        // manual keys and buttons drive the same axes, so W/S reach in a straight line too
        let axes = if self.use_emg { axes } else { self.manual };
        let mut inputs = self.reach.joint_inputs(&self.arm, axes, dt);
        // Drift only once the whole arm is idle, so holding one joint still
        // while driving another doesn't pull the held one back home. The gripper
        // only ever goes where its button sends it.
        let arm_joints = 0..GRIPPER;
        if inputs[arm_joints.clone()].iter().any(|&v| v != 0.0) || !(self.use_emg && self.drift) {
            self.idle = 0.0;
        } else {
            self.idle += dt;
            if self.idle >= DRIFT_DELAY_S {
                for j in arm_joints {
                    inputs[j] = self.arm.drift_input(j);
                }
            }
        }
        // muscle-pair EMG: the gripper closes while the grip gesture is engaged
        // and opens when it lets go
        if self.use_emg && !self.digit_mode && self.gestures.drives(GRIPPER) {
            self.gripper_target = if self.gestures.axis_inputs()[GRIPPER] < 0.0 { GRIPPER_CLOSED } else { GRIPPER_OPEN };
        }
        // digit array: the thumb moves it at a speed, and it stays wherever the thumb lets go
        let thumb = if self.use_emg && self.digit_mode { axes[GRIPPER] } else { 0.0 };
        let error = self.gripper_target - self.arm.angle(GRIPPER);
        inputs[GRIPPER] = if thumb != 0.0 {
            thumb
        } else if error.abs() < GRIPPER_TOLERANCE {
            0.0
        } else {
            (error * GRIPPER_GAIN / JOINTS[GRIPPER].max_speed).clamp(-1.0, 1.0)
        };
        for (j, v) in inputs.into_iter().enumerate() {
            self.arm.set_input(j, v);
        }
        self.arm.step(dt);
        if thumb != 0.0 {
            self.gripper_target = self.arm.angle(GRIPPER);
        }
    }

    /// Binary STL for link `i` of `link_mesh_names()`; until all are loaded
    /// the arm is drawn from primitives.
    pub fn load_link_mesh(&mut self, i: usize, bytes: &[u8]) -> Result<(), JsValue> {
        self.renderer.load_link(i, bytes)
    }

    pub fn link_mesh_names(&self) -> Vec<String> {
        render::scene::LINK_MESHES.iter().map(|s| s.to_string()).collect()
    }

    // ---- arm --------------------------------------------------------------

    /// Puts the joints at these angles (degrees) for the next frame, which then
    /// ignores the manual/EMG commands: the simulation mirrors the real arm.
    pub fn track_joints(&mut self, degrees: &[f32]) {
        if let Ok(q) = <[f32; NUM_JOINTS]>::try_from(degrees) {
            self.arm.set_angles(q.map(f32::to_radians));
            self.tracked = true;
            // the button carries on from wherever the real gripper is
            self.gripper_target = self.arm.angle(GRIPPER);
        }
    }

    /// Open (true) or close the gripper.
    pub fn set_gripper(&mut self, open: bool) {
        self.gripper_target = if open { GRIPPER_OPEN } else { GRIPPER_CLOSED };
    }

    /// Stop the gripper where it is now.
    pub fn hold_gripper(&mut self) {
        self.gripper_target = self.arm.angle(GRIPPER);
    }

    /// The gripper is open or opening.
    pub fn gripper_open(&self) -> bool {
        self.gripper_target > (GRIPPER_OPEN + GRIPPER_CLOSED) / 2.0
    }

    /// Manual axis command (-1..1) while EMG control is off: base, reach (in a
    /// straight line along the gripper, like EMG), elbow, gripper.
    pub fn set_input(&mut self, joint: usize, value: f32) {
        if let Some(m) = self.manual.get_mut(joint) {
            *m = value;
        }
    }

    pub fn set_use_emg(&mut self, on: bool) {
        self.use_emg = on;
    }

    /// Velocity commands currently applied to the arm.
    pub fn joint_inputs(&self) -> Vec<f32> {
        (0..NUM_JOINTS).map(|j| self.arm.input(j)).collect()
    }

    /// Joint angles in degrees.
    pub fn joint_angles(&self) -> Vec<f32> {
        (0..NUM_JOINTS).map(|j| self.arm.angle(j).to_degrees()).collect()
    }

    /// [min0, max0, min1, max1, …] in degrees.
    pub fn joint_limits(&self) -> Vec<f32> {
        JOINTS.iter().flat_map(|s| [s.min.to_degrees(), s.max.to_degrees()]).collect()
    }

    pub fn joint_names(&self) -> Vec<String> {
        JOINTS.iter().map(|s| s.name.to_string()).collect()
    }

    /// End-effector position in metres [x, y, z].
    pub fn end_effector(&self) -> Vec<f32> {
        self.arm.end_effector().to_array().to_vec()
    }

    /// Home pose, box back on the table, trail cleared.
    pub fn reset(&mut self) {
        self.arm.reset();
        self.gripper_target = GRIPPER_CLOSED;
        self.world.reset();
        self.trail.clear();
    }

    /// True while the claw is holding the box.
    pub fn holding(&self) -> bool {
        self.world.holding()
    }

    pub fn clear_trail(&mut self) {
        self.trail.clear();
    }

    // ---- OpenBCI / EMG ----------------------------------------------------

    /// Feed raw bytes from the Cyton serial stream.
    pub fn push_serial(&mut self, bytes: &[u8]) {
        let (emg, digits, gestures, digit_mode) = (&mut self.emg, &mut self.digits, &mut self.gestures, self.digit_mode);
        self.parser.push(bytes, |s| {
            emg.process(s);
            if digit_mode {
                digits.process(&emg.envelopes(), &emg.contact());
            } else {
                gestures.process(&emg.envelopes(), &emg.contact());
            }
        });
    }

    /// Call when (re)connecting: clears parser and filter state, keeps calibration.
    pub fn reset_stream(&mut self) {
        self.parser.reset();
        self.emg.reset_filters();
        self.digits.reset_state();
        self.gestures.reset_state();
    }

    /// Both schemes' calibrations (not the panel settings), as bytes to store.
    pub fn export_calibration(&self) -> Vec<u8> {
        let saved = SavedCalibration {
            version: CALIBRATION_VERSION,
            muscle: self.gestures.calibration(),
            digit: self.digits.calibration(),
        };
        bincode::serialize(&saved).expect("calibration serializes")
    }

    /// Restores what `export_calibration` saved, replacing the current calibration.
    pub fn import_calibration(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        let err = |e: bincode::Error| JsValue::from_str(&e.to_string());
        let version = bincode::deserialize::<u32>(bytes).map_err(err)?;
        match version {
            CALIBRATION_VERSION => {
                let saved: SavedCalibration = bincode::deserialize(bytes).map_err(err)?;
                self.gestures.restore(saved.muscle);
                self.digits.restore(saved.digit);
            }
            // the old finger grid's calibration means nothing to the digit array
            4 => {
                let saved: SavedCalibrationV4 = bincode::deserialize(bytes).map_err(err)?;
                self.gestures.restore(saved.muscle);
            }
            _ => {
                return Err(JsValue::from_str(&format!(
                    "saved calibration is format {version}, this build reads format {CALIBRATION_VERSION}"
                )))
            }
        }
        Ok(())
    }

    /// Format of `export_calibration`.
    pub fn calibration_version(&self) -> u32 {
        CALIBRATION_VERSION
    }

    /// Ease idle joints back to the home pose under EMG control.
    pub fn set_drift(&mut self, on: bool) {
        self.drift = on;
    }

    /// [packets, dropped, resyncs]
    pub fn stats(&self) -> Vec<f64> {
        vec![self.parser.packets as f64, self.parser.dropped as f64, self.parser.resyncs as f64]
    }

    /// Envelope per channel in µV.
    pub fn envelopes(&self) -> Vec<f32> {
        self.emg.envelopes().to_vec()
    }

    /// Activation per channel for its meter (0 = rest, 1 = a strong contraction):
    /// against the current scheme's calibration once there is one, otherwise default levels.
    pub fn activations(&self) -> Vec<f32> {
        match active!(self, c => c.channel_activation()) {
            Some(a) => a.to_vec(),
            None => self.emg.activations().to_vec(),
        }
    }

    /// Per channel: 1 if the electrode has contact, 0 while its input is railed.
    pub fn contact(&self) -> Vec<u8> {
        self.emg.contact().map(u8::from).to_vec()
    }

    /// Common-average reference: on for single-row wiring with a shared REF.
    /// Choosing a scheme sets it (on for muscle pairs, off for the bipolar digit array).
    pub fn set_car(&mut self, on: bool) {
        self.emg.car = on;
    }

    pub fn car(&self) -> bool {
        self.emg.car
    }

    /// Axis speed for a movement held at its level (both schemes).
    pub fn set_gain(&mut self, value: f32) {
        let gain = value.clamp(0.05, 3.0);
        self.gestures.gain = gain;
        self.digits.gain = gain;
    }

    pub fn set_mains(&mut self, hz: f32) {
        self.emg.set_mains(hz);
    }

    // ---- scheme -------------------------------------------------------------

    /// Digit array (on: eight bipolar pairs, no common-average reference) or
    /// muscle pairs (off: single-row cable with a shared REF, common average on).
    pub fn set_digit_mode(&mut self, on: bool) {
        if on != self.digit_mode {
            self.digits.cancel();
            self.digits.reset_state();
            self.gestures.cancel();
            self.gestures.reset_state();
        }
        self.digit_mode = on;
        self.emg.car = !on;
    }

    pub fn digit_mode(&self) -> bool {
        self.digit_mode
    }

    // ---- the current scheme's movements ----------------------------------------
    // Every `gesture_*` call works on the current scheme: the seven muscle-pair
    // gestures, or the eight digit-array movements (see gesture.rs for both).


    /// Phase 1 for every gesture (rest in three postures first), then phase 2 if `train_after`.
    pub fn gesture_calibrate(&mut self, train_after: bool) {
        active!(mut self, c => c.start_calibration(train_after));
    }

    /// Phase 1 for one gesture (keeps the rest), then its training if `train_after`.
    pub fn gesture_calibrate_one(&mut self, gesture: usize, train_after: bool) {
        active!(mut self, c => c.start_gesture(gesture, train_after));
    }

    /// Phase 2 only, for every calibrated gesture.
    pub fn gesture_train(&mut self) {
        active!(mut self, c => c.start_training());
    }

    /// Cancels a run of either scheme.
    pub fn gesture_cancel(&mut self) {
        self.gestures.cancel();
        self.digits.cancel();
    }

    pub fn gesture_skip(&mut self) {
        active!(mut self, c => c.skip());
    }

    /// The user is in position: start the current step's countdown.
    pub fn gesture_ready(&mut self) {
        active!(mut self, c => c.ready());
    }

    /// Some gesture is calibrated (the decoder runs).
    pub fn gesture_trained(&self) -> bool {
        active!(self, c => c.trained())
    }

    /// Every gesture is calibrated.
    pub fn gesture_calibrated(&self) -> bool {
        active!(self, c => c.calibrated())
    }

    /// Empty when idle, otherwise
    /// [0 step (0 rest, 1 hard, 2 medium, 3 faint, 4 train), 1 movement (-1 = none),
    ///  2 phase (0 prepare, 1 record / trial, 2 relax), 3 phase progress 0..1, 4 seconds left,
    ///  5 check (-1 none, 0 clean, 1 no contact, 2 moving, 3 too weak, 4 too strong,
    ///    5 another movement, 6 not relaxed, 7 above the faint box), 6 check argument
    ///    (channel / movement, -1 = n/a),
    ///  7 seconds without progress, 8 meter kind (0 activity, 1 fraction of hard,
    ///    2 power, 3 stillness), 9 meter, 10 target low, 11 target high (NaN = none),
    ///  12 segment, 13 segments, 14 round, 15 rounds (0 = n/a), 16 goal (3 while training, else -1),
    ///  17 training aim: the level to get as close to as possible (NaN otherwise),
    ///  18 rest posture (0 limp, 1 forearm raised, 2 arm unsupported; -1 = not a rest step),
    ///  19 waiting for the user to say they're ready (1) before the countdown].
    pub fn gesture_status(&self) -> Vec<f32> {
        use gesture::{Check as C, Meter as M, Step as S};
        let Some(s) = active!(self, c => c.status()) else { return Vec::new() };
        let (step, g, posture) = match s.step {
            S::Rest(p) => (0, -1, p as i32),
            S::Hard(g) => (1, g as i32, -1),
            S::Medium(g) => (2, g as i32, -1),
            S::Faint(g) => (3, g as i32, -1),
            S::Train(g) => (4, g as i32, -1),
        };
        let phase = match s.phase {
            gesture::Phase::Prepare => 0,
            gesture::Phase::Record => 1,
            gesture::Phase::Relax => 2,
        };
        let (check, arg) = match s.check {
            None => (-1, -1),
            Some(C::Clean) => (0, -1),
            Some(C::NoContact(c)) => (1, c as i32),
            Some(C::Moving(c)) => (2, c as i32),
            Some(C::TooWeak) => (3, -1),
            Some(C::TooStrong) => (4, -1),
            Some(C::OtherGesture(h)) => (5, h as i32),
            Some(C::NotRelaxed(c)) => (6, c as i32),
            Some(C::Fainter) => (7, -1),
        };
        let meter = match s.meter_kind {
            M::Activity => 0,
            M::OfHard => 1,
            M::Power => 2,
            M::Stillness => 3,
        };
        coach_status(CoachStatus {
            step,
            pattern: g,
            phase,
            progress: s.phase_progress,
            left: s.seconds_left,
            check,
            arg,
            stuck: s.stuck_seconds,
            meter_kind: meter,
            meter: s.meter,
            target: s.target,
            segment: s.segment,
            segments: s.segments,
            round: s.round,
            rounds: s.rounds,
            goal: if step == 4 { 3 } else { -1 },
            aim: s.aim,
            posture,
            waiting: s.waiting,
        })
    }

    /// Current arm posture as the share of each recorded rest posture; empty before calibration.
    pub fn gesture_posture(&self) -> Vec<f32> {
        active!(self, c => c.posture()).iter().map(|&m| m as f32).collect()
    }

    /// Decoded power per gesture (0 = rest, 1 = ceiling).
    pub fn gesture_powers(&self) -> Vec<f32> {
        active!(self, c => c.powers().iter().map(|&p| p as f32).collect())
    }

    /// Per gesture: 1 while it drives its axis.
    pub fn gesture_active(&self) -> Vec<u8> {
        active!(self, c => c.active().map(u8::from).to_vec())
    }

    /// Axis commands (-1..1) the current scheme sends: [base, reach, elbow, gripper].
    pub fn gesture_axis_inputs(&self) -> Vec<f32> {
        active!(self, c => c.axis_inputs()).to_vec()
    }

    /// Per gesture: [stasis, rest exit, level (full speed)].
    pub fn gesture_bands(&self) -> Vec<f32> {
        active!(self, c => (0..c.axes().len())
            .flat_map(|g| {
                let b = c.bands(g);
                [b.stasis, b.rest_exit, b.level].map(|v| v as f32)
            })
            .collect())
    }

    /// Per gesture: [calibrated, faint level, isolation, hits, trials, level, stasis, mixed].
    pub fn gesture_report(&self) -> Vec<f32> {
        let report = active!(self, c => c.report().to_vec());
        report
            .iter()
            .flat_map(|r| {
                [
                    f32::from(u8::from(r.calibrated)),
                    r.faint as f32,
                    r.isolation as f32,
                    r.hits as f32,
                    r.trials as f32,
                    r.level as f32,
                    r.stasis as f32,
                    f32::from(u8::from(r.mixed)),
                ]
            })
            .collect()
    }

    /// Per movement of a scheme (digit array or muscle pairs): [axis (0 base,
    /// 1 reach, 2 elbow, 3 gripper), direction ±1].
    pub fn scheme_axes(&self, digit: bool) -> Vec<f32> {
        let axes: Vec<(usize, f32)> = if digit { gesture::DIGIT_AXES.to_vec() } else { gesture::AXES.to_vec() };
        axes.iter().flat_map(|&(a, d)| [a as f32, d]).collect()
    }

    /// Per channel: the gesture whose pattern leans on it most, -1 if none.
    pub fn channel_gestures(&self) -> Vec<i32> {
        active!(self, c => c.channel_gestures()).iter().map(|g| g.map_or(-1, |g| g as i32)).collect()
    }
}

/// One calibration-coach status, as `finger_status` lays it out.
struct CoachStatus {
    step: i32,
    pattern: i32,
    phase: i32,
    progress: f32,
    left: f32,
    check: i32,
    arg: i32,
    stuck: f32,
    meter_kind: i32,
    meter: f64,
    target: (f64, f64),
    segment: usize,
    segments: usize,
    round: usize,
    rounds: usize,
    goal: i32,
    aim: f64,
    posture: i32,
    waiting: bool,
}

fn coach_status(s: CoachStatus) -> Vec<f32> {
    vec![
        s.step as f32,
        s.pattern as f32,
        s.phase as f32,
        s.progress,
        s.left,
        s.check as f32,
        s.arg as f32,
        s.stuck,
        s.meter_kind as f32,
        s.meter as f32,
        s.target.0 as f32,
        s.target.1 as f32,
        s.segment as f32,
        s.segments as f32,
        s.round as f32,
        s.rounds as f32,
        s.goal as f32,
        s.aim as f32,
        s.posture as f32,
        f32::from(u8::from(s.waiting)),
    ]
}
