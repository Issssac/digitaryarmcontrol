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
    finger::{self, Band, FingerControl, NUM_FINGERS},
    gesture::{self, GestureControl, NUM_GESTURES},
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
/// gesture is trained, it closes while you grip and opens when you let go.
const GRIPPER: usize = 3;
const GRIPPER_OPEN: f32 = deg(60.0);
const GRIPPER_CLOSED: f32 = 0.0;
/// Velocity command per radian of gripper error, and the error treated as arrived.
const GRIPPER_GAIN: f32 = 6.0;
const GRIPPER_TOLERANCE: f32 = deg(0.3);

fn band_code(b: Band) -> u8 {
    match b {
        Band::Rest => 0,
        Band::Pending => 1,
        Band::Minus => 2,
        Band::Plus => 3,
        Band::Over => 4,
    }
}

/// Bumped whenever a saved calibration's layout changes, so an old file is
/// refused instead of misread. 2: muscle pairs became gestures, not channels.
/// 3: gestures calibrate like the fingers (rest postures, rounds, training).
/// 4: gestures keep their hard-round samples (for the compound coupling).
const CALIBRATION_VERSION: u32 = 4;

/// Both schemes' calibrations, as saved by `export_calibration`.
#[derive(Serialize, Deserialize)]
struct SavedCalibration {
    version: u32,
    muscle: gesture::Calibration,
    finger: finger::Calibration,
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
    /// Muscle-pair control (when `finger_mode` is off).
    gestures: GestureControl,
    /// Finger-grid control (instead of muscle pairs) when `finger_mode` is on.
    fingers: FingerControl,
    finger_mode: bool,
    drift: bool,
    /// Seconds since any joint last had a command.
    idle: f32,
    /// Angle the gripper is moving to (radians).
    gripper_target: f32,
    /// The joints were set from outside (the real arm) for this frame: no inputs, no drift.
    tracked: bool,
    trail: Vec<Vec3>,
    /// Muscle pairs: turns the reach axis into shoulder + elbow motion.
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
            gestures: GestureControl::new(),
            fingers: {
                // ring + little would drive the gripper, which has its own button
                let mut f = FingerControl::new();
                f.set_joint(3, None);
                f
            },
            finger_mode: false,
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
        // phase 1, the trained finger's joint or gesture's axis in phase 2.
        let mut inputs = match (self.use_emg, self.finger_mode) {
            (false, _) => self.manual,
            (true, true) => self.fingers.joint_inputs(),
            // muscle pairs drive axes (base, reach, elbow, gripper), not joints
            (true, false) => self.reach.joint_inputs(&self.arm, self.gestures.axis_inputs(), dt),
        };
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
        // and opens when it lets go (the finger grid leaves it to the buttons)
        if self.use_emg && !self.finger_mode && self.gestures.drives(GRIPPER) {
            self.gripper_target = if self.gestures.axis_inputs()[GRIPPER] < 0.0 { GRIPPER_CLOSED } else { GRIPPER_OPEN };
        }
        let error = self.gripper_target - self.arm.angle(GRIPPER);
        inputs[GRIPPER] = if error.abs() < GRIPPER_TOLERANCE {
            0.0
        } else {
            (error * GRIPPER_GAIN / JOINTS[GRIPPER].max_speed).clamp(-1.0, 1.0)
        };
        for (j, v) in inputs.into_iter().enumerate() {
            self.arm.set_input(j, v);
        }
        self.arm.step(dt);
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

    /// Manual velocity command (-1..1), used while EMG control is off.
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
        let (emg, fingers, gestures, finger_mode) = (&mut self.emg, &mut self.fingers, &mut self.gestures, self.finger_mode);
        self.parser.push(bytes, |s| {
            emg.process(s);
            if finger_mode {
                fingers.process(&emg.envelopes(), &emg.contact());
            } else {
                gestures.process(&emg.envelopes(), &emg.contact());
            }
        });
    }

    /// Call when (re)connecting: clears parser and filter state, keeps calibration.
    pub fn reset_stream(&mut self) {
        self.parser.reset();
        self.emg.reset_filters();
        self.fingers.reset_state();
        self.gestures.reset_state();
    }

    /// Both schemes' calibrations (not the panel settings), as bytes to store.
    pub fn export_calibration(&self) -> Vec<u8> {
        let saved = SavedCalibration {
            version: CALIBRATION_VERSION,
            muscle: self.gestures.calibration(),
            finger: self.fingers.calibration(),
        };
        bincode::serialize(&saved).expect("calibration serializes")
    }

    /// Restores what `export_calibration` saved, replacing the current calibration.
    pub fn import_calibration(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        let version = bincode::deserialize::<u32>(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
        if version != CALIBRATION_VERSION {
            return Err(JsValue::from_str(&format!(
                "saved calibration is format {version}, this build reads format {CALIBRATION_VERSION}"
            )));
        }
        let saved: SavedCalibration = bincode::deserialize(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.gestures.restore(saved.muscle);
        self.fingers.restore(saved.finger);
        Ok(())
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
    /// against the muscle-pair calibration once there is one, otherwise default levels.
    pub fn activations(&self) -> Vec<f32> {
        match self.gestures.channel_activation().filter(|_| !self.finger_mode) {
            Some(a) => a.to_vec(),
            None => self.emg.activations().to_vec(),
        }
    }

    /// Per channel: 1 if the electrode has contact, 0 while its input is railed.
    pub fn contact(&self) -> Vec<u8> {
        self.emg.contact().map(u8::from).to_vec()
    }

    /// Common-average reference: on for single-row wiring with a shared REF.
    pub fn set_car(&mut self, on: bool) {
        self.emg.car = on;
    }

    /// Muscle pairs: axis speed for a gesture held at its level.
    pub fn set_gain(&mut self, value: f32) {
        self.gestures.gain = value.clamp(0.05, 3.0);
    }

    pub fn set_mains(&mut self, hz: f32) {
        self.emg.set_mains(hz);
    }

    // ---- finger grid ------------------------------------------------------

    /// Finger-grid control (on) or muscle pairs (off).
    pub fn set_finger_mode(&mut self, on: bool) {
        if on != self.finger_mode {
            self.fingers.cancel();
            self.fingers.reset_state();
            self.gestures.cancel();
            self.gestures.reset_state();
        }
        self.finger_mode = on;
    }

    /// Phase 1 for every finger, then phase 2 if `train_after`.
    pub fn finger_calibrate(&mut self, train_after: bool) {
        self.fingers.start_calibration(train_after);
    }

    /// Phase 1 for one finger (keeps the rest level), then its training if `train_after`.
    pub fn finger_calibrate_one(&mut self, finger: usize, train_after: bool) {
        self.fingers.start_finger(finger, train_after);
    }

    /// Phase 2 only, for every calibrated finger.
    pub fn finger_train(&mut self) {
        self.fingers.start_training();
    }

    pub fn finger_cancel(&mut self) {
        self.fingers.cancel();
    }

    pub fn finger_skip(&mut self) {
        self.fingers.skip();
    }

    pub fn finger_trained(&self) -> bool {
        self.fingers.trained()
    }

    /// Empty when idle, otherwise
    /// [0 step (0 rest, 1 hard, 2 medium, 3 faint, 4 train), 1 finger (-1 = none),
    ///  2 phase (0 prepare, 1 record / trial, 2 relax), 3 phase progress 0..1, 4 seconds left,
    ///  5 check (-1 none, 0 clean, 1 no contact, 2 moving, 3 too weak, 4 too strong,
    ///    5 other finger, 6 not relaxed, 7 above the faint box), 6 check argument
    ///    (channel / finger, -1 = n/a),
    ///  7 seconds without progress, 8 meter kind (0 activity, 1 fraction of hard,
    ///    2 power, 3 stillness), 9 meter, 10 target low, 11 target high (NaN = none),
    ///  12 segment, 13 segments, 14 round, 15 rounds (0 = n/a),
    ///  16 goal band (-1 none, 2 faint −, 3 firmer +),
    ///  17 training aim: the level to get as close to as possible (NaN otherwise),
    ///  18 rest posture (0 limp, 1 forearm raised, 2 arm unsupported; -1 = not a rest step),
    ///  19 waiting for the user to say they're ready (1) before the countdown (gestures only)].
    pub fn finger_status(&self) -> Vec<f32> {
        use finger::{Check as C, Meter as M, Step as S};
        let Some(s) = self.fingers.status() else { return Vec::new() };
        let (step, finger, posture) = match s.step {
            S::Rest(p) => (0, -1, p as i32),
            S::Hard(f) => (1, f as i32, -1),
            S::Medium(f) => (2, f as i32, -1),
            S::Faint(f, _) => (3, f as i32, -1),
            S::Train(f) => (4, f as i32, -1),
        };
        let phase = match s.phase {
            finger::Phase::Prepare => 0,
            finger::Phase::Record => 1,
            finger::Phase::Relax => 2,
        };
        let (check, arg) = match s.check {
            None => (-1, -1),
            Some(C::Clean) => (0, -1),
            Some(C::NoContact(c)) => (1, c as i32),
            Some(C::Moving(c)) => (2, c as i32),
            Some(C::TooWeak) => (3, -1),
            Some(C::TooStrong) => (4, -1),
            Some(C::OtherFinger(g)) => (5, g as i32),
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
            pattern: finger,
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
            goal: s.goal.map_or(-1, |b| i32::from(band_code(b))),
            aim: s.aim,
            posture,
            waiting: false,
        })
    }

    /// Current arm posture as the share of each recorded rest posture
    /// [limp, forearm raised, arm unsupported]; empty before calibration.
    pub fn finger_posture(&self) -> Vec<f32> {
        self.fingers.posture().iter().map(|&m| m as f32).collect()
    }

    /// Decoded power per finger (0 = rest, 1 = ceiling).
    pub fn finger_powers(&self) -> Vec<f32> {
        self.fingers.powers().iter().map(|&p| p as f32).collect()
    }

    /// Per finger: 0 rest, 1 faint (not committed yet), 2 faint −, 3 firmer +, 4 over the ceiling.
    pub fn finger_states(&self) -> Vec<u8> {
        self.fingers.band_states().map(band_code).to_vec()
    }

    /// Per finger: [stasis, rest exit, level (faint target), split, plus enter, plus exit, plus target].
    pub fn finger_bands(&self) -> Vec<f32> {
        (0..NUM_FINGERS)
            .flat_map(|f| {
                let b = self.fingers.bands(f);
                [b.stasis, b.rest_exit, b.level, b.split, b.plus_enter, b.plus_exit, b.plus_target].map(|v| v as f32)
            })
            .collect()
    }

    /// Per finger: [calibrated, faintest level, faint rounds, isolation, hits, trials,
    /// faint target, stasis, firmer target, where the person's faint and firm
    /// average in raw power (0 = not measured), smoothing in ms, faint rounds
    /// that looked like another finger].
    pub fn finger_report(&self) -> Vec<f32> {
        self.fingers
            .report()
            .iter()
            .flat_map(|r| {
                [
                    f32::from(u8::from(r.calibrated)),
                    r.faint as f32,
                    r.rounds as f32,
                    r.isolation as f32,
                    r.hits as f32,
                    r.trials as f32,
                    r.level as f32,
                    r.stasis as f32,
                    r.firm as f32,
                    r.held[0] as f32,
                    r.held[1] as f32,
                    (r.smoothing_s * 1000.0) as f32,
                    r.mixed as f32,
                ]
            })
            .collect()
    }

    /// Joint a finger drives; -1 turns the finger off.
    pub fn set_finger_joint(&mut self, finger: usize, joint: i32) {
        self.fingers.set_joint(finger, usize::try_from(joint).ok());
    }

    /// Joint each finger drives, -1 = off.
    pub fn finger_joints(&self) -> Vec<i32> {
        self.fingers.joints().map(|j| j.map_or(-1, |j| j as i32)).to_vec()
    }

    /// Simple fixed bands (rest < ¼ < − < ½ < +, no ceiling) or the adaptive ones.
    pub fn set_finger_simple_bands(&mut self, on: bool) {
        self.fingers.set_simple_bands(on);
    }

    pub fn finger_simple_bands(&self) -> bool {
        self.fingers.simple_bands
    }

    /// Joint speed (fraction of its max) while a finger holds a band.
    pub fn set_finger_speed(&mut self, value: f32) {
        self.fingers.speed = value.clamp(0.05, 1.0);
    }

    // ---- muscle pairs (gestures) ------------------------------------------------

    /// Phase 1 for every gesture (rest in three postures first), then phase 2 if `train_after`.
    pub fn gesture_calibrate(&mut self, train_after: bool) {
        self.gestures.start_calibration(train_after);
    }

    /// Phase 1 for one gesture (keeps the rest), then its training if `train_after`.
    pub fn gesture_calibrate_one(&mut self, gesture: usize, train_after: bool) {
        self.gestures.start_gesture(gesture, train_after);
    }

    /// Phase 2 only, for every calibrated gesture.
    pub fn gesture_train(&mut self) {
        self.gestures.start_training();
    }

    pub fn gesture_cancel(&mut self) {
        self.gestures.cancel();
    }

    pub fn gesture_skip(&mut self) {
        self.gestures.skip();
    }

    /// The user is in position: start the current step's countdown.
    pub fn gesture_ready(&mut self) {
        self.gestures.ready();
    }

    /// Some gesture is calibrated (the decoder runs).
    pub fn gesture_trained(&self) -> bool {
        self.gestures.trained()
    }

    /// Every gesture is calibrated.
    pub fn gesture_calibrated(&self) -> bool {
        self.gestures.calibrated()
    }

    /// Same layout as `finger_status` (field 1 is the gesture, check 5 another
    /// gesture, goal 3 during training).
    pub fn gesture_status(&self) -> Vec<f32> {
        use gesture::{Check as C, Meter as M, Step as S};
        let Some(s) = self.gestures.status() else { return Vec::new() };
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
        self.gestures.posture().iter().map(|&m| m as f32).collect()
    }

    /// Decoded power per gesture (0 = rest, 1 = ceiling).
    pub fn gesture_powers(&self) -> Vec<f32> {
        self.gestures.powers().iter().map(|&p| p as f32).collect()
    }

    /// Per gesture: 1 while it drives its axis.
    pub fn gesture_active(&self) -> Vec<u8> {
        self.gestures.active().map(u8::from).to_vec()
    }

    /// Per gesture: [stasis, rest exit, level (full speed)].
    pub fn gesture_bands(&self) -> Vec<f32> {
        (0..NUM_GESTURES)
            .flat_map(|g| {
                let b = self.gestures.bands(g);
                [b.stasis, b.rest_exit, b.level].map(|v| v as f32)
            })
            .collect()
    }

    /// Per gesture: [calibrated, faint level, isolation, hits, trials, level, stasis, mixed].
    pub fn gesture_report(&self) -> Vec<f32> {
        self.gestures
            .report()
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

    /// Per gesture: [axis (0 base, 1 reach, 2 elbow, 3 gripper), direction ±1].
    pub fn gesture_axes(&self) -> Vec<f32> {
        gesture::AXES.iter().flat_map(|&(a, d)| [a as f32, d]).collect()
    }

    /// Per channel: the gesture whose pattern leans on it most, -1 if none.
    pub fn channel_gestures(&self) -> Vec<i32> {
        self.gestures.channel_gestures().iter().map(|g| g.map_or(-1, |g| g as i32)).collect()
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
