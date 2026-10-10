//! WASM entry point. `App` is the only type exposed to JavaScript.

mod arm;
mod emg;
mod render;
mod world;

use glam::Vec3;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use arm::{Arm, JOINTS, NUM_JOINTS};
use emg::{
    cyton::CytonParser,
    decoder::Label,
    pipeline::{CalStep, Check, EmgPipeline, Phase},
};
use render::{camera::OrbitCamera, Renderer};
use world::World;

const TRAIL_MAX: usize = 1500;
const TRAIL_MIN_STEP: f32 = 0.003;

/// JS encoding of a calibration step: -1 = rest, otherwise the channel.
fn step_code(step: CalStep) -> i32 {
    match step {
        CalStep::Rest => -1,
        CalStep::Max(ch) => ch as i32,
    }
}

/// JS encoding of a decoder class: -1 = rest, otherwise the channel whose
/// movement it is; -2 = none.
fn label_code(label: Option<Label>) -> f32 {
    match label {
        None => -2.0,
        Some(Label::Rest) => -1.0,
        Some(Label::Move(ch)) => ch as f32,
    }
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
    trail: Vec<Vec3>,
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
            trail: Vec::new(),
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
        // Hold the arm still while a calibration is asking for pulses.
        let calibrating = self.emg.calibration_status().is_some();
        let inputs = match (self.use_emg, calibrating) {
            (false, _) => self.manual,
            (true, false) => self.emg.joint_inputs(),
            (true, true) => [0.0; NUM_JOINTS],
        };
        for (j, v) in inputs.into_iter().enumerate() {
            self.arm.set_input(j, v);
        }
        self.arm.step(dt);
        self.world.step(dt, self.arm.frames().tool);

        let ee = self.arm.end_effector();
        if self.trail.last().map_or(true, |p| p.distance(ee) > TRAIL_MIN_STEP) {
            if self.trail.len() >= TRAIL_MAX {
                self.trail.remove(0);
            }
            self.trail.push(ee);
        }

        self.renderer.render(&self.camera, &self.arm, &self.world, &self.trail);
    }

    // ---- arm --------------------------------------------------------------

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
        self.parser.push(bytes, |s| self.emg.process(s));
    }

    /// Call when (re)connecting: clears parser and filter state, keeps calibration.
    pub fn reset_stream(&mut self) {
        self.parser.reset();
        self.emg.reset_filters();
    }

    /// [packets, dropped, resyncs]
    pub fn stats(&self) -> Vec<f64> {
        vec![self.parser.packets as f64, self.parser.dropped as f64, self.parser.resyncs as f64]
    }

    /// Envelope per channel in µV.
    pub fn envelopes(&self) -> Vec<f32> {
        self.emg.envelopes().to_vec()
    }

    /// Normalised activation per channel (0 = rest, 1 = max contraction).
    pub fn activations(&self) -> Vec<f32> {
        self.emg.activations().to_vec()
    }

    pub fn rest_levels(&self) -> Vec<f32> {
        self.emg.rest_levels().to_vec()
    }

    pub fn max_levels(&self) -> Vec<f32> {
        self.emg.max_levels().to_vec()
    }

    /// Per channel: 1 if the electrode has contact, 0 while its input is railed.
    pub fn contact(&self) -> Vec<u8> {
        self.emg.contact().map(u8::from).to_vec()
    }

    /// Per channel: 1 while its electrode is offline and its envelope is
    /// estimated from the other channels (needs a trained decoder).
    pub fn estimated(&self) -> Vec<u8> {
        self.emg.estimated().map(u8::from).to_vec()
    }

    /// Common-average reference: on for single-row wiring with a shared REF.
    pub fn set_car(&mut self, on: bool) {
        self.emg.car = on;
    }

    /// Guided run: rest, then a max contraction for every mapped channel.
    pub fn calibrate_full(&mut self) {
        self.emg.start_full_calibration();
    }

    pub fn calibrate_rest(&mut self) {
        self.emg.start_calibration(vec![CalStep::Rest]);
    }

    pub fn calibrate_max(&mut self, channel: usize) {
        self.emg.start_calibration(vec![CalStep::Max(channel)]);
    }

    pub fn cancel_calibration(&mut self) {
        self.emg.cancel_calibration();
    }

    /// Steps of the current run: -1 = rest, otherwise the channel to contract.
    pub fn calibration_plan(&self) -> Vec<i32> {
        self.emg.calibration_plan().iter().map(|&s| step_code(s)).collect()
    }

    /// Empty when idle, otherwise
    /// [step, steps, channel (-1 = rest), phase (0 prepare, 1 record, 2 relax),
    ///  phase progress 0..1, seconds left in phase, total progress 0..1,
    ///  check (-1 none, 0 clean, 1 no contact, 2 moving, 3 no pulse, 4 not relaxed),
    ///  check channel (-1 = n/a), meter (level / limit), seconds without progress,
    ///  proxy (1 = cued electrode offline, pulse read from the other channels)].
    pub fn calibration_status(&self) -> Vec<f32> {
        let Some(s) = self.emg.calibration_status() else { return Vec::new() };
        let phase = match s.phase {
            Phase::Prepare => 0.0,
            Phase::Record => 1.0,
            Phase::Relax => 2.0,
        };
        let (check, check_ch) = match s.check {
            None => (-1.0, -1.0),
            Some(Check::Clean) => (0.0, -1.0),
            Some(Check::NoContact(ch)) => (1.0, ch as f32),
            Some(Check::Moving(ch)) => (2.0, ch as f32),
            Some(Check::NoPulse) => (3.0, -1.0),
            Some(Check::NotRelaxed(ch)) => (4.0, ch as f32),
        };
        vec![
            s.step as f32,
            s.steps as f32,
            step_code(s.kind) as f32,
            phase,
            s.phase_progress,
            s.seconds_left,
            s.total_progress,
            check,
            check_ch,
            s.meter,
            s.stuck_seconds,
            f32::from(u8::from(s.proxy)),
        ]
    }

    /// Per channel: 1 if its movement was calibrated from the other channels
    /// because its own electrode was offline.
    pub fn calibrated_via_others(&self) -> Vec<u8> {
        self.emg.via_others().map(u8::from).to_vec()
    }

    /// Abandon the current calibration step and continue with the next one.
    pub fn skip_calibration_step(&mut self) {
        self.emg.skip_calibration_step();
    }

    pub fn is_calibrated(&self) -> bool {
        self.emg.is_calibrated()
    }

    /// Pattern decoder (all channels → one movement) vs. direct channel pairs.
    /// Pattern mode falls back to direct until a calibration has trained it.
    pub fn set_pattern_decoder(&mut self, on: bool) {
        self.emg.pattern = on;
    }

    pub fn decoder_trained(&self) -> bool {
        self.emg.decoder_trained()
    }

    /// [relaxed (1/0), then (channel, intensity) for each movement currently
    /// driving the arm: none, one, or two at once].
    pub fn intent(&self) -> Vec<f32> {
        let intent = self.emg.intent();
        std::iter::once(f32::from(u8::from(intent.relaxed)))
            .chain(intent.moves.iter().flat_map(|&(ch, w)| [ch as f32, w]))
            .collect()
    }

    /// Cross-validated recognition per class, flattened:
    /// [class, accuracy 0..1, most-confused-with class (-2 none), …].
    pub fn decoder_report(&self) -> Vec<f32> {
        self.emg
            .decoder_report()
            .iter()
            .flat_map(|r| [label_code(Some(r.label)), r.accuracy as f32, label_code(r.confused_with)])
            .collect()
    }

    /// Per channel: 1 once its max contraction has been calibrated.
    pub fn max_calibrated(&self) -> Vec<u8> {
        self.emg.max_done().map(u8::from).to_vec()
    }

    /// Channel indices are 0-based; pass -1 for "none".
    pub fn set_mapping(&mut self, joint: usize, pos: i32, neg: i32) {
        let ch = |c: i32| usize::try_from(c).ok();
        self.emg.set_mapping(joint, ch(pos), ch(neg));
    }

    /// [pos0, neg0, pos1, neg1, …], -1 for "none".
    pub fn mapping(&self) -> Vec<i32> {
        let ch = |c: Option<usize>| c.map_or(-1, |c| c as i32);
        self.emg.mapping().iter().flat_map(|&(p, n)| [ch(p), ch(n)]).collect()
    }

    pub fn set_threshold(&mut self, value: f32) {
        self.emg.threshold = value;
    }

    pub fn set_gain(&mut self, value: f32) {
        self.emg.gain = value;
    }

    pub fn set_mains(&mut self, hz: f32) {
        self.emg.set_mains(hz);
    }
}
