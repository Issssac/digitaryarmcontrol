//! Waveshare RoArm-M2-S: joint state, velocity integration and forward kinematics.
//!
//! The joint chain is Waveshare's own (roarm_description/urdf/roarm_m2.xacro,
//! the same origins roarm-haptic-bridge renders with), and joint angles mean
//! exactly what the real arm reports, so the simulation and the arm can mirror
//! each other 1:1. At all zeros the arm points straight up; positive shoulder
//! leans it forward, elbow 90° holds the forearm level (the arm's start-up
//! pose), and positive base turns it counter-clockwise seen from above. The
//! fourth joint opens the gripper jaw (0 = closed).
//!
//! The URDF is Z-up; the scene is Y-up with X forward: world = Rx(−90°) · URDF.

use std::f32::consts::{FRAC_PI_2, PI};

use glam::{Mat4, Vec2, Vec3};

pub const NUM_JOINTS: usize = 4;

/// Time constant (s) of the first-order velocity smoothing.
const VEL_TAU: f32 = 0.05;

pub struct JointSpec {
    pub name: &'static str,
    pub min: f32,
    pub max: f32,
    /// rad/s at full input.
    pub max_speed: f32,
}

pub const fn deg(d: f32) -> f32 {
    d * PI / 180.0
}

/// Limits as roarm-haptic-bridge enforces them on the real arm.
pub const JOINTS: [JointSpec; NUM_JOINTS] = [
    JointSpec { name: "Base", min: deg(-180.0), max: deg(180.0), max_speed: deg(60.0) },
    JointSpec { name: "Shoulder", min: deg(-90.0), max: deg(90.0), max_speed: deg(45.0) },
    JointSpec { name: "Elbow", min: deg(-20.0), max: deg(169.0), max_speed: deg(60.0) },
    JointSpec { name: "Gripper", min: deg(-10.0), max: deg(90.0), max_speed: deg(90.0) },
];

/// The arm's own start-up pose: upper arm up, forearm level, gripper closed.
const HOME: [f32; NUM_JOINTS] = [0.0, 0.0, deg(90.0), 0.0];

/// The tool point never goes lower than this above the desk the arm stands on:
/// with Follow on, the real arm is sent the same pose.
const MIN_TOOL_HEIGHT: f32 = 0.01;

/// Idle drift back to the home pose: speed proportional to the distance, capped.
const DRIFT_GAIN: f32 = 2.0;
const DRIFT_MAX_SPEED: f32 = deg(6.0);

/// URDF joint origin: translation, then fixed roll-pitch-yaw (Rz · Ry · Rx).
fn origin(xyz: [f32; 3], rpy: [f32; 3]) -> Mat4 {
    Mat4::from_translation(Vec3::from(xyz))
        * Mat4::from_rotation_z(rpy[2])
        * Mat4::from_rotation_y(rpy[1])
        * Mat4::from_rotation_x(rpy[0])
}

/// Tool point (between the jaw tips) in the elbow link's frame, as the bridge places it.
const TCP: Vec3 = Vec3::new(0.002, -0.2802, 0.0);

/// World-space frames of each link (the frame its mesh is modelled in). Every
/// joint turns about its own frame's Z axis.
pub struct Frames {
    pub base: Mat4,
    pub link1: Mat4,
    pub link2: Mat4,
    pub link3: Mat4,
    pub gripper: Mat4,
    /// Tool point between the jaws: the end effector.
    pub tool: Mat4,
}

pub struct Arm {
    q: [f32; NUM_JOINTS],
    vel: [f32; NUM_JOINTS],
    input: [f32; NUM_JOINTS],
}

impl Arm {
    pub fn new() -> Self {
        Self { q: HOME, vel: [0.0; NUM_JOINTS], input: [0.0; NUM_JOINTS] }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Velocity command for a joint, -1..1 of its max speed.
    pub fn set_input(&mut self, joint: usize, value: f32) {
        if let Some(i) = self.input.get_mut(joint) {
            *i = if value.is_finite() { value.clamp(-1.0, 1.0) } else { 0.0 };
        }
    }

    pub fn input(&self, joint: usize) -> f32 {
        self.input[joint]
    }

    pub fn angle(&self, joint: usize) -> f32 {
        self.q[joint]
    }

    /// Puts the joints straight at these angles (radians, clamped), at rest.
    pub fn set_angles(&mut self, q: [f32; NUM_JOINTS]) {
        for (j, spec) in JOINTS.iter().enumerate() {
            if q[j].is_finite() {
                self.q[j] = q[j].clamp(spec.min, spec.max);
            }
        }
        self.vel = [0.0; NUM_JOINTS];
        self.input = [0.0; NUM_JOINTS];
    }

    pub fn velocity(&self, joint: usize) -> f32 {
        self.vel[joint]
    }

    /// Velocity command (-1..1) that eases an idle joint back towards home.
    pub fn drift_input(&self, joint: usize) -> f32 {
        let v = (DRIFT_GAIN * (HOME[joint] - self.q[joint])).clamp(-DRIFT_MAX_SPEED, DRIFT_MAX_SPEED);
        v / JOINTS[joint].max_speed
    }

    pub fn step(&mut self, dt: f32) {
        let before = self.q;
        let height = self.end_effector().y;
        let k = 1.0 - (-dt / VEL_TAU).exp();
        for (j, spec) in JOINTS.iter().enumerate() {
            let target = self.input[j] * spec.max_speed;
            self.vel[j] += (target - self.vel[j]) * k;
            let q = self.q[j] + self.vel[j] * dt;
            self.q[j] = q.clamp(spec.min, spec.max);
            if q != self.q[j] {
                self.vel[j] = 0.0;
            }
        }
        // Shoulder and elbow stop where they would push the tool into the desk.
        let now = self.end_effector().y;
        if now < MIN_TOOL_HEIGHT && now < height {
            for j in [1, 2] {
                self.q[j] = before[j];
                self.vel[j] = 0.0;
            }
        }
    }

    pub fn frames(&self) -> Frames {
        let [q0, q1, q2, q3] = self.q;
        let base = Mat4::from_rotation_x(-FRAC_PI_2);
        let link1 = base * origin([0.0100000009, 0.0, 0.1230592705], [0.0; 3]) * Mat4::from_rotation_z(q0);
        let link2 = link1 * origin([0.0; 3], [-FRAC_PI_2, -FRAC_PI_2, 0.0]) * Mat4::from_rotation_z(q1);
        let link3 = link2 * origin([0.2368151329, 0.0300023995, 0.0], [0.0, 0.0, FRAC_PI_2]) * Mat4::from_rotation_z(q2);
        let gripper =
            link3 * origin([0.002906, -0.21599, -0.00066683], [-FRAC_PI_2, 0.0, -FRAC_PI_2]) * Mat4::from_rotation_z(q3);
        let tool = link3 * Mat4::from_translation(TCP);
        Frames { base, link1, link2, link3, gripper, tool }
    }

    pub fn end_effector(&self) -> Vec3 {
        self.frames().tool.transform_point3(Vec3::ZERO)
    }

    /// The tool point and the way the gripper points, in the arm's vertical
    /// plane: (outward from the base, up).
    pub fn planar_tool(&self) -> (Vec2, Vec2) {
        planar_at(self.q)
    }

    /// Shoulder and elbow rates (rad/s) that move the tool point at `v` (m/s,
    /// in the vertical plane); None near a singularity (arm nearly straight).
    pub fn planar_joint_rates(&self, v: Vec2) -> Option<(f32, f32)> {
        const EPS: f32 = 1e-3;
        let (p, _) = planar_at(self.q);
        let column = |j: usize| {
            let mut q = self.q;
            q[j] += EPS;
            (planar_at(q).0 - p) / EPS
        };
        let (a, b) = (column(1), column(2));
        // |det| ≈ upper arm × forearm × sin(bend): ~0.066 m² at home; below this
        // (within ~9° of straight) the rates blow up
        let det = a.x * b.y - b.x * a.y;
        if det.abs() < 0.01 {
            return None;
        }
        Some(((v.x * b.y - b.x * v.y) / det, (a.x * v.y - v.x * a.y) / det))
    }
}

/// See `Arm::planar_tool`. The forearm (and the gripper on it) points along
/// link3's −Y, from the elbow to the tool point.
fn planar_at(q: [f32; NUM_JOINTS]) -> (Vec2, Vec2) {
    let f = Arm { q, vel: [0.0; NUM_JOINTS], input: [0.0; NUM_JOINTS] }.frames();
    // the horizontal direction the arm faces (+X at base 0, counter-clockwise from above)
    let out = Vec3::new(q[0].cos(), 0.0, -q[0].sin());
    let p = f.tool.transform_point3(Vec3::ZERO);
    let w = f.link3.transform_vector3(Vec3::NEG_Y);
    (Vec2::new(p.dot(out), p.y), Vec2::new(w.dot(out), w.y).normalize_or_zero())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ee_at(q: [f32; NUM_JOINTS]) -> Vec3 {
        let mut arm = Arm::new();
        arm.set_angles(q.map(deg));
        arm.end_effector()
    }

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-3
    }

    #[test]
    fn zero_pose_points_straight_up() {
        // the tool point 0.123 + 0.237 + 0.280 m above the base
        assert!(close(ee_at([0.0; 4]), Vec3::new(0.042, 0.640, 0.0)), "{}", ee_at([0.0; 4]));
    }

    #[test]
    fn home_pose_holds_the_forearm_level() {
        let ee = ee_at([0.0, 0.0, 90.0, 0.0]);
        assert!(close(ee, Vec3::new(0.320, 0.358, 0.0)), "{ee}");
    }

    #[test]
    fn positive_shoulder_leans_forward_and_elbow_bends_down() {
        let home = ee_at([0.0, 0.0, 90.0, 0.0]);
        let leaned = ee_at([0.0, 30.0, 90.0, 0.0]);
        assert!(leaned.x > home.x + 0.05 && leaned.y < home.y - 0.1, "{leaned}");
        let bent = ee_at([0.0, 0.0, 120.0, 0.0]);
        assert!(bent.y < home.y - 0.1, "{bent}");
    }

    #[test]
    fn base_turns_counter_clockwise_about_the_vertical() {
        let home = ee_at([0.0, 0.0, 90.0, 0.0]);
        let ee = ee_at([90.0, 0.0, 90.0, 0.0]);
        // seen from above (Y up, X forward), counter-clockwise takes +X towards −Z
        assert!((ee.y - home.y).abs() < 1e-4 && ee.z < -0.3 && ee.x.abs() < 0.05, "{ee}");
    }

    #[test]
    fn gripper_jaw_turns_about_its_own_axis_and_leaves_the_tool_point() {
        let mut arm = Arm::new();
        let closed = arm.frames().gripper;
        arm.set_angles([0.0, 0.0, deg(90.0), deg(60.0)]);
        let open = arm.frames().gripper;
        assert!(close(closed.w_axis.truncate(), open.w_axis.truncate()));
        assert!(close(closed.z_axis.truncate(), open.z_axis.truncate()));
        assert!((closed.x_axis.truncate().dot(open.x_axis.truncate()) - 0.5).abs() < 1e-4);
        assert_eq!(ee_at([0.0, 0.0, 90.0, 0.0]), ee_at([0.0, 0.0, 90.0, 60.0]));
    }

    #[test]
    fn box_on_the_desk_is_reachable() {
        let target = crate::world::World::new().box_pose(Mat4::IDENTITY).transform_point3(Vec3::ZERO);
        let range = |j: usize| {
            let (lo, hi) = (JOINTS[j].min.to_degrees() as i32, JOINTS[j].max.to_degrees() as i32);
            (lo..=hi).step_by(3).map(|d| d as f32)
        };
        let mut best = f32::INFINITY;
        for q0 in range(0) {
            for q1 in range(1) {
                for q2 in range(2) {
                    best = best.min(ee_at([q0, q1, q2, 0.0]).distance(target));
                }
            }
        }
        // well inside the 3 cm grab distance even on a 3° grid
        assert!(best < 0.015, "closest approach to the box: {best} m");
    }

    #[test]
    fn drift_eases_back_to_home_slowly() {
        let mut arm = Arm::new();
        arm.q[2] = HOME[2] + deg(40.0);
        let start = arm.q[2];
        for _ in 0..60 {
            arm.set_input(2, arm.drift_input(2));
            arm.step(1.0 / 60.0);
        }
        // at most ~6°/s for one second
        let moved = start - arm.q[2];
        assert!(moved > deg(4.0) && moved < deg(6.5), "moved {}°", moved.to_degrees());
        for _ in 0..60 * 30 {
            arm.set_input(2, arm.drift_input(2));
            arm.step(1.0 / 60.0);
        }
        assert!((arm.q[2] - HOME[2]).abs() < deg(0.5));
    }

    #[test]
    fn the_tool_stops_at_the_desk_and_can_still_lift_off() {
        let mut arm = Arm::new();
        arm.set_input(1, 1.0);
        arm.set_input(2, 1.0);
        for _ in 0..600 {
            arm.step(0.016);
            assert!(arm.end_effector().y > MIN_TOOL_HEIGHT - 0.005, "{}", arm.end_effector());
        }
        assert!(arm.end_effector().y < MIN_TOOL_HEIGHT + 0.01, "should have reached the desk");
        arm.set_input(1, -1.0);
        arm.set_input(2, 0.0);
        let low = arm.end_effector().y;
        for _ in 0..30 {
            arm.step(0.016);
        }
        assert!(arm.end_effector().y > low + 0.02);
    }

    #[test]
    fn planar_pose_and_rates_agree_with_the_kinematics() {
        let arm = Arm::new(); // home: forearm level, pointing straight out
        let (p, d) = arm.planar_tool();
        assert!((p - Vec2::new(0.320, 0.358)).length() < 1e-3, "{p}");
        assert!((d - Vec2::X).length() < 1e-3, "{d}");
        // the rates for "straight out at 0.1 m/s", applied briefly, move it that way
        let (w1, w2) = arm.planar_joint_rates(Vec2::new(0.1, 0.0)).unwrap();
        let mut q = arm.q;
        q[1] += w1 * 0.01;
        q[2] += w2 * 0.01;
        let moved = planar_at(q).0 - p;
        assert!((moved - Vec2::new(0.001, 0.0)).length() < 1e-4, "{moved}");
        // arm straight up: singular
        let mut up = Arm::new();
        up.set_angles([0.0; NUM_JOINTS]);
        assert!(up.planar_joint_rates(Vec2::X * 0.1).is_none());
    }

    #[test]
    fn joints_stop_at_limits() {
        let mut arm = Arm::new();
        arm.set_angles([0.0; NUM_JOINTS]); // straight up: leaning all the way forward clears the desk
        arm.set_input(1, 1.0);
        for _ in 0..1000 {
            arm.step(0.016);
        }
        assert_eq!(arm.angle(1), JOINTS[1].max);
        arm.set_angles([0.0, deg(200.0), 0.0, f32::NAN]);
        assert_eq!(arm.angle(1), JOINTS[1].max);
        assert_eq!(arm.angle(3), 0.0);
    }
}
