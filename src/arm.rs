//! 4-axis wall-mounted arm: joint state, velocity integration and forward kinematics.
//!
//! Convention: Y is up, metres and radians. The wall is the plane x = 0 and the
//! arm sticks straight out along +X when every joint is at zero. The base is a
//! two-axis gimbal: yaw about the vertical (left/right), then pitch about the
//! horizontal (up/down). Elbow and wrist are further pitch joints in the arm's
//! plane. Every link extends along the +Y axis of its joint frame and pitch
//! joints rotate about the local Z axis; positive pitch bends upwards.

use std::f32::consts::FRAC_PI_2;

use glam::{Mat4, Vec3};

pub const NUM_JOINTS: usize = 4;

/// Height of the wall mount above the floor.
pub const MOUNT_H: f32 = 0.85;
/// Distance from the wall to the vertical yaw axis.
pub const YAW_X: f32 = 0.085;
/// Distance from the yaw axis to the horizontal pitch axis.
pub const PITCH_OFF: f32 = 0.12;
pub const UPPER_L: f32 = 0.35;
pub const FORE_L: f32 = 0.30;
/// Wrist axis to where the claw's fingertips meet (the tool point).
pub const WRIST_L: f32 = 0.13;

/// Time constant (s) of the first-order velocity smoothing.
const VEL_TAU: f32 = 0.05;

pub struct JointSpec {
    pub name: &'static str,
    pub min: f32,
    pub max: f32,
    /// rad/s at full input.
    pub max_speed: f32,
}

const fn deg(d: f32) -> f32 {
    d * std::f32::consts::PI / 180.0
}

pub const JOINTS: [JointSpec; NUM_JOINTS] = [
    // ±80° keeps the arm from swinging flat against the wall.
    JointSpec { name: "Base yaw", min: deg(-80.0), max: deg(80.0), max_speed: deg(60.0) },
    JointSpec { name: "Base pitch", min: deg(-75.0), max: deg(75.0), max_speed: deg(50.0) },
    JointSpec { name: "Elbow", min: deg(-140.0), max: deg(140.0), max_speed: deg(75.0) },
    JointSpec { name: "Wrist", min: deg(-110.0), max: deg(110.0), max_speed: deg(120.0) },
];

const HOME: [f32; NUM_JOINTS] = [deg(0.0), deg(25.0), deg(-60.0), deg(-35.0)];

/// World-space frames of each joint (origin at the joint, link along +Y).
pub struct Frames {
    /// Vertical yaw axis at the wall (frame Y is vertical, X points out of the wall at q0 = 0).
    pub yaw: Mat4,
    pub pitch: Mat4,
    pub elbow: Mat4,
    pub wrist: Mat4,
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

    pub fn velocity(&self, joint: usize) -> f32 {
        self.vel[joint]
    }

    pub fn step(&mut self, dt: f32) {
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
    }

    pub fn frames(&self) -> Frames {
        let [q0, q1, q2, q3] = self.q;
        let yaw = Mat4::from_translation(Vec3::new(YAW_X, MOUNT_H, 0.0)) * Mat4::from_rotation_y(q0);
        // The extra -90° turns the link axis (+Y) to point out of the wall at q1 = 0.
        let pitch = yaw * Mat4::from_translation(Vec3::X * PITCH_OFF) * Mat4::from_rotation_z(q1 - FRAC_PI_2);
        let elbow = pitch * Mat4::from_translation(Vec3::Y * UPPER_L) * Mat4::from_rotation_z(q2);
        let wrist = elbow * Mat4::from_translation(Vec3::Y * FORE_L) * Mat4::from_rotation_z(q3);
        let tool = wrist * Mat4::from_translation(Vec3::Y * WRIST_L);
        Frames { yaw, pitch, elbow, wrist, tool }
    }

    pub fn end_effector(&self) -> Vec3 {
        self.frames().tool.transform_point3(Vec3::ZERO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REACH: f32 = UPPER_L + FORE_L + WRIST_L;

    fn ee_at(q: [f32; NUM_JOINTS]) -> Vec3 {
        let mut arm = Arm::new();
        arm.q = q;
        arm.end_effector()
    }

    #[test]
    fn zero_pose_sticks_straight_out_of_the_wall() {
        let ee = ee_at([0.0; NUM_JOINTS]);
        assert!((ee - Vec3::new(YAW_X + PITCH_OFF + REACH, MOUNT_H, 0.0)).length() < 1e-5);
    }

    #[test]
    fn positive_base_pitch_raises_the_arm() {
        let ee = ee_at([0.0, deg(90.0), 0.0, 0.0]);
        assert!((ee - Vec3::new(YAW_X + PITCH_OFF, MOUNT_H + REACH, 0.0)).length() < 1e-5);
    }

    #[test]
    fn base_yaw_swings_parallel_to_the_wall() {
        let ee = ee_at([deg(90.0), 0.0, 0.0, 0.0]);
        assert!((ee - Vec3::new(YAW_X, MOUNT_H, -(PITCH_OFF + REACH))).length() < 1e-5);
    }

    #[test]
    fn elbow_and_wrist_bend_in_the_arm_plane() {
        // Yawed 30°, elbow up 90°: the forearm stays in the vertical plane of the upper arm.
        let ee = ee_at([deg(30.0), 0.0, deg(90.0), 0.0]);
        let out = Vec3::new(deg(30.0).cos(), 0.0, -deg(30.0).sin());
        let expected = Vec3::new(YAW_X, MOUNT_H, 0.0) + out * (PITCH_OFF + UPPER_L) + Vec3::Y * (FORE_L + WRIST_L);
        assert!((ee - expected).length() < 1e-5);
    }

    #[test]
    fn box_on_the_table_is_reachable_claw_down() {
        let target = crate::world::World::new().box_pose(Mat4::IDENTITY).transform_point3(Vec3::ZERO);
        let range = |j: usize| {
            let (lo, hi) = (JOINTS[j].min.to_degrees() as i32, JOINTS[j].max.to_degrees() as i32);
            (lo..=hi).step_by(5).map(|d| deg(d as f32))
        };
        let mut arm = Arm::new();
        let mut best = f32::INFINITY;
        for q0 in range(0) {
            for q1 in range(1) {
                for q2 in range(2) {
                    for q3 in range(3) {
                        arm.q = [q0, q1, q2, q3];
                        let tool = arm.frames().tool;
                        let down = -tool.y_axis.truncate().y; // claw pointing down = 1
                        if down > 0.8 {
                            best = best.min(tool.transform_point3(Vec3::ZERO).distance(target));
                        }
                    }
                }
            }
        }
        // well inside the 4.5 cm grab distance even on a 5° grid
        assert!(best < 0.025, "closest claw-down approach to the box: {best} m");
    }

    #[test]
    fn joints_stop_at_limits() {
        let mut arm = Arm::new();
        arm.set_input(1, 1.0);
        for _ in 0..1000 {
            arm.step(0.016);
        }
        assert_eq!(arm.angle(1), JOINTS[1].max);
    }
}
