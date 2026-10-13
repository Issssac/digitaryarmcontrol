//! Muscle-pair control axes → joint commands. The muscle pairs don't drive the
//! four joints directly but four axes: base, reach, elbow and gripper. Reach
//! moves the tool point in a straight line along where the gripper points, by
//! working the shoulder and elbow together.
//!
//! The line is taken where the gripper points when a reach starts, and kept
//! through pauses and reversals, so reaching out and back runs on one axis
//! even though the gripper tilts as the arm extends. A new line is only taken
//! after the elbow has deliberately re-aimed (a brief blip doesn't count), or
//! once something else has moved the tip off the line (reset, drifting home).
//! Turning the base needs no new line: the line lives in the arm's plane.

use glam::Vec2;

use crate::arm::{Arm, JOINTS, NUM_JOINTS};

/// The reach axis takes the shoulder's slot: the shoulder only moves as part of a reach.
pub const REACH: usize = 1;
const SHOULDER: usize = 1;
const ELBOW: usize = 2;

/// Tool-point speed at full reach input (m/s).
const REACH_SPEED: f32 = 0.10;
/// Pull back onto the line (1/s), so small integration errors don't bend the path.
const LINE_HOLD: f32 = 3.0;
/// Elbow aiming for this long without a break re-aims the reach.
const REAIM_S: f32 = 0.25;
/// The tip further than this off the line: it was moved some other way, take a new one.
const LINE_LOST_M: f32 = 0.02;

#[derive(Default)]
pub struct Reach {
    /// (start point, unit direction) of the current reach, in the arm's vertical plane.
    line: Option<(Vec2, Vec2)>,
    /// Seconds the elbow has been aiming without a break.
    aimed: f32,
}

impl Reach {
    /// Joint velocity commands (-1..1) for these axis commands (-1..1), `dt` seconds apart.
    pub fn joint_inputs(&mut self, arm: &Arm, axes: [f32; NUM_JOINTS], dt: f32) -> [f32; NUM_JOINTS] {
        let mut inputs = axes;
        inputs[SHOULDER] = 0.0;
        let (tip, pointing) = arm.planar_tool();
        self.aimed = if axes[ELBOW] != 0.0 { self.aimed + dt } else { 0.0 };
        let lost = self.line.is_some_and(|(start, dir)| {
            let off = tip - start;
            (off - dir * off.dot(dir)).length() > LINE_LOST_M
        });
        if self.aimed >= REAIM_S || lost {
            self.line = None;
        }
        let r = axes[REACH];
        if r == 0.0 {
            return inputs;
        }
        let (start, dir) = *self.line.get_or_insert((tip, pointing));
        if dir == Vec2::ZERO {
            return inputs;
        }

        let off = tip - start;
        let v = dir * (r * REACH_SPEED) - (off - dir * off.dot(dir)) * LINE_HOLD;
        let Some((w1, w2)) = arm.planar_joint_rates(v) else { return inputs };
        let (s1, s2) = (w1 / JOINTS[SHOULDER].max_speed, w2 / JOINTS[ELBOW].max_speed);
        // at a joint limit the other joint alone would bend the path: stop instead
        let at_limit = |j: usize, s: f32| {
            (s < 0.0 && arm.angle(j) <= JOINTS[j].min + 1e-4) || (s > 0.0 && arm.angle(j) >= JOINTS[j].max - 1e-4)
        };
        if at_limit(SHOULDER, s1) || at_limit(ELBOW, s2) {
            return inputs;
        }
        // too fast for a joint: slow both, so the direction holds
        let scale = s1.abs().max(s2.abs()).max(1.0);
        inputs[SHOULDER] = s1 / scale;
        inputs[ELBOW] = (inputs[ELBOW] + s2 / scale).clamp(-1.0, 1.0);
        inputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(arm: &mut Arm, reach: &mut Reach, axes: [f32; NUM_JOINTS], seconds: f32) {
        let dt = 1.0 / 120.0;
        for _ in 0..(seconds / dt) as usize {
            let inputs = reach.joint_inputs(arm, axes, dt);
            for (j, v) in inputs.into_iter().enumerate() {
                arm.set_input(j, v);
            }
            arm.step(dt);
        }
    }

    #[test]
    fn reach_moves_the_tip_straight_along_the_gripper() {
        let mut arm = Arm::new(); // home: forearm level, pointing straight out
        let mut reach = Reach::default();
        let (start, _) = arm.planar_tool();
        run(&mut arm, &mut reach, [0.0, 1.0, 0.0, 0.0], 0.8);
        let (out, _) = arm.planar_tool();
        assert!(out.x - start.x > 0.05, "moved out {}", out.x - start.x);
        assert!((out.y - start.y).abs() < 0.004, "stayed level: {} → {}", start.y, out.y);
        // and back in along the same line
        run(&mut arm, &mut reach, [0.0, -1.0, 0.0, 0.0], 0.8);
        let (back, _) = arm.planar_tool();
        assert!((back - start).length() < 0.01, "{start} → {back}");
    }

    /// Distance of the tip from the line through `start` along `dir`.
    fn off_line(arm: &Arm, start: Vec2, dir: Vec2) -> f32 {
        let off = arm.planar_tool().0 - start;
        (off - dir * off.dot(dir)).length()
    }

    #[test]
    fn reaching_out_and_back_repeatedly_stays_on_one_line() {
        let mut arm = Arm::new();
        // aimed a little down, so the gripper visibly tilts as the arm extends
        arm.set_angles([0.0, 0.0, crate::arm::deg(110.0), 0.0]);
        let mut reach = Reach::default();
        let (start, dir) = arm.planar_tool();
        let mut worst = 0.0f32;
        for stroke in 0..8 {
            let r = if stroke % 2 == 0 { 1.0 } else { -1.0 };
            for _ in 0..60 {
                run(&mut arm, &mut reach, [0.0, r, 0.0, 0.0], 1.0 / 60.0);
                worst = worst.max(off_line(&arm, start, dir));
            }
            // a pause between strokes, and a brief crosstalk blip on the elbow
            run(&mut arm, &mut reach, [0.0, 0.0, 0.0, 0.0], 0.2);
            run(&mut arm, &mut reach, [0.0, r, 0.15, 0.0], 0.08);
        }
        assert!(worst < 0.006, "strayed {worst} m from the first stroke's line");
    }

    #[test]
    fn re_aiming_takes_a_new_line() {
        let mut arm = Arm::new();
        let mut reach = Reach::default();
        // aim down with the elbow, then reach: the tip goes out and down
        run(&mut arm, &mut reach, [0.0, 0.0, 1.0, 0.0], 0.5);
        let (start, dir) = arm.planar_tool();
        assert!(dir.y < -0.3, "aimed down: {dir}");
        run(&mut arm, &mut reach, [0.0, 1.0, 0.0, 0.0], 0.5);
        let (end, _) = arm.planar_tool();
        let moved = (end - start).normalize();
        assert!(moved.dot(dir) > 0.99, "along the aim: {moved} vs {dir}");
    }

    #[test]
    fn base_and_gripper_pass_through_and_the_shoulder_only_reaches() {
        let arm = Arm::new();
        let mut reach = Reach::default();
        let inputs = reach.joint_inputs(&arm, [0.4, 0.0, -0.2, 0.7], 1.0 / 60.0);
        assert_eq!(inputs, [0.4, 0.0, -0.2, 0.7]);
    }
}
