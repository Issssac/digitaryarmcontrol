//! The desk the arm stands on (the plane y = 0) and a small box on it. Opening
//! the gripper and closing it with the tool point at the box picks the box up;
//! opening it again drops the box onto the desk below.

use glam::{Mat4, Vec3};

use crate::arm::deg;

pub const BOX_SIZE: f32 = 0.03;
const BOX_START: Vec3 = Vec3::new(0.28, BOX_SIZE / 2.0, 0.08);

/// The tool point this close to the box centre, as the jaw closes, grabs it.
const GRAB_DIST: f32 = 0.03;
/// Jaw angle at or below which it counts as closed (grabs), and above which it lets go.
const CLOSED: f32 = deg(15.0);
const OPEN: f32 = deg(25.0);

pub struct World {
    /// Box pose while it rests on the desk (upright, rotated about Y only).
    box_pose: Mat4,
    /// While held: the box pose relative to the tool frame.
    held: Option<Mat4>,
    /// The jaw has opened since it last closed away from the box, so closing it now can grab.
    armed: bool,
}

impl World {
    pub fn new() -> Self {
        Self { box_pose: Mat4::from_translation(BOX_START), held: None, armed: false }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn holding(&self) -> bool {
        self.held.is_some()
    }

    /// Box pose in world space (unit cube scaled by BOX_SIZE).
    pub fn box_pose(&self, tool: Mat4) -> Mat4 {
        match self.held {
            Some(rel) => tool * rel,
            None => self.box_pose,
        }
    }

    /// `tool` is the tool frame (origin between the jaw tips), `gripper` the jaw angle.
    pub fn step(&mut self, tool: Mat4, gripper: f32) {
        match self.held {
            Some(rel) if gripper > OPEN => {
                // dropped: lands upright on the desk below, keeping its heading
                let pose = tool * rel;
                let c = pose.transform_point3(Vec3::ZERO);
                let x = pose.x_axis.truncate();
                let yaw = (-x.z).atan2(x.x);
                self.box_pose = Mat4::from_translation(Vec3::new(c.x, BOX_SIZE / 2.0, c.z)) * Mat4::from_rotation_y(yaw);
                self.held = None;
                self.armed = true;
            }
            Some(_) => {}
            None if gripper > OPEN => self.armed = true,
            None if gripper <= CLOSED => {
                let near = tool.transform_point3(Vec3::ZERO).distance(self.box_pose.transform_point3(Vec3::ZERO)) < GRAB_DIST;
                if self.armed && near {
                    self.held = Some(tool.inverse() * self.box_pose);
                }
                self.armed = false;
            }
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tool frame pointing straight down with its tip at `p`.
    fn tool_at(p: Vec3) -> Mat4 {
        Mat4::from_translation(p) * Mat4::from_rotation_x(std::f32::consts::PI)
    }

    fn box_center(w: &World, tool: Mat4) -> Vec3 {
        w.box_pose(tool).transform_point3(Vec3::ZERO)
    }

    #[test]
    fn open_close_on_the_box_carries_it_and_opening_drops_it() {
        let mut w = World::new();
        let at_box = tool_at(BOX_START);
        // starts closed: touching the box doesn't grab it
        w.step(at_box, 0.0);
        assert!(!w.holding());
        // open, then close on it
        w.step(at_box, deg(40.0));
        w.step(at_box, deg(5.0));
        assert!(w.holding());
        // carry it up and across
        let up = tool_at(Vec3::new(0.2, 0.2, -0.1));
        w.step(up, deg(5.0));
        assert!((box_center(&w, up) - Vec3::new(0.2, 0.2, -0.1)).length() < 1e-4);
        // opening drops it onto the desk below
        w.step(up, deg(40.0));
        assert!(!w.holding());
        let c = box_center(&w, Mat4::IDENTITY);
        assert!((c - Vec3::new(0.2, BOX_SIZE / 2.0, -0.1)).length() < 1e-4, "{c}");
    }

    #[test]
    fn a_jaw_closed_elsewhere_doesnt_grab_when_it_bumps_the_box() {
        let mut w = World::new();
        w.step(tool_at(Vec3::new(0.0, 0.3, 0.0)), deg(40.0));
        w.step(tool_at(Vec3::new(0.0, 0.3, 0.0)), deg(5.0)); // closed in mid-air
        w.step(tool_at(BOX_START), deg(5.0));
        assert!(!w.holding());
        // half-open (between the thresholds) and closing again still doesn't count
        w.step(tool_at(BOX_START), deg(20.0));
        w.step(tool_at(BOX_START), deg(5.0));
        assert!(!w.holding());
        w.step(tool_at(BOX_START), deg(30.0));
        w.step(tool_at(BOX_START), deg(10.0));
        assert!(w.holding());
    }
}
