//! The workspace in front of the arm: a table and a small box. The claw closes
//! by itself when its fingertips reach the box, carries it, and lets go once the
//! box has been lifted and is set back down on the table or the floor.

use glam::{Mat4, Vec3};

/// Table footprint centre (on the floor) and size; the top is at TABLE_TOP.
pub const TABLE_CENTER: Vec3 = Vec3::new(0.62, 0.0, 0.0);
pub const TABLE_HALF: (f32, f32) = (0.25, 0.35); // half extents in x and z
pub const TABLE_TOP: f32 = 0.45;
pub const BOX_SIZE: f32 = 0.05;
const BOX_START: Vec3 = Vec3::new(0.55, TABLE_TOP + BOX_SIZE / 2.0, 0.18);

/// Fingertips this close to the box centre grab it.
const GRAB_DIST: f32 = 0.045;
/// After letting go, the hand must move this far from the box before it can grab again.
const REARM_DIST: f32 = 0.09;
/// The box must be lifted this far clear of what it stood on before setting it down releases it.
const LIFT_CLEAR: f32 = 0.01;
/// Contact tolerance when setting the box down.
const TOUCH_EPS: f32 = 0.002;
/// Claw open/close speed (full travel per second).
const GRIP_SPEED: f32 = 4.0;

pub struct World {
    /// Box pose while it rests somewhere (upright, rotated about Y only).
    box_pose: Mat4,
    /// While held: the box pose relative to the tool frame, and whether it has been lifted yet.
    held: Option<(Mat4, bool)>,
    armed: bool,
    /// 0 = claw open, 1 = closed.
    grip: f32,
}

impl World {
    pub fn new() -> Self {
        Self { box_pose: Mat4::from_translation(BOX_START), held: None, armed: true, grip: 0.0 }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn grip(&self) -> f32 {
        self.grip
    }

    pub fn holding(&self) -> bool {
        self.held.is_some()
    }

    /// Box pose in world space (unit cube scaled by BOX_SIZE).
    pub fn box_pose(&self, tool: Mat4) -> Mat4 {
        match self.held {
            Some((rel, _)) => tool * rel,
            None => self.box_pose,
        }
    }

    /// Height of whatever is under a point: the table top or the floor.
    fn support(p: Vec3) -> f32 {
        let on_table = (p.x - TABLE_CENTER.x).abs() <= TABLE_HALF.0 && (p.z - TABLE_CENTER.z).abs() <= TABLE_HALF.1;
        if on_table { TABLE_TOP } else { 0.0 }
    }

    /// How far the box's lowest corner is above its support (negative = into it).
    fn clearance(pose: Mat4) -> f32 {
        let h = BOX_SIZE / 2.0;
        let lowest = [-h, h]
            .iter()
            .flat_map(|&x| [-h, h].map(move |y| (x, y)))
            .flat_map(|(x, y)| [-h, h].map(move |z| Vec3::new(x, y, z)))
            .map(|c| pose.transform_point3(c).y)
            .fold(f32::INFINITY, f32::min);
        lowest - Self::support(pose.transform_point3(Vec3::ZERO))
    }

    /// `tool` is the claw's tool frame (origin where the fingertips meet).
    pub fn step(&mut self, dt: f32, tool: Mat4) {
        let tip = tool.transform_point3(Vec3::ZERO);
        match self.held {
            Some((rel, lifted)) => {
                let pose = tool * rel;
                let clear = Self::clearance(pose);
                if !lifted && clear > LIFT_CLEAR {
                    self.held = Some((rel, true));
                } else if lifted && clear <= TOUCH_EPS {
                    // set down: rest it upright on the surface, keeping its heading
                    let c = pose.transform_point3(Vec3::ZERO);
                    let x = pose.x_axis.truncate();
                    let yaw = (-x.z).atan2(x.x);
                    let rest = Vec3::new(c.x, Self::support(c) + BOX_SIZE / 2.0, c.z);
                    self.box_pose = Mat4::from_translation(rest) * Mat4::from_rotation_y(yaw);
                    self.held = None;
                    self.armed = false;
                }
            }
            None => {
                let d = tip.distance(self.box_pose.transform_point3(Vec3::ZERO));
                if !self.armed && d > REARM_DIST {
                    self.armed = true;
                }
                if self.armed && d < GRAB_DIST {
                    self.held = Some((tool.inverse() * self.box_pose, false));
                }
            }
        }
        let target = if self.held.is_some() { 1.0 } else { 0.0 };
        let step = GRIP_SPEED * dt;
        self.grip += (target - self.grip).clamp(-step, step);
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
    fn grabs_carries_and_sets_down() {
        let mut w = World::new();
        let dt = 0.016;
        // far away: nothing happens
        w.step(dt, tool_at(BOX_START + Vec3::Y * 0.3));
        assert!(!w.holding());
        // fingertips reach the box: it is grabbed, without being released on the spot
        let grab = tool_at(BOX_START + Vec3::Y * 0.01);
        for _ in 0..10 {
            w.step(dt, grab);
        }
        assert!(w.holding());
        assert!(w.grip() > 0.5);
        // lift and carry it over the floor beside the table
        let up = tool_at(BOX_START + Vec3::new(0.0, 0.2, 0.0));
        w.step(dt, up);
        assert!((box_center(&w, up).y - (BOX_START.y + 0.19)).abs() < 1e-4);
        let over_floor = Vec3::new(0.6, 0.3, 0.6);
        w.step(dt, tool_at(over_floor));
        assert!(w.holding());
        // lower it until it touches the floor: released upright on the floor
        let mut y = 0.3;
        while w.holding() && y > -0.2 {
            y -= 0.002;
            w.step(dt, tool_at(Vec3::new(0.6, y, 0.6)));
        }
        assert!(!w.holding());
        let c = box_center(&w, Mat4::IDENTITY);
        assert!((c.y - BOX_SIZE / 2.0).abs() < 1e-4, "{c}");
        assert!((c.x - 0.6).abs() < 0.01 && (c.z - 0.6).abs() < 0.01);
    }

    #[test]
    fn needs_to_move_away_before_grabbing_again() {
        let mut w = World::new();
        let dt = 0.016;
        let above = tool_at(BOX_START + Vec3::Y * 0.01);
        w.step(dt, above);
        let lifted = tool_at(BOX_START + Vec3::Y * 0.1);
        w.step(dt, lifted);
        w.step(dt, tool_at(BOX_START + Vec3::Y * 0.005)); // set it straight back down
        assert!(!w.holding());
        for _ in 0..30 {
            w.step(dt, tool_at(box_center(&w, Mat4::IDENTITY) + Vec3::Y * 0.01));
        }
        assert!(!w.holding(), "must not re-grab while the hand is still there");
        assert!(w.grip() < 0.01, "claw opened again");
        w.step(dt, tool_at(box_center(&w, Mat4::IDENTITY) + Vec3::Y * 0.2));
        w.step(dt, tool_at(box_center(&w, Mat4::IDENTITY) + Vec3::Y * 0.01));
        assert!(w.holding());
    }
}
