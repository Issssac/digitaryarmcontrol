//! Turns the arm's kinematic frames into a list of coloured parts: Waveshare's
//! own link meshes once they are loaded, simple primitives along the same
//! frames until then.

use glam::{Mat4, Quat, Vec3};

use crate::arm::{Arm, JOINTS};
use crate::world::{World, BOX_SIZE};

pub type Color = [f32; 4];

pub const BODY: Color = [0.80, 0.82, 0.85, 1.0];
pub const GRAPHITE: Color = [0.17, 0.19, 0.22, 1.0];
pub const ACCENT: Color = [0.20, 0.85, 1.00, 1.0];
pub const JAW: Color = [0.33, 0.89, 0.65, 1.0];
pub const CRATE: Color = [0.95, 0.80, 0.25, 1.0];
pub const CRATE_BAND: Color = [0.70, 0.36, 0.07, 1.0];

/// Link meshes in kinematic order; the page loads `models/<name>.stl` in this order.
pub const LINK_MESHES: [&str; 5] = ["base_link", "link1", "link2", "link3", "gripper_link"];

#[derive(Clone, Copy)]
pub enum Shape {
    Cube,
    Cylinder,
    /// One of LINK_MESHES.
    Link(usize),
}

pub struct Part {
    pub shape: Shape,
    pub model: Mat4,
    pub color: Color,
}

fn mix(a: Color, b: Color, t: f32) -> Color {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

fn block(frame: Mat4, offset: Vec3, size: Vec3, color: Color) -> Part {
    Part { shape: Shape::Cube, model: frame * Mat4::from_scale_rotation_translation(size, Quat::IDENTITY, offset), color }
}

/// Round bar from `a` to `b` (both in `frame` coordinates).
fn strut(frame: Mat4, a: Vec3, b: Vec3, radius: f32, color: Color) -> Part {
    let d = b - a;
    let len = d.length().max(1e-6);
    let rot = Quat::from_rotation_arc(Vec3::Y, d / len);
    let scale = Vec3::new(radius * 2.0, len, radius * 2.0);
    Part { shape: Shape::Cylinder, model: frame * Mat4::from_scale_rotation_translation(scale, rot, (a + b) * 0.5), color }
}

/// The box on the desk (or in the gripper).
pub fn world_parts(world: &World, tool: Mat4) -> Vec<Part> {
    let crate_ = world.box_pose(tool);
    vec![
        block(crate_, Vec3::ZERO, Vec3::splat(BOX_SIZE), CRATE),
        // a darker band so the box's rotation reads
        block(crate_, Vec3::ZERO, Vec3::new(BOX_SIZE * 1.02, BOX_SIZE * 0.25, BOX_SIZE * 1.02), CRATE_BAND),
    ]
}

/// `meshes`: the link meshes are loaded.
pub fn arm_parts(arm: &Arm, meshes: bool) -> Vec<Part> {
    let f = arm.frames();
    // Each link glows with the accent colour in proportion to how fast its joint moves.
    let glow = |j: usize, base: Color| {
        let t = (arm.velocity(j).abs() / JOINTS[j].max_speed).min(1.0);
        mix(base, ACCENT, t * 0.7)
    };
    let frames = [f.base, f.link1, f.link2, f.link3, f.gripper];
    let colors = [GRAPHITE, glow(0, BODY), glow(1, BODY), glow(2, BODY), glow(3, JAW)];
    if meshes {
        return frames
            .iter()
            .zip(colors)
            .enumerate()
            .map(|(i, (&model, color))| Part { shape: Shape::Link(i), model, color })
            .collect();
    }
    // Primitives in the same link frames (URDF axes: the base stands on Z, each joint turns about Z).
    let z_up = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let cyl = |frame: Mat4, offset: Vec3, r: f32, h: f32, color| Part {
        shape: Shape::Cylinder,
        model: frame * Mat4::from_scale_rotation_translation(Vec3::new(2.0 * r, h, 2.0 * r), z_up, offset),
        color,
    };
    vec![
        cyl(f.base, Vec3::Z * 0.042, 0.045, 0.084, colors[0]),
        cyl(f.link1, Vec3::Z * -0.015, 0.032, 0.05, colors[1]),
        cyl(f.link2, Vec3::ZERO, 0.03, 0.07, colors[2]),
        strut(f.link2, Vec3::ZERO, Vec3::new(0.2368, 0.03, 0.0), 0.017, colors[2]),
        cyl(f.link3, Vec3::ZERO, 0.025, 0.06, colors[3]),
        strut(f.link3, Vec3::ZERO, Vec3::new(0.002, -0.2802, 0.0), 0.013, colors[3]),
        block(f.gripper, Vec3::new(0.034, 0.0, 0.0), Vec3::new(0.068, 0.01, 0.03), colors[4]),
    ]
}
