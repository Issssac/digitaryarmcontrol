//! Turns the arm's kinematic frames into a list of coloured primitives.

use std::f32::consts::TAU;

use glam::{Mat4, Quat, Vec3};

use crate::arm::{Arm, FORE_L, JOINTS, MOUNT_H, PITCH_OFF, UPPER_L, YAW_X};
use crate::world::{World, BOX_SIZE, TABLE_CENTER, TABLE_HALF, TABLE_TOP};

pub type Color = [f32; 4];

pub const ORANGE: Color = [0.95, 0.52, 0.12, 1.0];
/// Crossbars and diagonals: a shade darker than the chords so the lattice reads.
pub const ORANGE_DARK: Color = [0.70, 0.36, 0.07, 1.0];
pub const BASE: Color = [0.30, 0.33, 0.37, 1.0];
pub const GRAPHITE: Color = [0.17, 0.19, 0.22, 1.0];
pub const STEEL: Color = [0.72, 0.75, 0.80, 1.0];
pub const ACCENT: Color = [0.20, 0.85, 1.00, 1.0];

#[derive(Clone, Copy)]
pub enum Shape {
    Cube,
    Cylinder,
}

pub struct Part {
    pub shape: Shape,
    pub model: Mat4,
    pub color: Color,
}

fn mix(a: Color, b: Color, t: f32) -> Color {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

fn part(shape: Shape, frame: Mat4, offset: Vec3, rot: Quat, scale: Vec3, color: Color) -> Part {
    Part { shape, model: frame * Mat4::from_scale_rotation_translation(scale, rot, offset), color }
}

fn block(frame: Mat4, offset: Vec3, size: Vec3, color: Color) -> Part {
    part(Shape::Cube, frame, offset, Quat::IDENTITY, size, color)
}

/// Upright cylinder (axis along frame Y).
fn post(frame: Mat4, offset: Vec3, radius: f32, height: f32, color: Color) -> Part {
    let scale = Vec3::new(radius * 2.0, height, radius * 2.0);
    part(Shape::Cylinder, frame, offset, Quat::IDENTITY, scale, color)
}

/// Joint barrel (axis along frame Z, the pitch axis).
fn barrel(frame: Mat4, offset: Vec3, radius: f32, length: f32, color: Color) -> Part {
    let scale = Vec3::new(radius * 2.0, length, radius * 2.0);
    let rot = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    part(Shape::Cylinder, frame, offset, rot, scale, color)
}

/// Round bar from `a` to `b` (both in `frame` coordinates).
fn strut(frame: Mat4, a: Vec3, b: Vec3, radius: f32, color: Color) -> Part {
    let d = b - a;
    let len = d.length().max(1e-6);
    let rot = Quat::from_rotation_arc(Vec3::Y, d / len);
    part(Shape::Cylinder, frame, (a + b) * 0.5, rot, Vec3::new(radius * 2.0, len, radius * 2.0), color)
}

/// Tapering box truss along the frame's +Y from `y0` to `y1`. Four corner chords
/// shrink from cross-section `w0` to `w1` (x = in the arm's plane, z = across
/// it); crossbars ring each of the `bays + 1` stations, and diagonals zig-zag
/// along the two broad faces.
struct Truss {
    y0: f32,
    y1: f32,
    w0: (f32, f32),
    w1: (f32, f32),
    bays: usize,
}

const CHORD_R: f32 = 0.0065;
const BAR_R: f32 = 0.0042;

impl Truss {
    /// Corner `c` (0..4, around the section) at fraction `t` of the length.
    fn corner(&self, c: usize, t: f32) -> Vec3 {
        let wx = self.w0.0 + (self.w1.0 - self.w0.0) * t;
        let wz = self.w0.1 + (self.w1.1 - self.w0.1) * t;
        let (sx, sz) = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)][c];
        Vec3::new(sx * wx / 2.0, self.y0 + (self.y1 - self.y0) * t, sz * wz / 2.0)
    }

    fn parts(&self, frame: Mat4, out: &mut Vec<Part>) {
        let t = |k: usize| k as f32 / self.bays as f32;
        for c in 0..4 {
            out.push(strut(frame, self.corner(c, 0.0), self.corner(c, 1.0), CHORD_R, ORANGE));
        }
        for k in 0..=self.bays {
            for c in 0..4 {
                out.push(strut(frame, self.corner(c, t(k)), self.corner((c + 1) % 4, t(k)), BAR_R, ORANGE_DARK));
            }
        }
        // zig-zag on the z− face (corners 0–1) and the z+ face (corners 3–2)
        for k in 0..self.bays {
            let (a, b) = if k % 2 == 0 { (0, 1) } else { (1, 0) };
            out.push(strut(frame, self.corner(a, t(k)), self.corner(b, t(k + 1)), BAR_R, ORANGE_DARK));
            let (a, b) = if k % 2 == 0 { (3, 2) } else { (2, 3) };
            out.push(strut(frame, self.corner(a, t(k)), self.corner(b, t(k + 1)), BAR_R, ORANGE_DARK));
        }
        // end fittings where the chords meet the joints
        let (a, b) = (self.corner(2, 0.0), self.corner(2, 1.0));
        out.push(block(frame, Vec3::Y * self.y0, Vec3::new(a.x * 2.0 + 0.016, 0.02, a.z * 2.0 + 0.016), GRAPHITE));
        out.push(block(frame, Vec3::Y * self.y1, Vec3::new(b.x * 2.0 + 0.014, 0.018, b.z * 2.0 + 0.014), GRAPHITE));
    }
}

/// Finger segments: (length, radius at start, radius at end).
const FINGER: [(f32, f32, f32); 3] = [(0.036, 0.0065, 0.0055), (0.03, 0.0055, 0.0045), (0.024, 0.0045, 0.0015)];
/// Segment tilt away from the wrist axis (degrees), claw open vs. closed on a box.
const FINGER_OPEN: [f32; 3] = [44.0, 0.0, -35.0];
const FINGER_CLOSED: [f32; 3] = [30.0, -12.0, -50.0];

/// Three-fingered claw on the wrist frame; `grip` 0 = open, 1 = closed (the
/// fingertips then meet around the tool point).
fn claw(frame: Mat4, grip: f32, out: &mut Vec<Part>) {
    out.push(post(frame, Vec3::Y * 0.03, 0.026, 0.035, GRAPHITE)); // wrist housing
    out.push(post(frame, Vec3::Y * 0.052, 0.042, 0.012, STEEL)); // palm
    for i in 0..3 {
        // each finger lives in a vertical plane around the wrist axis and curls inward
        let f = frame * Mat4::from_rotation_y(i as f32 * TAU / 3.0 + 0.5);
        let p0 = Vec3::new(0.03, 0.056, 0.0);
        let mut p = p0;
        out.push(barrel(f, p0, 0.009, 0.016, GRAPHITE));
        for (n, &(len, r0, r1)) in FINGER.iter().enumerate() {
            let tilt = FINGER_OPEN[n] + (FINGER_CLOSED[n] - FINGER_OPEN[n]) * grip;
            let dir = Vec3::new(tilt.to_radians().sin(), tilt.to_radians().cos(), 0.0);
            let q = p + dir * len;
            // taper by drawing each segment as two halves of shrinking radius
            let mid = (p + q) * 0.5;
            out.push(strut(f, p, mid, r0, STEEL));
            out.push(strut(f, mid, q, (r0 + r1) * 0.5, STEEL));
            if n < FINGER.len() - 1 {
                out.push(barrel(f, q, r1 * 1.25, 0.012, GRAPHITE)); // knuckle
            }
            p = q;
        }
    }
}

pub const TABLE: Color = [0.36, 0.30, 0.25, 1.0];
pub const CRATE: Color = [0.95, 0.80, 0.25, 1.0];

/// Table and box.
pub fn world_parts(world: &World, tool: Mat4) -> Vec<Part> {
    let top_t = 0.03;
    let (hx, hz) = TABLE_HALF;
    let mut parts = vec![block(
        Mat4::IDENTITY,
        TABLE_CENTER + Vec3::Y * (TABLE_TOP - top_t / 2.0),
        Vec3::new(hx * 2.0, top_t, hz * 2.0),
        TABLE,
    )];
    for (sx, sz) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let leg = TABLE_CENTER + Vec3::new(sx * (hx - 0.04), (TABLE_TOP - top_t) / 2.0, sz * (hz - 0.04));
        parts.push(block(Mat4::IDENTITY, leg, Vec3::new(0.035, TABLE_TOP - top_t, 0.035), GRAPHITE));
    }
    let crate_ = world.box_pose(tool);
    parts.push(block(crate_, Vec3::ZERO, Vec3::splat(BOX_SIZE), CRATE));
    // darker bands so the box's rotation reads
    parts.push(block(crate_, Vec3::ZERO, Vec3::new(BOX_SIZE * 1.02, BOX_SIZE * 0.25, BOX_SIZE * 1.02), ORANGE_DARK));
    parts
}

pub fn arm_parts(arm: &Arm, grip: f32) -> Vec<Part> {
    let f = arm.frames();
    // Joints glow with the accent colour in proportion to how fast they move.
    let joint_color = |j: usize| {
        let t = (arm.velocity(j).abs() / JOINTS[j].max_speed).min(1.0);
        mix(GRAPHITE, ACCENT, t)
    };

    // Fixed wall mount: a plate on the wall and two flanges holding the yaw pin.
    let mount = Mat4::from_translation(Vec3::Y * MOUNT_H);
    let flange_len = YAW_X + 0.03;
    let flange = |y: f32| block(mount, Vec3::new(0.03 + flange_len / 2.0, y, 0.0), Vec3::new(flange_len, 0.025, 0.12), BASE);

    let mut parts = vec![
        block(mount, Vec3::X * 0.015, Vec3::new(0.03, 0.30, 0.24), BASE),
        flange(0.0775),
        flange(-0.0775),
        // yaw hub (vertical axis, left/right)
        post(f.yaw, Vec3::ZERO, 0.05, 0.13, joint_color(0)),
        // clevis carrying the pitch axis
        block(f.yaw, Vec3::X * (PITCH_OFF - 0.06), Vec3::new(0.03, 0.10, 0.17), ORANGE),
        block(f.yaw, Vec3::new(PITCH_OFF, 0.0, 0.075), Vec3::new(0.11, 0.11, 0.02), ORANGE),
        block(f.yaw, Vec3::new(PITCH_OFF, 0.0, -0.075), Vec3::new(0.11, 0.11, 0.02), ORANGE),
        // joint barrels
        barrel(f.pitch, Vec3::ZERO, 0.05, 0.13, joint_color(1)),
        barrel(f.elbow, Vec3::ZERO, 0.042, 0.11, joint_color(2)),
        barrel(f.wrist, Vec3::ZERO, 0.032, 0.085, joint_color(3)),
    ];
    // upper arm and forearm: trusses tapering away from their driving joint
    Truss { y0: 0.05, y1: UPPER_L - 0.045, w0: (0.085, 0.075), w1: (0.05, 0.05), bays: 4 }.parts(f.pitch, &mut parts);
    Truss { y0: 0.045, y1: FORE_L - 0.035, w0: (0.06, 0.055), w1: (0.036, 0.036), bays: 3 }.parts(f.elbow, &mut parts);
    claw(f.wrist, grip, &mut parts);
    parts
}
