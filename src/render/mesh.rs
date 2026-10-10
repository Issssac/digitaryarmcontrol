//! Unit mesh generators (interleaved position + normal, 6 floats per vertex)
//! and their GPU counterparts.

use glam::Vec3;
use wasm_bindgen::JsValue;
use web_sys::{WebGl2RenderingContext as GL, WebGlBuffer, WebGlVertexArrayObject};

const FLOATS_PER_VERTEX: i32 = 6;

fn push(v: &mut Vec<f32>, p: Vec3, n: Vec3) {
    v.extend_from_slice(&[p.x, p.y, p.z, n.x, n.y, n.z]);
}

/// Unit cube centred on the origin (-0.5..0.5).
pub fn cube() -> Vec<f32> {
    let mut v = Vec::with_capacity(36 * 6);
    let faces = [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (Vec3::NEG_X, Vec3::Y, Vec3::Z),
        (Vec3::Y, Vec3::X, Vec3::Z),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (Vec3::NEG_Z, Vec3::X, Vec3::Y),
    ];
    for (n, u, w) in faces {
        let p = |a: f32, b: f32| (n + u * a + w * b) * 0.5;
        let (p0, p1, p2, p3) = (p(-1.0, -1.0), p(1.0, -1.0), p(1.0, 1.0), p(-1.0, 1.0));
        for q in [p0, p1, p2, p0, p2, p3] {
            push(&mut v, q, n);
        }
    }
    v
}

/// Unit cylinder: radius 0.5, height 1 along Y, centred on the origin.
pub fn cylinder(segments: u32) -> Vec<f32> {
    let mut v = Vec::new();
    let top = Vec3::Y * 0.5;
    let bottom = Vec3::NEG_Y * 0.5;
    for i in 0..segments {
        let a0 = i as f32 / segments as f32 * std::f32::consts::TAU;
        let a1 = (i + 1) as f32 / segments as f32 * std::f32::consts::TAU;
        let n0 = Vec3::new(a0.cos(), 0.0, a0.sin());
        let n1 = Vec3::new(a1.cos(), 0.0, a1.sin());
        let (r0, r1) = (n0 * 0.5, n1 * 0.5);
        // side
        for (p, n) in [
            (r0 + bottom, n0),
            (r1 + bottom, n1),
            (r1 + top, n1),
            (r0 + bottom, n0),
            (r1 + top, n1),
            (r0 + top, n0),
        ] {
            push(&mut v, p, n);
        }
        // caps
        for p in [top, r0 + top, r1 + top] {
            push(&mut v, p, Vec3::Y);
        }
        for p in [bottom, r1 + bottom, r0 + bottom] {
            push(&mut v, p, Vec3::NEG_Y);
        }
    }
    v
}

/// Ground grid in the XZ plane as GL_LINES.
pub fn grid(half: f32, step: f32) -> Vec<f32> {
    let mut v = Vec::new();
    let n = (half / step).round() as i32;
    for i in -n..=n {
        let c = i as f32 * step;
        push(&mut v, Vec3::new(c, 0.0, -half), Vec3::Y);
        push(&mut v, Vec3::new(c, 0.0, half), Vec3::Y);
        push(&mut v, Vec3::new(-half, 0.0, c), Vec3::Y);
        push(&mut v, Vec3::new(half, 0.0, c), Vec3::Y);
    }
    v
}

pub struct GpuMesh {
    pub vao: WebGlVertexArrayObject,
    vbo: WebGlBuffer,
    pub count: i32,
    pub mode: u32,
}

impl GpuMesh {
    pub fn new(gl: &GL, data: &[f32], mode: u32, dynamic: bool) -> Result<Self, JsValue> {
        let vao = gl.create_vertex_array().ok_or("create_vertex_array failed")?;
        let vbo = gl.create_buffer().ok_or("create_buffer failed")?;
        gl.bind_vertex_array(Some(&vao));
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&vbo));
        let stride = FLOATS_PER_VERTEX * 4;
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_with_i32(0, 3, GL::FLOAT, false, stride, 0);
        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_with_i32(1, 3, GL::FLOAT, false, stride, 12);
        gl.bind_vertex_array(None);
        let mut mesh = Self { vao, vbo, count: 0, mode };
        mesh.upload(gl, data, if dynamic { GL::DYNAMIC_DRAW } else { GL::STATIC_DRAW });
        Ok(mesh)
    }

    pub fn update(&mut self, gl: &GL, data: &[f32]) {
        self.upload(gl, data, GL::DYNAMIC_DRAW);
    }

    fn upload(&mut self, gl: &GL, data: &[f32], usage: u32) {
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.vbo));
        gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &js_sys::Float32Array::from(data), usage);
        self.count = data.len() as i32 / FLOATS_PER_VERTEX;
    }
}
