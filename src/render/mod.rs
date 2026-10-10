//! WebGL2 renderer: one Blinn-Phong program, unit meshes scaled per part,
//! planar projected shadows, a ground grid and an end-effector trail.

pub mod camera;
pub mod mesh;
pub mod scene;

use std::f32::consts::FRAC_PI_2;

use glam::{Mat3, Mat4, Quat, Vec3};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext as GL, WebGlProgram, WebGlShader, WebGlUniformLocation};

use crate::arm::Arm;
use crate::world::World;
use camera::OrbitCamera;
use mesh::GpuMesh;
use scene::{Color, Shape};

const BACKGROUND: Color = [0.10, 0.11, 0.13, 1.0];
const GROUND: Color = [0.16, 0.17, 0.19, 1.0];
const WALL: Color = [0.19, 0.20, 0.23, 1.0];
const GRID: Color = [0.24, 0.26, 0.30, 1.0];
const SHADOW: Color = [0.0, 0.0, 0.0, 0.35];
const TRAIL: Color = [0.20, 0.85, 1.00, 1.0];
/// Direction *towards* the light: above and in front of the wall, so shadows
/// land on both the floor and the wall.
const LIGHT_DIR: Vec3 = Vec3::new(0.6, 1.0, 0.35);

const VERTEX_SHADER: &str = r#"#version 300 es
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
uniform mat4 u_viewproj;
uniform mat4 u_model;
uniform mat3 u_normal_mat;
out vec3 v_normal;
out vec3 v_world;
void main() {
    vec4 w = u_model * vec4(a_pos, 1.0);
    v_world = w.xyz;
    v_normal = u_normal_mat * a_normal;
    gl_Position = u_viewproj * w;
}
"#;

const FRAGMENT_SHADER: &str = r#"#version 300 es
precision highp float;
in vec3 v_normal;
in vec3 v_world;
uniform vec4 u_color;
uniform vec3 u_light_dir;
uniform vec3 u_eye;
uniform vec3 u_fog;
uniform float u_lit;
out vec4 o_color;
void main() {
    vec3 col = u_color.rgb;
    if (u_lit > 0.5) {
        vec3 albedo = pow(col, vec3(2.2));
        vec3 n = normalize(v_normal);
        vec3 l = normalize(u_light_dir);
        vec3 v = normalize(u_eye - v_world);
        vec3 h = normalize(l + v);
        float diff = max(dot(n, l), 0.0);
        float spec = pow(max(dot(n, h), 0.0), 48.0) * 0.25;
        float hemi = 0.5 + 0.5 * n.y;
        vec3 ambient = albedo * mix(vec3(0.05, 0.045, 0.04), vec3(0.16, 0.19, 0.24), hemi);
        col = pow(ambient + albedo * diff * 0.9 + vec3(spec), vec3(1.0 / 2.2));
    }
    float d = length(u_eye - v_world);
    col = mix(col, u_fog, clamp((d - 2.5) / 6.0, 0.0, 1.0));
    o_color = vec4(col, u_color.a);
}
"#;

struct Uniforms {
    viewproj: WebGlUniformLocation,
    model: WebGlUniformLocation,
    normal_mat: WebGlUniformLocation,
    color: WebGlUniformLocation,
    light_dir: WebGlUniformLocation,
    eye: WebGlUniformLocation,
    fog: WebGlUniformLocation,
    lit: WebGlUniformLocation,
}

pub struct Renderer {
    gl: GL,
    canvas: HtmlCanvasElement,
    program: WebGlProgram,
    u: Uniforms,
    cube: GpuMesh,
    cylinder: GpuMesh,
    grid: GpuMesh,
    trail: GpuMesh,
    trail_data: Vec<f32>,
}

impl Renderer {
    pub fn new(canvas: HtmlCanvasElement) -> Result<Self, JsValue> {
        let opts = js_sys::Object::new();
        js_sys::Reflect::set(&opts, &"antialias".into(), &true.into())?;
        let gl: GL = canvas
            .get_context_with_context_options("webgl2", &opts)?
            .ok_or("WebGL2 is not supported by this browser")?
            .dyn_into()?;

        let program = link_program(&gl, VERTEX_SHADER, FRAGMENT_SHADER)?;
        let loc = |name: &str| {
            gl.get_uniform_location(&program, name)
                .ok_or_else(|| JsValue::from_str(&format!("missing uniform {name}")))
        };
        let u = Uniforms {
            viewproj: loc("u_viewproj")?,
            model: loc("u_model")?,
            normal_mat: loc("u_normal_mat")?,
            color: loc("u_color")?,
            light_dir: loc("u_light_dir")?,
            eye: loc("u_eye")?,
            fog: loc("u_fog")?,
            lit: loc("u_lit")?,
        };

        let cube = GpuMesh::new(&gl, &mesh::cube(), GL::TRIANGLES, false)?;
        let cylinder = GpuMesh::new(&gl, &mesh::cylinder(32), GL::TRIANGLES, false)?;
        let grid = GpuMesh::new(&gl, &mesh::grid(2.0, 0.1), GL::LINES, false)?;
        let trail = GpuMesh::new(&gl, &[], GL::LINE_STRIP, true)?;

        Ok(Self { gl, canvas, program, u, cube, cylinder, grid, trail, trail_data: Vec::new() })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.canvas.set_width(width.max(1));
        self.canvas.set_height(height.max(1));
    }

    pub fn render(&mut self, cam: &OrbitCamera, arm: &Arm, world: &World, trail: &[Vec3]) {
        self.trail_data.clear();
        for p in trail {
            self.trail_data.extend_from_slice(&[p.x, p.y, p.z, 0.0, 1.0, 0.0]);
        }
        self.trail.update(&self.gl, &self.trail_data);

        let gl = &self.gl;
        let (w, h) = (self.canvas.width() as i32, self.canvas.height() as i32);
        gl.viewport(0, 0, w, h);
        let [r, g, b, a] = BACKGROUND;
        gl.clear_color(r, g, b, a);
        gl.clear(GL::COLOR_BUFFER_BIT | GL::DEPTH_BUFFER_BIT);
        gl.enable(GL::DEPTH_TEST);
        gl.depth_func(GL::LESS);
        gl.use_program(Some(&self.program));

        let viewproj = cam.proj(w as f32 / h.max(1) as f32) * cam.view();
        let eye = cam.eye();
        gl.uniform_matrix4fv_with_f32_array(Some(&self.u.viewproj), false, &viewproj.to_cols_array());
        gl.uniform3fv_with_f32_array(Some(&self.u.light_dir), &LIGHT_DIR.normalize().to_array());
        gl.uniform3fv_with_f32_array(Some(&self.u.eye), &eye.to_array());
        gl.uniform3f(Some(&self.u.fog), r, g, b);

        // ground slab (top face at y = 0), wall slab (front face at x = 0), grids
        let ground = Mat4::from_scale_rotation_translation(Vec3::new(12.0, 0.02, 12.0), Quat::IDENTITY, Vec3::Y * -0.01);
        self.draw(&self.cube, ground, GROUND, true);
        let wall = Mat4::from_scale_rotation_translation(Vec3::new(0.04, 4.0, 12.0), Quat::IDENTITY, Vec3::new(-0.02, 2.0, 0.0));
        self.draw(&self.cube, wall, WALL, true);
        self.draw(&self.grid, Mat4::from_translation(Vec3::Y * 0.0015), GRID, false);
        let wall_grid = Mat4::from_translation(Vec3::new(0.0015, 2.0, 0.0)) * Mat4::from_rotation_z(FRAC_PI_2);
        self.draw(&self.grid, wall_grid, GRID, false);

        // Planar shadows: every part projected onto the floor and onto the wall.
        // Each point's shadow lands on whichever plane the light ray hits first;
        // the other projection falls behind the wall or under the floor and is
        // hidden by the depth test. Within one plane all shadow fragments share
        // the same depth, so overlaps fail the LESS test and don't double-darken.
        let mut parts = scene::arm_parts(arm, world.grip());
        parts.extend(scene::world_parts(world, arm.frames().tool));
        let onto_floor = shadow_matrix(LIGHT_DIR, Vec3::Y, 0.003);
        let onto_wall = shadow_matrix(LIGHT_DIR, Vec3::X, 0.003);
        gl.enable(GL::BLEND);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE_MINUS_SRC_ALPHA);
        for flatten in [onto_floor, onto_wall] {
            for p in &parts {
                self.draw(self.mesh(p.shape), flatten * p.model, SHADOW, false);
            }
        }
        gl.disable(GL::BLEND);

        for p in &parts {
            self.draw(self.mesh(p.shape), p.model, p.color, true);
        }

        if self.trail.count >= 2 {
            self.draw(&self.trail, Mat4::IDENTITY, TRAIL, false);
        }
    }

    fn mesh(&self, shape: Shape) -> &GpuMesh {
        match shape {
            Shape::Cube => &self.cube,
            Shape::Cylinder => &self.cylinder,
        }
    }

    fn draw(&self, mesh: &GpuMesh, model: Mat4, color: Color, lit: bool) {
        let gl = &self.gl;
        let normal_mat = if lit { Mat3::from_mat4(model).inverse().transpose() } else { Mat3::IDENTITY };
        gl.uniform_matrix4fv_with_f32_array(Some(&self.u.model), false, &model.to_cols_array());
        gl.uniform_matrix3fv_with_f32_array(Some(&self.u.normal_mat), false, &normal_mat.to_cols_array());
        gl.uniform4fv_with_f32_array(Some(&self.u.color), &color);
        gl.uniform1f(Some(&self.u.lit), if lit { 1.0 } else { 0.0 });
        gl.bind_vertex_array(Some(&mesh.vao));
        gl.draw_arrays(mesh.mode, 0, mesh.count);
    }
}

/// Projects points along `light` (direction towards the light) onto the plane
/// `normal · p = d`: p' = p − light · (normal · p − d) / (normal · light).
fn shadow_matrix(light: Vec3, normal: Vec3, d: f32) -> Mat4 {
    let k = normal.dot(light);
    Mat4::from_cols(
        (Vec3::X - light * (normal.x / k)).extend(0.0),
        (Vec3::Y - light * (normal.y / k)).extend(0.0),
        (Vec3::Z - light * (normal.z / k)).extend(0.0),
        (light * (d / k)).extend(1.0),
    )
}

fn compile(gl: &GL, kind: u32, src: &str) -> Result<WebGlShader, JsValue> {
    let shader = gl.create_shader(kind).ok_or("create_shader failed")?;
    gl.shader_source(&shader, src);
    gl.compile_shader(&shader);
    if gl.get_shader_parameter(&shader, GL::COMPILE_STATUS).as_bool().unwrap_or(false) {
        Ok(shader)
    } else {
        Err(gl.get_shader_info_log(&shader).unwrap_or_default().into())
    }
}

fn link_program(gl: &GL, vs: &str, fs: &str) -> Result<WebGlProgram, JsValue> {
    let program = gl.create_program().ok_or("create_program failed")?;
    gl.attach_shader(&program, &compile(gl, GL::VERTEX_SHADER, vs)?);
    gl.attach_shader(&program, &compile(gl, GL::FRAGMENT_SHADER, fs)?);
    gl.link_program(&program);
    if gl.get_program_parameter(&program, GL::LINK_STATUS).as_bool().unwrap_or(false) {
        Ok(program)
    } else {
        Err(gl.get_program_info_log(&program).unwrap_or_default().into())
    }
}
