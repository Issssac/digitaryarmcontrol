use glam::{Mat4, Vec3};

pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        // In front of the arm and to its right, looking slightly down.
        Self { yaw: 1.05, pitch: 0.38, distance: 1.05, target: Vec3::new(0.12, 0.18, 0.0) }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + Vec3::new(cp * sy, sp, cp * cy) * self.distance
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    pub fn proj(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh_gl(45f32.to_radians(), aspect.max(0.01), 0.01, 20.0)
    }

    /// Mouse drag in CSS pixels.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw = (self.yaw - dx * 0.008).rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch + dy * 0.008).clamp(0.02, 1.5);
    }

    /// Wheel delta (positive = zoom out).
    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance * (delta * 0.001).exp()).clamp(0.25, 3.0);
    }
}
