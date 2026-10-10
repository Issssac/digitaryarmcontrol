//! RBJ-cookbook biquads (transposed direct form II, f64 state).

use std::f64::consts::PI;

/// Q values for a 4th-order Butterworth built from two biquads.
pub const BUTTER4_Q: [f64; 2] = [0.541_196_1, 1.306_563];
pub const BUTTER2_Q: f64 = std::f64::consts::FRAC_1_SQRT_2;

#[derive(Clone, Copy, Debug)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    fn from_raw(b: [f64; 3], a: [f64; 3]) -> Self {
        Self { b0: b[0] / a[0], b1: b[1] / a[0], b2: b[2] / a[0], a1: a[1] / a[0], a2: a[2] / a[0], z1: 0.0, z2: 0.0 }
    }

    fn params(fs: f32, f0: f32, q: f64) -> (f64, f64) {
        let w0 = 2.0 * PI * f0 as f64 / fs as f64;
        (w0.cos(), w0.sin() / (2.0 * q))
    }

    pub fn lowpass(fs: f32, f0: f32, q: f64) -> Self {
        let (c, alpha) = Self::params(fs, f0, q);
        Self::from_raw([(1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0], [1.0 + alpha, -2.0 * c, 1.0 - alpha])
    }

    pub fn highpass(fs: f32, f0: f32, q: f64) -> Self {
        let (c, alpha) = Self::params(fs, f0, q);
        Self::from_raw([(1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0], [1.0 + alpha, -2.0 * c, 1.0 - alpha])
    }

    pub fn notch(fs: f32, f0: f32, q: f64) -> Self {
        let (c, alpha) = Self::params(fs, f0, q);
        Self::from_raw([1.0, -2.0 * c, 1.0], [1.0 + alpha, -2.0 * c, 1.0 - alpha])
    }

    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Set the state as if `x` had been the input forever (avoids a startup
    /// transient when the signal has a large DC offset).
    pub fn prime(&mut self, x: f64) {
        let dc_gain = (self.b0 + self.b1 + self.b2) / (1.0 + self.a1 + self.a2);
        let y = dc_gain * x;
        self.z2 = self.b2 * x - self.a2 * y;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms_after(mut f: impl FnMut(f64) -> f64, freq: f64, fs: f64) -> f64 {
        let n = (fs * 4.0) as usize;
        let mut acc = 0.0;
        let mut cnt = 0;
        for i in 0..n {
            let y = f((2.0 * PI * freq * i as f64 / fs).sin());
            if i > n / 2 {
                acc += y * y;
                cnt += 1;
            }
        }
        (acc / cnt as f64).sqrt()
    }

    #[test]
    fn notch_kills_mains_passes_emg_band() {
        let mut n = Biquad::notch(250.0, 60.0, 30.0);
        assert!(rms_after(|x| n.process(x), 60.0, 250.0) < 0.02);
        let mut n = Biquad::notch(250.0, 60.0, 30.0);
        assert!(rms_after(|x| n.process(x), 90.0, 250.0) > 0.65);
    }

    #[test]
    fn highpass_blocks_dc_and_prime_has_no_transient() {
        let mut h = Biquad::highpass(250.0, 20.0, BUTTER2_Q);
        h.prime(5000.0);
        for _ in 0..10 {
            assert!(h.process(5000.0).abs() < 1e-6);
        }
    }

    #[test]
    fn lowpass_unity_dc() {
        let mut l = Biquad::lowpass(250.0, 4.0, BUTTER2_Q);
        let mut y = 0.0;
        for _ in 0..2000 {
            y = l.process(1.0);
        }
        assert!((y - 1.0).abs() < 1e-6);
    }
}
