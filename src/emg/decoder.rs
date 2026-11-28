//! Covariance-correlation feature extractor and regularized Shrinkage LDA
//! decoder shared by both control schemes (muscle pairs and digit array).
//!
//! Surface EMG from neighbouring muscles often shares similar channel-amplitude
//! ratios, yet distinct motor-unit pools produce reproducible instantaneous
//! cross-channel correlations across the electrode array — and correlation is
//! invariant to contraction effort. Capturing the 200 ms sample covariance
//! matrix gives 8 relative log-RMS features (effort-invariant amplitude ratios),
//! 28 Fisher-z inter-channel correlations, and 1 saturated overall-effort
//! feature (37 features total), classified by closed-form Shrinkage LDA in
//! microseconds with zero hot-path heap allocations.

use super::cyton::NUM_CHANNELS;

/// Per-channel feature vector (envelope, log-RMS, or amplitude profile).
pub type Features = [f64; NUM_CHANNELS];

/// Most patterns one fit can take (single movements, compounds and postures together).
pub const NNLS_MAX: usize = 32;

/// Small dense linear solve of the leading n×n block, in place (Gaussian
/// elimination with partial pivoting): the solution is left in `b`. False if singular.
fn solve(a: &mut [[f64; NNLS_MAX]; NNLS_MAX], b: &mut [f64; NNLS_MAX], n: usize) -> bool {
    for col in 0..n {
        let Some(piv) = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())) else { return false };
        if a[piv][col].abs() < 1e-12 {
            return false;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for i in col + 1..n {
            let f = a[i][col] / a[col][col];
            for j in col..n {
                a[i][j] -= f * a[col][j];
            }
            b[i] -= f * b[col];
        }
    }
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|j| a[i][j] * b[j]).sum();
        b[i] = (b[i] - s) / a[i][i];
    }
    true
}

/// Non-negative least squares in normal-equation form: minimise ‖Aw − y‖ with
/// w ≥ 0, given G = AᵀA and h = Aᵀy (Lawson–Hanson active set), for up to
/// NNLS_MAX patterns, without allocating. Weights past h.len() are 0.
pub fn nnls(g: &[[f64; NNLS_MAX]], h: &[f64]) -> [f64; NNLS_MAX] {
    let n = h.len().min(NNLS_MAX).min(g.len());
    let mut w = [0.0; NNLS_MAX];
    let mut passive = [false; NNLS_MAX];
    let mut grad = [0.0; NNLS_MAX];
    let mut idx = [0usize; NNLS_MAX];
    let mut sub = [[0.0; NNLS_MAX]; NNLS_MAX];
    let mut sp = [0.0; NNLS_MAX];
    for _ in 0..3 * n + 3 {
        for i in 0..n {
            grad[i] = h[i] - (0..n).map(|j| g[i][j] * w[j]).sum::<f64>();
        }
        let Some(t) = (0..n).filter(|&i| !passive[i]).max_by(|&i, &j| grad[i].total_cmp(&grad[j])) else { break };
        if grad[t] <= 1e-10 {
            break;
        }
        passive[t] = true;
        loop {
            let mut m = 0;
            for i in (0..n).filter(|&i| passive[i]) {
                idx[m] = i;
                m += 1;
            }
            for a in 0..m {
                for b in 0..m {
                    sub[a][b] = g[idx[a]][idx[b]];
                }
                sp[a] = h[idx[a]];
            }
            if !solve(&mut sub, &mut sp, m) {
                return w;
            }
            if sp[..m].iter().all(|&v| v > 0.0) {
                for k in 0..m {
                    w[idx[k]] = sp[k];
                }
                break;
            }
            // step towards the unconstrained solution until a weight hits zero
            let alpha = (0..m)
                .filter(|&k| sp[k] <= 0.0)
                .map(|k| w[idx[k]] / (w[idx[k]] - sp[k]))
                .fold(1.0, f64::min);
            for k in 0..m {
                let i = idx[k];
                w[i] += alpha * (sp[k] - w[i]);
                if w[i] <= 1e-12 {
                    w[i] = 0.0;
                    passive[i] = false;
                }
            }
        }
    }
    w
}

/// Upper-triangle entries of an 8×8 symmetric covariance matrix: 8 + 28 = 36.
pub const NUM_COV: usize = NUM_CHANNELS * (NUM_CHANNELS + 1) / 2;
/// Off-diagonal pairs (i < j) in 8 channels.
pub const NUM_CORR: usize = NUM_CHANNELS * (NUM_CHANNELS - 1) / 2;
/// Full LDA feature dimension: 8 rel log-RMS + 28 Fisher-z correlations + 1 sat effort.
pub const NUM_FEATS: usize = NUM_CHANNELS + NUM_CORR + 1;
/// Rolling covariance window: 50 samples = 200 ms at 250 Hz.
pub const COV_WIN: usize = 50;
/// Diagonal floor (µV²) so quiet or silent channels stay well-conditioned.
pub const COV_EPS: f64 = 0.1;
/// Scale (nepers) for the saturated overall-effort feature: separates rest from
/// contraction while keeping medium and hard holds close together.
const EFFORT_SCALE: f64 = 1.5;
/// Recompute the running sum from the ring buffer every N samples to prevent
/// floating-point cancellation drift over multi-hour sessions.
const REFRESH_EVERY: u32 = 500;
/// Maximum classes supported without allocation in `LdaModel::scores`.
pub const MAX_LDA_CLASSES: usize = 16;

/// Upper-triangle entries of an 8×8 sample covariance matrix (µV²):
/// index `0..8` are the diagonal variances `C[i][i]`, and `8..36` are the
/// off-diagonal covariances `C[i][j]` for `0 <= i < j < 8` in row-major order.
pub type CovWindow = [f64; NUM_COV];

/// Index into `CovWindow` for off-diagonal `(i, j)` with `i < j`.
#[inline]
const fn off_index(i: usize, j: usize) -> usize {
    // 8 diagonal entries first, then row-major upper triangle
    NUM_CHANNELS + i * (2 * NUM_CHANNELS - i - 1) / 2 + (j - i - 1)
}

/// Rolling 200 ms sample covariance tracker over filtered (pre-rectified) EMG.
#[derive(Clone)]
pub struct CovTracker {
    ring: [[f64; NUM_CHANNELS]; COV_WIN],
    pos: usize,
    filled: usize,
    sum: CovWindow,
    ticks: u32,
}

impl Default for CovTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CovTracker {
    pub fn new() -> Self {
        Self {
            ring: [[0.0; NUM_CHANNELS]; COV_WIN],
            pos: 0,
            filled: 0,
            sum: [0.0; NUM_COV],
            ticks: 0,
        }
    }

    pub fn reset(&mut self) {
        self.ring = [[0.0; NUM_CHANNELS]; COV_WIN];
        self.pos = 0;
        self.filled = 0;
        self.sum = [0.0; NUM_COV];
        self.ticks = 0;
    }

    /// Push one filtered EMG sample (µV) and update the rolling covariance in O(1).
    pub fn push(&mut self, sample: &[f32; NUM_CHANNELS]) {
        let cur: [f64; NUM_CHANNELS] = std::array::from_fn(|c| sample[c] as f64);
        if self.filled == COV_WIN {
            let old = self.ring[self.pos];
            Self::add_outer(&mut self.sum, &old, -1.0);
        } else {
            self.filled += 1;
        }
        self.ring[self.pos] = cur;
        Self::add_outer(&mut self.sum, &cur, 1.0);
        self.pos = (self.pos + 1) % COV_WIN;
        self.ticks = self.ticks.wrapping_add(1);
        if self.ticks % REFRESH_EVERY == 0 {
            self.recompute();
        }
    }

    pub fn filled(&self) -> usize {
        self.filled
    }

    /// Current sample covariance matrix (`sum / filled`) in µV².
    pub fn window(&self) -> CovWindow {
        let inv = 1.0 / self.filled.max(1) as f64;
        std::array::from_fn(|k| self.sum[k] * inv)
    }

    #[inline]
    fn add_outer(sum: &mut CovWindow, x: &[f64; NUM_CHANNELS], sign: f64) {
        for i in 0..NUM_CHANNELS {
            sum[i] += sign * x[i] * x[i];
        }
        let mut idx = NUM_CHANNELS;
        for i in 0..NUM_CHANNELS {
            let xi = sign * x[i];
            for j in (i + 1)..NUM_CHANNELS {
                sum[idx] += xi * x[j];
                idx += 1;
            }
        }
    }

    fn recompute(&mut self) {
        self.sum = [0.0; NUM_COV];
        for k in 0..self.filled {
            let x = self.ring[k];
            Self::add_outer(&mut self.sum, &x, 1.0);
        }
    }
}

/// Per-channel log-RMS (`ln(sqrt(C[c][c] + COV_EPS))`) from a covariance window.
pub fn cov_log_rms(cov: &CovWindow) -> Features {
    std::array::from_fn(|c| (cov[c].max(0.0) + COV_EPS).sqrt().ln())
}

/// Extracts the 37D LDA feature vector and per-channel log-RMS from a covariance window:
/// - `0..8`: relative log-RMS (`lr[c] - mean_usable(lr)`), effort-invariant spatial ratio
/// - `8..36`: Fisher-z inter-channel correlations (`atanh(clamp(r_ij, -0.98, 0.98))`),
///   effort-invariant volume-conduction signature
/// - `36`: saturated overall effort (`tanh((mean_lr - rest0_mlr) / 1.5)`), separates
///   rest from movement without pulling hard holds away from medium holds.
///
/// Unusable channels (disconnected or flagged bad) are zeroed out cleanly.
pub fn extract_features(
    cov: &CovWindow,
    usable: &[bool; NUM_CHANNELS],
    rest0_mlr: f64,
) -> ([f64; NUM_FEATS], Features) {
    let mut d = [0.0; NUM_CHANNELS];
    let mut lr = [0.0; NUM_CHANNELS];
    let mut lr_sum = 0.0;
    let mut n_usable = 0usize;
    for c in 0..NUM_CHANNELS {
        d[c] = (cov[c].max(0.0) + COV_EPS).sqrt();
        lr[c] = d[c].ln();
        if usable[c] {
            lr_sum += lr[c];
            n_usable += 1;
        }
    }
    let mlr = if n_usable > 0 { lr_sum / n_usable as f64 } else { 0.0 };

    let mut feat = [0.0; NUM_FEATS];
    for c in 0..NUM_CHANNELS {
        feat[c] = if usable[c] { lr[c] - mlr } else { 0.0 };
    }
    let mut idx = NUM_CHANNELS;
    for i in 0..NUM_CHANNELS {
        for j in (i + 1)..NUM_CHANNELS {
            if usable[i] && usable[j] {
                let r = (cov[idx] / (d[i] * d[j])).clamp(-0.98, 0.98);
                feat[idx] = r.atanh();
            }
            idx += 1;
        }
    }
    feat[NUM_FEATS - 1] = ((mlr - rest0_mlr) / EFFORT_SCALE).tanh();
    (feat, lr)
}

/// The feature vector with the raw mean log-RMS over usable channels in the
/// effort slot (see `with_effort`), so it can be stored before the limp rest's
/// level is known.
pub fn raw_features(cov: &CovWindow, usable: &[bool; NUM_CHANNELS]) -> [f64; NUM_FEATS] {
    let (mut f, lr) = extract_features(cov, usable, 0.0);
    let n = usable.iter().filter(|&&u| u).count();
    f[NUM_FEATS - 1] = if n > 0 { (0..NUM_CHANNELS).filter(|&c| usable[c]).map(|c| lr[c]).sum::<f64>() / n as f64 } else { 0.0 };
    f
}

/// `raw_features` with its effort slot saturated against the limp rest's mean log-RMS.
pub fn with_effort(mut f: [f64; NUM_FEATS], rest_mlr: f64) -> [f64; NUM_FEATS] {
    f[NUM_FEATS - 1] = ((f[NUM_FEATS - 1] - rest_mlr) / EFFORT_SCALE).tanh();
    f
}

/// Regularized Shrinkage Linear Discriminant Analysis over `NUM_FEATS` dimensions.
#[derive(Clone, Debug)]
pub struct LdaModel {
    mean: [f64; NUM_FEATS],
    sd: [f64; NUM_FEATS],
    active: [bool; NUM_FEATS],
    classes: Vec<usize>,
    weights: Vec<[f64; NUM_FEATS]>,
    biases: Vec<f64>,
}

impl LdaModel {
    /// Fits a shrinkage LDA classifier with equal-weight pooled within-class
    /// covariance (so longer holds do not dominate the noise estimate).
    /// Returns `None` if fewer than 2 classes are present.
    pub fn fit(samples: &[(usize, [f64; NUM_FEATS])], shrink: f64) -> Option<Self> {
        if samples.len() < 2 {
            return None;
        }
        let mut classes: Vec<usize> = samples.iter().map(|&(c, _)| c).collect();
        classes.sort_unstable();
        classes.dedup();
        if classes.len() < 2 || classes.len() > MAX_LDA_CLASSES {
            return None;
        }

        let n = samples.len() as f64;
        let mut mean = [0.0; NUM_FEATS];
        for (_, x) in samples {
            for d in 0..NUM_FEATS {
                mean[d] += x[d];
            }
        }
        for d in 0..NUM_FEATS {
            mean[d] /= n;
        }

        let mut var = [0.0; NUM_FEATS];
        for (_, x) in samples {
            for d in 0..NUM_FEATS {
                let diff = x[d] - mean[d];
                var[d] += diff * diff;
            }
        }
        let mut sd = [1.0; NUM_FEATS];
        let mut active = [false; NUM_FEATS];
        let mut active_count = 0usize;
        for d in 0..NUM_FEATS {
            let v = var[d] / n;
            if v > 1e-12 {
                sd[d] = v.sqrt().max(1e-3);
                active[d] = true;
                active_count += 1;
            }
        }
        if active_count == 0 {
            return None;
        }

        let k = classes.len();
        let mut class_means = vec![[0.0; NUM_FEATS]; k];
        let mut class_counts = vec![0usize; k];
        for (c, x) in samples {
            let ci = classes.binary_search(c).ok()?;
            class_counts[ci] += 1;
            for d in 0..NUM_FEATS {
                if active[d] {
                    class_means[ci][d] += (x[d] - mean[d]) / sd[d];
                }
            }
        }
        for ci in 0..k {
            let inv = 1.0 / class_counts[ci].max(1) as f64;
            for d in 0..NUM_FEATS {
                class_means[ci][d] *= inv;
            }
        }

        // Equal-weight pooled within-class covariance Sw (NUM_FEATS × NUM_FEATS)
        let mut sw = [0.0; NUM_FEATS * NUM_FEATS];
        for (c, x) in samples {
            let ci = classes.binary_search(c).ok()?;
            let w_sample = 1.0 / (k as f64 * class_counts[ci].max(1) as f64);
            let mut diff = [0.0; NUM_FEATS];
            for d in 0..NUM_FEATS {
                if active[d] {
                    diff[d] = (x[d] - mean[d]) / sd[d] - class_means[ci][d];
                }
            }
            for i in 0..NUM_FEATS {
                if !active[i] {
                    continue;
                }
                let wi = w_sample * diff[i];
                let row = i * NUM_FEATS;
                for j in i..NUM_FEATS {
                    if active[j] {
                        sw[row + j] += wi * diff[j];
                    }
                }
            }
        }
        // Symmetrize and compute trace over active dimensions
        let mut trace = 0.0;
        for i in 0..NUM_FEATS {
            if active[i] {
                trace += sw[i * NUM_FEATS + i];
                for j in (i + 1)..NUM_FEATS {
                    sw[j * NUM_FEATS + i] = sw[i * NUM_FEATS + j];
                }
            }
        }
        let shrink = shrink.clamp(0.01, 0.99);
        let iso = (trace / active_count as f64).max(1e-4);
        let mut sw_reg = [0.0; NUM_FEATS * NUM_FEATS];
        for i in 0..NUM_FEATS {
            for j in 0..NUM_FEATS {
                let idx = i * NUM_FEATS + j;
                if active[i] && active[j] {
                    sw_reg[idx] = (1.0 - shrink) * sw[idx] + if i == j { shrink * iso } else { 0.0 };
                } else {
                    sw_reg[idx] = if i == j { 1.0 } else { 0.0 };
                }
            }
        }

        // Cholesky factorization Sw_reg = L L^T once, then solve for every class mean
        let chol = cholesky(&sw_reg)?;
        let mut weights = Vec::with_capacity(k);
        let mut biases = Vec::with_capacity(k);
        for ci in 0..k {
            let w = cholesky_solve(&chol, &class_means[ci]);
            let mut b = 0.0;
            for d in 0..NUM_FEATS {
                b += w[d] * class_means[ci][d];
            }
            weights.push(w);
            biases.push(-0.5 * b);
        }

        Some(Self { mean, sd, active, classes, weights, biases })
    }

    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Evaluates linear discriminant scores `w_k · xs + b_k` into `out` without
    /// heap allocation. Returns the number of entries written.
    pub fn scores(&self, x: &[f64; NUM_FEATS], out: &mut [(usize, f64); MAX_LDA_CLASSES]) -> usize {
        let mut xs = [0.0; NUM_FEATS];
        for d in 0..NUM_FEATS {
            if self.active[d] {
                xs[d] = (x[d] - self.mean[d]) / self.sd[d];
            }
        }
        let k = self.classes.len().min(MAX_LDA_CLASSES);
        for ci in 0..k {
            let w = &self.weights[ci];
            let mut s = self.biases[ci];
            for d in 0..NUM_FEATS {
                s += w[d] * xs[d];
            }
            out[ci] = (self.classes[ci], s);
        }
        k
    }
}

/// Lower-triangular Cholesky factor `L` such that `L L^T = A` for SPD `A`.
fn cholesky(a: &[f64; NUM_FEATS * NUM_FEATS]) -> Option<[f64; NUM_FEATS * NUM_FEATS]> {
    let mut l = [0.0; NUM_FEATS * NUM_FEATS];
    for i in 0..NUM_FEATS {
        for j in 0..=i {
            let mut sum = a[i * NUM_FEATS + j];
            for k in 0..j {
                sum -= l[i * NUM_FEATS + k] * l[j * NUM_FEATS + k];
            }
            if i == j {
                if sum <= 1e-12 || !sum.is_finite() {
                    return None;
                }
                l[i * NUM_FEATS + j] = sum.sqrt();
            } else {
                l[i * NUM_FEATS + j] = sum / l[j * NUM_FEATS + j];
            }
        }
    }
    Some(l)
}

/// Solves `(L L^T) x = b` given lower-triangular `L`.
fn cholesky_solve(l: &[f64; NUM_FEATS * NUM_FEATS], b: &[f64; NUM_FEATS]) -> [f64; NUM_FEATS] {
    let mut y = [0.0; NUM_FEATS];
    for i in 0..NUM_FEATS {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * NUM_FEATS + k] * y[k];
        }
        y[i] = sum / l[i * NUM_FEATS + i];
    }
    let mut x = [0.0; NUM_FEATS];
    for i in (0..NUM_FEATS).rev() {
        let mut sum = y[i];
        for k in (i + 1)..NUM_FEATS {
            sum -= l[k * NUM_FEATS + i] * x[k];
        }
        x[i] = sum / l[i * NUM_FEATS + i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u32) -> f64 {
        *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (*s as f64 / u32::MAX as f64) * 2.0 - 1.0
    }

    #[test]
    fn cov_tracker_matches_direct_covariance_and_recovers_correlation() {
        let mut tr = CovTracker::new();
        let mut rng = 42u32;
        let mut buf = Vec::new();
        for _ in 0..120 {
            let shared = 20.0 * lcg(&mut rng);
            let mut s = [0.0f32; NUM_CHANNELS];
            s[0] = (shared + 2.0 * lcg(&mut rng)) as f32;
            s[1] = (shared + 2.0 * lcg(&mut rng)) as f32;
            s[2] = (-shared + 2.0 * lcg(&mut rng)) as f32;
            for c in 3..NUM_CHANNELS {
                s[c] = (5.0 * lcg(&mut rng)) as f32;
            }
            tr.push(&s);
            buf.push(s);
        }
        let cov = tr.window();
        let tail = &buf[buf.len() - COV_WIN..];
        let direct_00: f64 = tail.iter().map(|s| (s[0] as f64).powi(2)).sum::<f64>() / COV_WIN as f64;
        let direct_01: f64 = tail.iter().map(|s| s[0] as f64 * s[1] as f64).sum::<f64>() / COV_WIN as f64;
        assert!((cov[0] - direct_00).abs() < 1e-6);
        assert!((cov[off_index(0, 1)] - direct_01).abs() < 1e-6);

        let usable = [true; NUM_CHANNELS];
        let (f, _) = extract_features(&cov, &usable, 0.0);
        // CH0 & CH1 positively correlated, CH0 & CH2 negatively correlated
        assert!(f[off_index(0, 1)] > 1.0, "expected strong +corr, got {}", f[off_index(0, 1)]);
        assert!(f[off_index(0, 2)] < -1.0, "expected strong -corr, got {}", f[off_index(0, 2)]);
    }

    #[test]
    fn rel_logrms_and_correlation_are_effort_invariant() {
        let mut tr1 = CovTracker::new();
        let mut tr2 = CovTracker::new();
        let mut rng = 99u32;
        for _ in 0..COV_WIN {
            // realistic EMG amplitudes (tens of µV), where the COV_EPS floor is negligible
            let a = 10.0 * lcg(&mut rng);
            let b = 10.0 * lcg(&mut rng);
            let base = [
                (15.0 * a + 2.0 * b) as f32,
                (10.0 * a - 3.0 * b) as f32,
                (5.0 * b) as f32,
                (4.0 * a + 4.0 * b) as f32,
                (3.0 * b) as f32,
                (2.0 * a) as f32,
                (6.0 * a + 1.0 * b) as f32,
                (2.0 * b) as f32,
            ];
            let scaled = base.map(|v| v * 3.0);
            tr1.push(&base);
            tr2.push(&scaled);
        }
        let usable = [true; NUM_CHANNELS];
        let (f1, _) = extract_features(&tr1.window(), &usable, 0.0);
        let (f2, _) = extract_features(&tr2.window(), &usable, 0.0);
        // First 36 features (rel_logrms + corr) stay very close under 3x effort scaling
        for d in 0..(NUM_FEATS - 1) {
            assert!((f1[d] - f2[d]).abs() < 0.08, "dim {d}: {} vs {}", f1[d], f2[d]);
        }
    }

    #[test]
    fn shrinkage_lda_separates_correlated_classes_and_ignores_masked_channels() {
        let mut rng = 7u32;
        let mut samples = Vec::new();
        // Two classes with identical channel amplitudes but opposite CH0-CH1 correlation!
        for cls in 0..2 {
            let sign = if cls == 0 { 1.0 } else { -1.0 };
            let mut tr = CovTracker::new();
            for t in 0..250 {
                let src = 20.0 * lcg(&mut rng);
                let mut s = [0.0f32; NUM_CHANNELS];
                s[0] = (src + 3.0 * lcg(&mut rng)) as f32;
                s[1] = (sign * src + 3.0 * lcg(&mut rng)) as f32;
                for c in 2..NUM_CHANNELS {
                    s[c] = (5.0 * lcg(&mut rng)) as f32;
                }
                tr.push(&s);
                if t >= COV_WIN && t % 10 == 0 {
                    let mut usable = [true; NUM_CHANNELS];
                    usable[7] = false; // masked channel
                    let (f, _) = extract_features(&tr.window(), &usable, 0.0);
                    samples.push((cls, f));
                }
            }
        }
        let model = LdaModel::fit(&samples, 0.25).expect("LDA fits");
        let mut out = [(0, 0.0); MAX_LDA_CLASSES];
        let mut hits = 0usize;
        for (cls, f) in &samples {
            let k = model.scores(f, &mut out);
            let best = out[..k].iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
            if best == *cls {
                hits += 1;
            }
        }
        assert_eq!(hits, samples.len(), "correlation-only classes should separate 100%");
    }

    #[test]
    fn nnls_recovers_a_non_negative_mix() {
        // A = 3 patterns over 4 channels; y = 2·a0 + 0.5·a2
        let a = [[1.0, 0.2, 0.0, 0.1], [0.3, 1.0, 0.2, 0.0], [0.0, 0.1, 1.0, 0.4]];
        let y: Vec<f64> = (0..4).map(|c| 2.0 * a[0][c] + 0.5 * a[2][c]).collect();
        let mut g = [[0.0; NNLS_MAX]; 3];
        for i in 0..3 {
            for j in 0..3 {
                g[i][j] = (0..4).map(|c| a[i][c] * a[j][c]).sum();
            }
        }
        let h: Vec<f64> = (0..3).map(|i| (0..4).map(|c| a[i][c] * y[c]).sum()).collect();
        let w = nnls(&g, &h);
        assert!((w[0] - 2.0).abs() < 1e-9 && w[1].abs() < 1e-9 && (w[2] - 0.5).abs() < 1e-9, "{w:?}");
        // a target needing a negative weight is clipped to zero, not made negative
        let h2: Vec<f64> = (0..3).map(|i| -(0..4).map(|c| a[i][c] * a[1][c]).sum::<f64>()).collect();
        assert!(nnls(&g, &h2).iter().all(|&v| v == 0.0));
    }
}
