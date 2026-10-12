//! Movement-intention decoding from the pattern across all channels, trained on
//! the calibration recordings (rest plus one pulse per movement):
//!
//! - `Synergy` drives the arm. It explains the activity as a non-negative mix of
//!   the calibrated movement patterns, so crosstalk is accounted for and two
//!   movements at once come out as both, each with its own intensity.
//! - `Imputer` estimates channels whose electrode is offline from the working
//!   ones. It is a class-conditional Gaussian over log envelopes with a pooled,
//!   shrunk covariance, i.e. the LDA model.

use serde::{Deserialize, Serialize};

use super::cyton::NUM_CHANNELS;

pub type Features = [f64; NUM_CHANNELS];

/// Added to the envelope (µV) before the log so silent channels stay finite.
const EPS_UV: f64 = 0.5;
/// Covariance shrinkage towards a scaled identity: keeps the inverse stable with
/// few, strongly correlated samples and with channels that have no contact.
const SHRINKAGE: f64 = 0.15;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Label {
    Rest,
    /// The movement calibrated on this channel's electrode site.
    Move(usize),
}

pub fn features(env: &[f32; NUM_CHANNELS]) -> Features {
    env.map(|e| (e.max(0.0) as f64 + EPS_UV).ln())
}

/// Inverse of `features` for one channel.
pub fn envelope_from_feature(x: f64) -> f32 {
    (x.exp() - EPS_UV).max(0.0) as f32
}

fn dot(a: &Features, b: &Features) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn mean(xs: &[Features]) -> Features {
    let mut m = [0.0; NUM_CHANNELS];
    for x in xs {
        for (mi, xi) in m.iter_mut().zip(x) {
            *mi += xi;
        }
    }
    m.map(|v| v / xs.len() as f64)
}

type Matrix = [[f64; NUM_CHANNELS]; NUM_CHANNELS];

/// Gauss-Jordan with partial pivoting; None if singular.
fn invert(mut a: Matrix) -> Option<Matrix> {
    let n = NUM_CHANNELS;
    let mut inv = [[0.0; NUM_CHANNELS]; NUM_CHANNELS];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let d = a[col][col];
        for j in 0..n {
            a[col][j] /= d;
            inv[col][j] /= d;
        }
        for i in 0..n {
            if i != col {
                let f = a[i][col];
                for j in 0..n {
                    a[i][j] -= f * a[col][j];
                    inv[i][j] -= f * inv[col][j];
                }
            }
        }
    }
    Some(inv)
}

pub struct Imputer {
    /// Mean log envelope of each class (rest and every movement).
    means: Vec<Features>,
    /// Pooled within-class covariance (shrunk).
    cov: Matrix,
    /// Imputation weights for the last set of missing channels.
    impute_cache: Option<([bool; NUM_CHANNELS], Imputation)>,
}

/// Precomputed for one set of missing channels.
struct Imputation {
    /// Σ_oo⁻¹ embedded in a full matrix (zero outside the observed block).
    inv_oo: Matrix,
    /// Per missing channel m: Σ_mo Σ_oo⁻¹ (zero for missing columns).
    rows: Vec<(usize, Features)>,
}

/// Fewer working channels than this and there is too little to guess from.
const IMPUTE_MIN_OBSERVED: usize = 2;

impl Imputer {
    /// Needs a rest class and at least one movement, each with a few samples.
    pub fn train(data: &[(Label, Vec<Features>)]) -> Option<Self> {
        let classes: Vec<&(Label, Vec<Features>)> = data.iter().filter(|(_, xs)| xs.len() >= 2).collect();
        classes.iter().position(|(l, _)| *l == Label::Rest)?;
        if classes.len() < 2 {
            return None;
        }
        let means: Vec<Features> = classes.iter().map(|(_, xs)| mean(xs)).collect();

        // pooled within-class covariance, shrunk towards a scaled identity
        let mut cov = [[0.0; NUM_CHANNELS]; NUM_CHANNELS];
        let mut count = 0;
        for ((_, xs), m) in classes.iter().zip(&means) {
            for x in xs {
                let d: Features = std::array::from_fn(|i| x[i] - m[i]);
                for i in 0..NUM_CHANNELS {
                    for j in 0..NUM_CHANNELS {
                        cov[i][j] += d[i] * d[j];
                    }
                }
            }
            count += xs.len();
        }
        let dof = count.saturating_sub(classes.len()).max(1) as f64;
        let avg_var = ((0..NUM_CHANNELS).map(|i| cov[i][i]).sum::<f64>() / dof / NUM_CHANNELS as f64).max(1e-6);
        for (i, row) in cov.iter_mut().enumerate() {
            for (j, c) in row.iter_mut().enumerate() {
                *c = *c / dof * (1.0 - SHRINKAGE) + if i == j { SHRINKAGE * avg_var } else { 0.0 };
            }
        }
        invert(cov)?; // must be well-conditioned
        Some(Self { means, cov, impute_cache: None })
    }

    fn imputation(&mut self, missing: &[bool; NUM_CHANNELS]) -> Option<&Imputation> {
        if self.impute_cache.as_ref().map(|(m, _)| m) != Some(missing) {
            // Σ with the missing rows/columns replaced by identity: its inverse
            // holds Σ_oo⁻¹ in the observed block.
            let mut a = self.cov;
            for m in (0..NUM_CHANNELS).filter(|&c| missing[c]) {
                for k in 0..NUM_CHANNELS {
                    a[m][k] = 0.0;
                    a[k][m] = 0.0;
                }
                a[m][m] = 1.0;
            }
            let mut inv_oo = invert(a)?;
            for m in (0..NUM_CHANNELS).filter(|&c| missing[c]) {
                inv_oo[m][m] = 0.0;
            }
            let rows = (0..NUM_CHANNELS)
                .filter(|&m| missing[m])
                .map(|m| (m, std::array::from_fn(|o| (0..NUM_CHANNELS).map(|k| self.cov[m][k] * inv_oo[k][o]).sum())))
                .collect();
            self.impute_cache = Some((*missing, Imputation { inv_oo, rows }));
        }
        self.impute_cache.as_ref().map(|(_, imp)| imp)
    }

    /// Estimates the channels whose electrode lost contact from the ones that
    /// still work. First the working channels alone say how likely each class is;
    /// then each missing channel is that class mixture of "what this channel does
    /// in class k, given what the others are doing" (the Gaussian conditional
    /// mean μ_k,m + Σ_mo Σ_oo⁻¹ (x_o − μ_k,o)). Fills `x` in place; false if
    /// too few channels are left to go on.
    pub fn impute(&mut self, x: &mut Features, missing: &[bool; NUM_CHANNELS]) -> bool {
        if missing.iter().filter(|&&m| !m).count() < IMPUTE_MIN_OBSERVED {
            return false;
        }
        let means = self.means.clone();
        let Some(imp) = self.imputation(missing) else { return false };
        let obs = |v: &Features| -> Features { std::array::from_fn(|i| if missing[i] { 0.0 } else { v[i] }) };
        let xo = obs(x);

        // class probabilities from the observed channels only
        let g: Vec<f64> = means
            .iter()
            .map(|m| {
                let mo = obs(m);
                let w: Features = std::array::from_fn(|i| dot(&imp.inv_oo[i], &mo));
                dot(&w, &xo) - 0.5 * dot(&w, &mo)
            })
            .collect();
        let top = g.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = g.iter().map(|v| (v - top).exp()).collect();
        let sum: f64 = e.iter().sum();

        for (m, w) in &imp.rows {
            x[*m] = means
                .iter()
                .zip(&e)
                .map(|(mu, p)| {
                    let resid: Features = std::array::from_fn(|o| if missing[o] { 0.0 } else { xo[o] - mu[o] });
                    p / sum * (mu[*m] + dot(w, &resid))
                })
                .sum();
        }
        true
    }
}

/// Small dense linear solve (Gaussian elimination with partial pivoting).
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
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
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|j| a[i][j] * x[j]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    Some(x)
}

/// Non-negative least squares in normal-equation form: minimise ‖Aw − y‖ with
/// w ≥ 0, given G = AᵀA and h = Aᵀy (Lawson–Hanson active set).
pub fn nnls(g: &[Vec<f64>], h: &[f64]) -> Vec<f64> {
    let n = h.len();
    let mut w = vec![0.0; n];
    let mut passive = vec![false; n];
    for _ in 0..3 * n + 3 {
        let grad: Vec<f64> = (0..n).map(|i| h[i] - (0..n).map(|j| g[i][j] * w[j]).sum::<f64>()).collect();
        let Some(t) = (0..n).filter(|&i| !passive[i]).max_by(|&i, &j| grad[i].total_cmp(&grad[j])) else { break };
        if grad[t] <= 1e-10 {
            break;
        }
        passive[t] = true;
        loop {
            let idx: Vec<usize> = (0..n).filter(|&i| passive[i]).collect();
            let sub = idx.iter().map(|&i| idx.iter().map(|&j| g[i][j]).collect()).collect();
            let Some(sp) = solve(sub, idx.iter().map(|&i| h[i]).collect()) else { return w };
            if sp.iter().all(|&v| v > 0.0) {
                for (k, &i) in idx.iter().enumerate() {
                    w[i] = sp[k];
                }
                break;
            }
            // step towards the unconstrained solution until a weight hits zero
            let alpha = idx
                .iter()
                .enumerate()
                .filter(|&(k, _)| sp[k] <= 0.0)
                .map(|(k, &i)| w[i] / (w[i] - sp[k]))
                .fold(1.0, f64::min);
            for (k, &i) in idx.iter().enumerate() {
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

/// Simultaneous decoder for compound movements. Each sample's activity above
/// rest is explained as a non-negative mix of the calibrated movement patterns
/// ("synergies"), so two movements at once come out as both, each with its own
/// intensity (1 = the calibrated pulse). Similar patterns, such as wrist flexors
/// and extensors with heavy crosstalk, share the weight instead of flipping
/// between winners.
///
/// Channels are scaled by their resting noise so a noisy electrode can't
/// dominate the fit, and offline channels are simply left out of it.
pub struct Synergy {
    /// Channel (electrode site) of each movement.
    channels: Vec<usize>,
    /// Mean pattern of each movement, in units of resting noise above the rest mean.
    templates: Vec<Features>,
    rest_mean: Features,
    rest_scale: Features,
}

/// Relative ridge on the fit: keeps near-identical patterns from producing wild,
/// opposite weights.
const SYNERGY_RIDGE: f64 = 0.02;
/// A held-out sample counts as "rest" in the report if no movement reaches this.
const SYNERGY_REST_LEVEL: f64 = 0.35;

impl Synergy {
    pub fn train(data: &[(Label, Vec<Features>)]) -> Option<Self> {
        let env = |x: &Features| -> Features { x.map(|v| envelope_from_feature(v) as f64) };
        let rest: Vec<Features> = data.iter().find(|(l, _)| *l == Label::Rest)?.1.iter().map(env).collect();
        if rest.len() < 2 {
            return None;
        }
        let rest_mean = mean(&rest);
        let rest_scale: Features = std::array::from_fn(|ch| {
            let var = rest.iter().map(|r| (r[ch] - rest_mean[ch]).powi(2)).sum::<f64>() / (rest.len() - 1) as f64;
            var.sqrt().max(0.2 * rest_mean[ch]).max(0.3)
        });
        let mut channels = Vec::new();
        let mut templates = Vec::new();
        for (label, xs) in data {
            let Label::Move(ch) = *label else { continue };
            if xs.is_empty() {
                continue;
            }
            let z: Vec<Features> = xs.iter().map(|x| Self::normalise(&env(x), &rest_mean, &rest_scale)).collect();
            channels.push(ch);
            templates.push(mean(&z));
        }
        (!channels.is_empty()).then_some(Self { channels, templates, rest_mean, rest_scale })
    }

    fn normalise(env: &Features, mean: &Features, scale: &Features) -> Features {
        std::array::from_fn(|ch| ((env[ch] - mean[ch]) / scale[ch]).max(0.0))
    }

    /// Channel of each movement, in the order `decompose` returns weights.
    pub fn channels(&self) -> &[usize] {
        &self.channels
    }

    /// Weight of each movement (≈1 at the calibrated pulse), fitted on the
    /// `observed` channels only.
    pub fn decompose(&self, env: &[f32; NUM_CHANNELS], observed: &[bool; NUM_CHANNELS]) -> Vec<f64> {
        let e: Features = env.map(|v| v as f64);
        let z = Self::normalise(&e, &self.rest_mean, &self.rest_scale);
        let n = self.templates.len();
        let rows = || (0..NUM_CHANNELS).filter(|&ch| observed[ch]);
        let mut g: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| rows().map(|ch| self.templates[i][ch] * self.templates[j][ch]).sum()).collect())
            .collect();
        let ridge = SYNERGY_RIDGE * (0..n).map(|i| g[i][i]).sum::<f64>() / n.max(1) as f64;
        for (i, row) in g.iter_mut().enumerate() {
            row[i] += ridge;
        }
        let h: Vec<f64> = (0..n).map(|i| rows().map(|ch| self.templates[i][ch] * z[ch]).sum()).collect();
        nnls(&g, &h)
    }
}

/// Recognition per class for the synergy decoder, trained on one half of every
/// recording and tested on the other (and swapped). A sample is read as its
/// strongest movement, or as rest if none reaches SYNERGY_REST_LEVEL.
pub fn cross_validate_synergy(data: &[(Label, Vec<Features>)]) -> Vec<ClassReport> {
    let half = |first: bool| -> Vec<(Label, Vec<Features>)> {
        data.iter()
            .map(|(l, xs)| {
                let h = xs.len() / 2;
                (*l, if first { xs[..h].to_vec() } else { xs[h..].to_vec() })
            })
            .collect()
    };
    let (a, b) = (half(true), half(false));
    let all = [true; NUM_CHANNELS];
    let mut tally: Vec<(usize, usize, Vec<(Label, usize)>)> = data.iter().map(|_| (0, 0, Vec::new())).collect();
    for (train, test) in [(&a, &b), (&b, &a)] {
        let Some(syn) = Synergy::train(train) else { continue };
        for (ci, (label, xs)) in test.iter().enumerate() {
            for x in xs {
                let env = x.map(envelope_from_feature);
                let w = syn.decompose(&env, &all);
                let k = argmax(&w);
                let got = if w.get(k).copied().unwrap_or(0.0) < SYNERGY_REST_LEVEL { Label::Rest } else { Label::Move(syn.channels[k]) };
                let t = &mut tally[ci];
                t.0 += 1;
                if got == *label {
                    t.1 += 1;
                } else if let Some(e) = t.2.iter_mut().find(|(l, _)| *l == got) {
                    e.1 += 1;
                } else {
                    t.2.push((got, 1));
                }
            }
        }
    }
    data.iter()
        .zip(tally)
        .map(|((label, _), (tested, correct, mistakes))| ClassReport {
            label: *label,
            accuracy: if tested > 0 { correct as f64 / tested as f64 } else { 0.0 },
            confused_with: mistakes.iter().max_by_key(|(_, n)| *n).map(|(l, _)| *l),
        })
        .collect()
}

pub fn argmax(v: &[f64]) -> usize {
    (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap_or(0)
}

/// How well a class is recognised on held-out data (see cross_validate_synergy):
/// its accuracy and the class it is most often mistaken for.
pub struct ClassReport {
    pub label: Label,
    pub accuracy: f64,
    pub confused_with: Option<Label>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(state: &mut u32) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        (*state as f64 / u32::MAX as f64) * 2.0 - 1.0
    }

    /// Envelope pattern for a movement: strong on its own channel, but with heavy
    /// overlap onto two neighbours, like real crosstalk.
    fn pattern(class: Option<usize>, scale: f32, rng: &mut u32) -> [f32; NUM_CHANNELS] {
        std::array::from_fn(|ch| {
            let base = 3.0 * (1.0 + 0.3 * noise(rng) as f32);
            let drive = match class {
                Some(c) if ch == c => 1.0,
                Some(c) if ch == (c + 1) % NUM_CHANNELS => 0.7,
                Some(c) if ch == (c + 3) % NUM_CHANNELS => 0.5,
                _ => 0.0,
            };
            base + scale * drive * 40.0 * (1.0 + 0.25 * noise(rng) as f32)
        })
    }

    fn dataset(rng: &mut u32) -> Vec<(Label, Vec<Features>)> {
        let mut data = vec![(Label::Rest, (0..600).map(|_| features(&pattern(None, 0.0, rng))).collect())];
        for c in 0..NUM_CHANNELS {
            data.push((Label::Move(c), (0..400).map(|_| features(&pattern(Some(c), 1.0, rng))).collect()));
        }
        data
    }

    #[test]
    fn cross_validation_reports_accuracy_and_confusion() {
        let mut rng = 2;
        let mut data = dataset(&mut rng);
        // make movement 6 a near copy of movement 5: they should be reported as confused
        data[7].1 = (0..400).map(|_| features(&pattern(Some(5), 1.0, &mut rng))).collect();
        let report = cross_validate_synergy(&data);
        let r0 = report.iter().find(|r| r.label == Label::Move(0)).unwrap();
        assert!(r0.accuracy > 0.95, "{}", r0.accuracy);
        let r6 = report.iter().find(|r| r.label == Label::Move(6)).unwrap();
        assert!(r6.accuracy < 0.8);
        assert_eq!(r6.confused_with, Some(Label::Move(5)));
    }

    #[test]
    fn imputed_channel_lets_the_movement_still_be_decoded() {
        let mut rng = 4;
        let data = dataset(&mut rng);
        let mut imp = Imputer::train(&data).unwrap();
        let syn = Synergy::train(&data).unwrap();
        let mut missing = [false; NUM_CHANNELS];
        missing[3] = true;
        for c in 0..NUM_CHANNELS {
            let mut x = features(&pattern(Some(c), 1.0, &mut rng));
            x[3] = features(&[0.0; NUM_CHANNELS])[3]; // electrode off reads as silence
            assert!(imp.impute(&mut x, &missing));
            let w = syn.decompose(&x.map(envelope_from_feature), &[true; NUM_CHANNELS]);
            assert_eq!(syn.channels()[argmax(&w)], c, "movement {c}: {w:?}");
            if c == 3 {
                // its own electrode is the one missing: the estimate must still say "active"
                assert!(envelope_from_feature(x[3]) > 15.0, "estimated {}", envelope_from_feature(x[3]));
            }
        }
        let all_missing = [true; NUM_CHANNELS];
        let mut x = [0.0; NUM_CHANNELS];
        assert!(!imp.impute(&mut x, &all_missing));
    }

    #[test]
    fn nnls_recovers_a_non_negative_mix() {
        // A = 3 patterns over 4 channels; y = 2·a0 + 0.5·a2
        let a = [[1.0, 0.2, 0.0, 0.1], [0.3, 1.0, 0.2, 0.0], [0.0, 0.1, 1.0, 0.4]];
        let y: Vec<f64> = (0..4).map(|c| 2.0 * a[0][c] + 0.5 * a[2][c]).collect();
        let g: Vec<Vec<f64>> = (0..3).map(|i| (0..3).map(|j| (0..4).map(|c| a[i][c] * a[j][c]).sum()).collect()).collect();
        let h: Vec<f64> = (0..3).map(|i| (0..4).map(|c| a[i][c] * y[c]).sum()).collect();
        let w = nnls(&g, &h);
        assert!((w[0] - 2.0).abs() < 1e-9 && w[1].abs() < 1e-9 && (w[2] - 0.5).abs() < 1e-9, "{w:?}");
        // a target needing a negative weight is clipped to zero, not made negative
        let h2: Vec<f64> = (0..3).map(|i| -(0..4).map(|c| a[i][c] * a[1][c]).sum::<f64>()).collect();
        assert!(nnls(&g, &h2).iter().all(|&v| v == 0.0));
    }

    /// Linear envelope for a mix of movements (drive per class), same crosstalk as `pattern`.
    fn mix(drives: &[(usize, f32)], rng: &mut u32) -> [f32; NUM_CHANNELS] {
        std::array::from_fn(|ch| {
            let base = 3.0 * (1.0 + 0.3 * noise(rng) as f32);
            let d: f32 = drives
                .iter()
                .map(|&(c, s)| {
                    s * if ch == c {
                        1.0
                    } else if ch == (c + 1) % NUM_CHANNELS {
                        0.7
                    } else if ch == (c + 3) % NUM_CHANNELS {
                        0.5
                    } else {
                        0.0
                    }
                })
                .sum();
            base + d * 40.0 * (1.0 + 0.15 * noise(rng) as f32)
        })
    }

    #[test]
    fn synergy_decodes_single_and_compound_movements() {
        let mut rng = 6;
        let data = dataset(&mut rng);
        let syn = Synergy::train(&data).unwrap();
        let all = [true; NUM_CHANNELS];
        let idx = |ch: usize| syn.channels().iter().position(|&c| c == ch).unwrap();
        // single movement at half strength: one clear weight
        let w = syn.decompose(&mix(&[(4, 0.5)], &mut rng), &all);
        assert!((w[idx(4)] - 0.5).abs() < 0.15, "{w:?}");
        assert!(w.iter().enumerate().all(|(i, &v)| i == idx(4) || v < 0.15), "{w:?}");
        // two at once: both come out, each near its own intensity
        let w = syn.decompose(&mix(&[(1, 0.8), (6, 0.6)], &mut rng), &all);
        assert!((w[idx(1)] - 0.8).abs() < 0.2 && (w[idx(6)] - 0.6).abs() < 0.2, "{w:?}");
        assert!(w.iter().enumerate().all(|(i, &v)| i == idx(1) || i == idx(6) || v < 0.2), "{w:?}");
        // at rest: nothing
        let w = syn.decompose(&mix(&[], &mut rng), &all);
        assert!(w.iter().all(|&v| v < 0.15), "{w:?}");
    }

    #[test]
    fn synergy_separates_heavily_overlapping_antagonists() {
        // wrist flexors (6) and extensors (7) each show 70% on the other electrode
        let mut rng = 8;
        let pat = |c: usize, rng: &mut u32| -> [f32; NUM_CHANNELS] {
            std::array::from_fn(|ch| {
                let base = 3.0 * (1.0 + 0.3 * noise(rng) as f32);
                let d = match (c, ch) {
                    (6, 6) | (7, 7) => 1.0,
                    (6, 7) | (7, 6) => 0.7,
                    _ => 0.0,
                };
                base + d * 40.0 * (1.0 + 0.2 * noise(rng) as f32)
            })
        };
        let data = vec![
            (Label::Rest, (0..600).map(|_| features(&pat(99, &mut rng))).collect()),
            (Label::Move(6), (0..400).map(|_| features(&pat(6, &mut rng))).collect()),
            (Label::Move(7), (0..400).map(|_| features(&pat(7, &mut rng))).collect()),
        ];
        let report = cross_validate_synergy(&data);
        for r in &report {
            assert!(r.accuracy > 0.9, "{:?}: {}", r.label, r.accuracy);
        }
    }

    #[test]
    fn needs_rest_and_a_movement() {
        let mut rng = 3;
        let data = dataset(&mut rng);
        assert!(Imputer::train(&data[1..]).is_none());
        assert!(Imputer::train(&data[..1]).is_none());
        assert!(Imputer::train(&data[..2]).is_some());
    }
}
