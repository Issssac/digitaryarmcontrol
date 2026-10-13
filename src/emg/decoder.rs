//! Non-negative least squares, shared by the gesture and finger decoders: each
//! sample's activity is explained as a non-negative mix of calibrated patterns.

use super::cyton::NUM_CHANNELS;

pub type Features = [f64; NUM_CHANNELS];

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
