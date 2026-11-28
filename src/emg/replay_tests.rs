//! Replays the labelled calibration recordings in `recordings/` through the real
//! pipeline and decoder: calibrates on the first part of every hold, then scores
//! the rest of it (and the rest of every rest posture) in live decoding.
//!
//!     cargo test --release replay -- --ignored --nocapture

use super::*;
use crate::emg::pipeline::EmgPipeline;
use std::path::PathBuf;

const SESSIONS: [&str; 3] = ["20261124-142454-5x0i-julia1", "20261124-150008-8hgb-rafa1", "20261127-174918-zzcm-andy-right-arm-1"];
/// Calibration takes at most this much of each hold, and the test at most this much after it.
const CAL_MAX: usize = 1250;
const TEST_MAX: usize = 1250;
/// Samples left out between the calibration and test parts (longer than the 200 ms window).
const GAP: usize = 125;

struct Row {
    raw: [f32; NUM_CHANNELS],
    step: String,
    movement: i64,
    record: bool,
    check: String,
    posture: i64,
}

fn load(name: &str) -> Option<(Vec<Row>, bool)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("recordings").join(name);
    let text = std::fs::read_to_string(dir.join("samples.csv")).ok()?;
    let meta = std::fs::read_to_string(dir.join("meta.json")).ok()?;
    let car = meta.contains("\"car\":true") || meta.contains("\"car\": true");
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next()?.split(',').collect();
    let col = |n: &str| head.iter().position(|h| *h == n).expect(n);
    let (ch0, step, mv, phase, check, posture, rec) = (
        col("ch1_uV"),
        col("step_name"),
        col("movement"),
        col("phase_name"),
        col("check_name"),
        col("posture"),
        col("recording"),
    );
    let rows = lines
        .filter_map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (f.len() >= head.len()).then(|| Row {
                raw: std::array::from_fn(|c| f[ch0 + c].parse().unwrap_or(0.0)),
                step: f[step].to_string(),
                movement: f[mv].parse().unwrap_or(-1),
                record: f[phase] == "record" && f[rec] == "1",
                check: f[check].to_string(),
                posture: match f[posture] {
                    "limp" => 0,
                    "forearm raised" => 1,
                    "arm up" => 2,
                    s => s.parse().unwrap_or(-1),
                },
            })
        })
        .collect();
    Some((rows, car))
}

struct Segment {
    step: Step,
    cal: Vec<usize>,
    test: Vec<usize>,
}

/// Contiguous recording stretches of one step; the samples the coach would keep
/// (after the settle time; for half-strength, also those it took for another
/// movement, as the coach now does), split into calibration and test.
fn segments(rows: &[Row]) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < rows.len() {
        if !rows[i].record {
            i += 1;
            continue;
        }
        let key = (&rows[i].step, rows[i].movement, rows[i].posture);
        let start = i;
        while i < rows.len() && rows[i].record && (&rows[i].step, rows[i].movement, rows[i].posture) == key {
            i += 1;
        }
        let g = rows[start].movement.max(0) as usize;
        let step = match rows[start].step.as_str() {
            "rest" => Step::Rest(rows[start].posture.max(0) as usize),
            "hard" => Step::Hard(g),
            "medium" => Step::Medium(g),
            _ => continue,
        };
        let keep: Vec<usize> = (start + samples(SETTLE_S) as usize..i)
            .filter(|&k| match step {
                Step::Medium(_) => rows[k].check == "clean" || rows[k].check == "other",
                _ => rows[k].check == "clean",
            })
            .collect();
        if keep.len() < 2 * GAP {
            continue;
        }
        let n_cal = (keep.len() / 2).min(CAL_MAX);
        let test = keep[(n_cal + GAP).min(keep.len())..].iter().copied().take(TEST_MAX).collect();
        out.push(Segment { step, cal: keep[..n_cal].to_vec(), test });
    }
    out
}

fn run_for(step: Step, rec: Vec<Features>, feats: Vec<f32>, g: usize) -> Run {
    Run {
        step,
        phase: Phase::Record,
        ready: true,
        t: 0,
        good: 0,
        stuck: 0,
        rec,
        feats,
        next: None,
        rests: POSTURES,
        calibrate: (0..g).collect(),
        train: Vec::new(),
        train_after: false,
        trial: 0,
        hold: 0,
        holds: 0,
        streak: 0,
    }
}

#[derive(Default, Clone, Copy)]
struct Score {
    hit: f64,
    wrong: f64,
    miss: f64,
    ident: f64,
    /// Share of the test samples each movement was driving, and was the strongest.
    on: [f64; NUM_DIGITS],
    top: [f64; NUM_DIGITS],
}

/// Calibrates a fresh digit decoder on the calibration parts, then scores the test parts.
fn evaluate(rows: &[Row], car: bool, lda: bool, scale: f32) -> ([Score; NUM_DIGITS], [f64; POSTURES], [bool; NUM_CHANNELS]) {
    const G: usize = NUM_DIGITS;
    let segs = segments(rows);
    let mut role: Vec<Option<(usize, bool)>> = vec![None; rows.len()];
    for (s, seg) in segs.iter().enumerate() {
        seg.cal.iter().for_each(|&k| role[k] = Some((s, true)));
        seg.test.iter().for_each(|&k| role[k] = Some((s, false)));
    }
    let last_cal: Vec<usize> = segs.iter().map(|s| *s.cal.last().unwrap()).collect();

    let mut p = EmgPipeline::new();
    p.car = car;
    let mut gc = GestureControl::<G>::digits();
    gc.learn_in_use = false;
    gc.use_lda = lda;
    let mut rec: Vec<Features> = Vec::new();
    let mut feats: Vec<f32> = Vec::new();
    for (k, row) in rows.iter().enumerate() {
        p.process(&row.raw);
        p.set_excluded(gc.bad_channels());
        gc.push_filtered(&p.filtered());
        gc.process(&p.envelopes(), &p.contact());
        let Some((s, true)) = role[k] else { continue };
        let step = segs[s].step;
        rec.push(match step {
            Step::Rest(0) => gc.envelope.map(f64::from),
            Step::Rest(_) => gc.z_raw,
            _ => gc.z,
        });
        gc.record_feats(rec.len() - 1, &mut feats);
        if k == last_cal[s] {
            let mut run = run_for(step, std::mem::take(&mut rec), std::mem::take(&mut feats), G);
            match step {
                Step::Rest(q) => {
                    gc.finish_rest(&mut run, q);
                }
                _ => {
                    gc.finish_recording(&mut run);
                }
            }
        }
    }
    gc.finalize(&(0..G).collect::<Vec<_>>());
    let t0 = std::time::Instant::now();
    gc.update_stasis();
    gc.update_coupling();
    println!("    lda={lda} x{scale}: noise floors + coupling recomputed in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    let bad = gc.bad_channels();
    if std::env::var("REPLAY_DETAIL").is_ok() && scale == 1.0 {
        let seen = gc.seen();
        println!("    template cosines (lda={lda}):");
        for a in 0..G {
            let row: Vec<String> = (0..G).map(|b| match (gc.templates[a], gc.templates[b]) {
                (Some(x), Some(y)) => format!("{:5.2}", cosine(&x, &y, &seen)),
                _ => "    -".into(),
            }).collect();
            println!("      {a}: {}", row.join(" "));
        }
        for a in 0..G {
            if let Some(t) = gc.templates[a] {
                let n = t.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-9);
                println!("      pattern {a}: {:?}", t.map(|v| (v / n * 100.0).round() / 100.0));
            }
        }
    }

    let mut p = EmgPipeline::new();
    p.car = car;
    p.set_excluded(bad);
    gc.reset_state();
    // a fainter version of the session: the signal scaled down, over the limp rest's own noise
    let noise: Vec<[f32; NUM_CHANNELS]> = {
        let idx: Vec<usize> = segs.iter().filter(|s| s.step == Step::Rest(0)).flat_map(|s| s.cal.iter().chain(&s.test).copied()).collect();
        let n = idx.len().max(1) as f32;
        let m: [f32; NUM_CHANNELS] = std::array::from_fn(|c| idx.iter().map(|&k| rows[k].raw[c]).sum::<f32>() / n);
        idx.iter().map(|&k| std::array::from_fn(|c| rows[k].raw[c] - m[c])).collect()
    };
    let mut score = [Score::default(); G];
    let mut counts = [0usize; G];
    let mut rest_false = [0.0; POSTURES];
    let mut rest_n = [0usize; POSTURES];
    for (k, row) in rows.iter().enumerate() {
        let raw: [f32; NUM_CHANNELS] = if scale < 1.0 && !noise.is_empty() {
            let nz = &noise[k % noise.len()];
            std::array::from_fn(|c| row.raw[c] * scale + nz[c])
        } else {
            row.raw
        };
        p.process(&raw);
        gc.push_filtered(&p.filtered());
        gc.process(&p.envelopes(), &p.contact());
        let Some((s, false)) = role[k] else { continue };
        let active = gc.active();
        match segs[s].step {
            Step::Rest(q) => {
                rest_n[q] += 1;
                if active.iter().any(|&a| a) {
                    rest_false[q] += 1.0;
                }
            }
            Step::Hard(g) | Step::Medium(g) => {
                counts[g] += 1;
                let others = (0..G).any(|h| h != g && active[h]);
                let sc = &mut score[g];
                if others {
                    sc.wrong += 1.0;
                } else if active[g] {
                    sc.hit += 1.0;
                } else {
                    sc.miss += 1.0;
                }
                let strongest = (0..G)
                    .filter(|&h| gc.templates[h].is_some())
                    .max_by(|&a, &b| gc.excess(a).total_cmp(&gc.excess(b)));
                if strongest == Some(g) {
                    sc.ident += 1.0;
                }
                for h in 0..G {
                    sc.on[h] += active[h] as u8 as f64;
                }
                if let Some(h) = strongest {
                    sc.top[h] += 1.0;
                }
            }
            _ => {}
        }
    }
    for g in 0..G {
        let n = counts[g].max(1) as f64;
        let s = &mut score[g];
        (s.hit, s.wrong, s.miss, s.ident) = (s.hit / n, s.wrong / n, s.miss / n, s.ident / n);
        s.on = s.on.map(|v| v / n);
        s.top = s.top.map(|v| v / n);
    }
    let rest = std::array::from_fn(|q| rest_false[q] / rest_n[q].max(1) as f64);
    (score, rest, bad)
}

fn report(name: &str, label: &str, r: &([Score; NUM_DIGITS], [f64; POSTURES], [bool; NUM_CHANNELS])) -> f64 {
    let (score, rest, bad) = r;
    let mean = |f: fn(&Score) -> f64| score.iter().map(f).sum::<f64>() / NUM_DIGITS as f64;
    let hit = mean(|s| s.hit);
    println!(
        "{name:>8} {label:<9} hit {:.3}  wrong {:.3}  miss {:.3}  strongest {:.3}  rest false-active {:?}  bad channels {:?}",
        hit,
        mean(|s| s.wrong),
        mean(|s| s.miss),
        mean(|s| s.ident),
        rest.map(|v| (v * 1000.0).round() / 1000.0),
        (0..NUM_CHANNELS).filter(|&c| bad[c]).map(|c| c + 1).collect::<Vec<_>>(),
    );
    println!(
        "{:>18} per movement hit: {:?}",
        "",
        score.map(|s| (s.hit * 100.0).round() / 100.0)
    );
    if std::env::var("REPLAY_DETAIL").is_ok() {
        println!("{:>18} held movement -> share of samples each movement drove | was strongest", "");
        for (g, s) in score.iter().enumerate() {
            let f = |v: &[f64; NUM_DIGITS]| v.iter().map(|x| format!("{:4.0}", x * 100.0)).collect::<Vec<_>>().join("");
            println!("{:>20}: {} |{}", g, f(&s.on), f(&s.top));
        }
    }
    hit
}

#[test]
#[ignore]
fn replay_recordings() {
    for name in SESSIONS {
        if std::env::var("REPLAY_ONLY").is_ok_and(|only| !name.contains(&only)) {
            continue;
        }
        let Some((rows, car)) = load(name) else {
            println!("{name}: not found, skipped");
            continue;
        };
        let short = name.rsplit('-').next().unwrap();
        let before = report(short, "patterns", &evaluate(&rows, car, false, 1.0));
        let after = report(short, "+LDA", &evaluate(&rows, car, true, 1.0));
        println!("{:>18} hit rate {:.3} -> {:.3}\n", "", before, after);
        for scale in [0.5f32, 0.3, 0.15] {
            let b = report(short, &format!("pat x{scale}"), &evaluate(&rows, car, false, scale));
            let a = report(short, &format!("LDA x{scale}"), &evaluate(&rows, car, true, scale));
            println!("{:>18} at x{scale}: hit rate {:.3} -> {:.3}\n", "", b, a);
        }
    }
}
