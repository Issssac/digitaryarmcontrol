"""Offline decoder from labelled calibration recordings, with contrastive miss detection.

The coach labels every sample with the step it was in (rest / hard / medium / faint / train)
and the movement it asked for. Those labels are noisy: during a hold the user is often not
doing the asked movement (late start, early release, drifting into a neighbour, relaxing).
The idea here: windows on the intended contraction form a tight, repeatable cluster in a
learned embedding; misses are scattered. So

  1. window the filtered EMG and compute per-channel time-domain features,
  2. train an MLP encoder with a supervised contrastive (SupCon) loss on the step labels,
     with an intensity augmentation (the same movement at another strength is a positive),
  3. score every window out-of-fold: does it sit in its own class's cluster (nearest
     prototype margin + kNN label agreement), smoothed over time within the hold,
  4. call low-scoring windows misses, drop them, retrain, and repeat a few rounds.

Evaluation never trains on the hard holds: they're the cleanest labels in a run (max effort,
the coach marks ~all of it clean), so the decoder trains on rest + medium (+ faint/train) and
is tested on hard + held-out rest. Decoders compared: LDA on all labels, LDA on the cleaned
labels, and nearest-prototype in the contrastive embedding.

Usage:  python analysis/contrastive_calib.py rafa1 rafa1.1      (recording names or ids)
        python analysis/contrastive_calib.py julia1
Writes analysis/out/<name>.json (report + exported model).  numpy only.
"""
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT / 'recordings'
OUT = Path(__file__).resolve().parent / 'out'
FS = 250
WIN, HOP = 50, 25          # 200 ms windows, 100 ms hop
CHUNK_S = 4.0              # folds are made of contiguous 4 s chunks
EXCL_S = 4.0               # a window's neighbours must be this far away in time (or in another hold)
NCH = 8
REST = 8                   # class index for rest; movements are 0..7
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- loading + filtering

def find_recording(key):
    for line in (REC / 'index.jsonl').read_text().splitlines():
        e = json.loads(line)
        if key in (e['id'], e.get('name')):
            return REC / e['id']
    raise SystemExit(f'no recording named {key!r} in recordings/index.jsonl')


def biquad_hp(f0, q):
    w = 2 * math.pi * f0 / FS; a = math.sin(w) / (2 * q); c = math.cos(w)
    return np.array([(1 + c) / 2, -(1 + c), (1 + c) / 2]), np.array([1 + a, -2 * c, 1 - a])


def biquad_notch(f0, q):
    w = 2 * math.pi * f0 / FS; a = math.sin(w) / (2 * q); c = math.cos(w)
    return np.array([1, -2 * c, 1.0]), np.array([1 + a, -2 * c, 1 - a])


def filter_chain(mains):
    """The live chain: 20 Hz 4th-order Butterworth HP, notch mains x2 and 2*mains x2 (Q 5)."""
    sos = [biquad_hp(20, 0.54119610), biquad_hp(20, 1.30656296)]
    for f in (mains, mains, 2 * mains, 2 * mains):
        if f < FS / 2:
            sos.append(biquad_notch(f, 5))
    return [(b / a[0], a / a[0]) for b, a in sos]


def sosfilt(x, sos):
    """Causal direct-form-II-transposed biquads, all channels at once. x: (n, ch)."""
    y = x.astype(np.float64).copy()
    for b, a in sos:
        out = np.empty_like(y)
        z1 = np.zeros(y.shape[1]); z2 = np.zeros(y.shape[1])
        b0, b1, b2 = b; a1, a2 = a[1], a[2]
        for t in range(y.shape[0]):
            xt = y[t]
            yt = b0 * xt + z1
            z1 = b1 * xt - a1 * yt + z2
            z2 = b2 * xt - a2 * yt
            out[t] = yt
        y = out
    return y


def load(folder):
    meta = json.loads((folder / 'meta.json').read_text())
    cache = folder / 'filtered.npz'
    rows = list(csv.DictReader(open(folder / 'samples.csv')))
    if cache.exists():
        emg = np.load(cache)['emg']
    else:
        raw = np.array([[float(r[f'ch{c + 1}_uV']) for c in range(NCH)] for r in rows])
        raw -= raw[:FS].mean(0)  # start the HP near zero so it doesn't ring for seconds
        emg = sosfilt(raw, filter_chain(meta.get('mains', 60)))
        np.savez_compressed(cache, emg=emg.astype(np.float32))
    lab = {
        'step': np.array([r['step_name'] for r in rows]),
        'move': np.array([int(r['movement']) if r['movement'] not in ('', None) else -1 for r in rows]),
        'phase': np.array([r['phase_name'] for r in rows]),
        'check': np.array([r['check_name'] for r in rows]),
        'rec': np.array([r['recording'] == '1' for r in rows]),
        'posture': np.array([r.get('posture', '') for r in rows]),
    }
    return meta, emg, lab


# ---------------------------------------------------------------- windows + features

def features(w):
    """w: (WIN, ch) filtered µV -> 40 features: log MAV, log WL, ZC, SSC per channel, + MAV pattern."""
    mav = np.abs(w).mean(0)
    d = np.diff(w, axis=0)
    wl = np.abs(d).mean(0)
    th = 1.0  # µV, ignores quantisation-level wiggles
    zc = ((w[:-1] * w[1:] < 0) & (np.abs(d) > th)).mean(0)
    ssc = ((d[:-1] * d[1:] < 0) & ((np.abs(d[:-1]) > th) | (np.abs(d[1:]) > th))).mean(0)
    lm = np.log(mav + 1e-3)
    return np.concatenate([lm, np.log(wl + 1e-3), zc, ssc, lm - lm.mean()])


AMP = np.r_[np.arange(0, 16)]  # indices of the log-amplitude features (shift together with effort)


def windows(emg, lab, source):
    """Labelled windows lying entirely inside one recorded hold (phase record, not stalled)."""
    n = emg.shape[0]
    key = np.char.add(np.char.add(lab['step'], '|'), lab['move'].astype(str))
    seg_change = np.r_[True, (key[1:] != key[:-1]) | (lab['phase'][1:] != lab['phase'][:-1])
                       | (lab['rec'][1:] != lab['rec'][:-1])]
    seg_id = np.cumsum(seg_change) - 1
    X, meta = [], []
    for s in range(0, n - WIN, HOP):
        e = s + WIN
        if seg_id[s] != seg_id[e - 1] or not lab['rec'][s] or lab['phase'][s] != 'record':
            continue
        step = lab['step'][s]
        y = REST if step == 'rest' else int(lab['move'][s])
        if y < 0:
            continue
        X.append(features(emg[s:e]))
        meta.append((y, step, f'{source}:{seg_id[s]}', s / FS, lab['check'][e - 1], source, lab['posture'][s]))
    return np.array(X), meta


# ---------------------------------------------------------------- SupCon encoder (numpy, manual backprop)

class Encoder:
    def __init__(self, d_in, d_hid=64, d_out=16, seed=0):
        r = np.random.default_rng(seed)
        self.p = {'W1': r.normal(0, 1 / math.sqrt(d_in), (d_in, d_hid)), 'b1': np.zeros(d_hid),
                  'W2': r.normal(0, 1 / math.sqrt(d_hid), (d_hid, d_out)), 'b2': np.zeros(d_out)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, X):
        a1 = np.tanh(X @ self.p['W1'] + self.p['b1'])
        h = a1 @ self.p['W2'] + self.p['b2']
        nrm = np.linalg.norm(h, axis=1, keepdims=True) + 1e-8
        return h / nrm, (X, a1, h, nrm)

    def step(self, X, y, tau=0.1, lr=3e-3, wd=1e-4):
        z, (X, a1, h, nrm) = self.forward(X)
        n = len(y)
        S = z @ z.T / tau
        np.fill_diagonal(S, -np.inf)
        S -= S.max(1, keepdims=True)
        P = np.exp(S); P /= P.sum(1, keepdims=True)
        M = (y[:, None] == y[None, :]).astype(float); np.fill_diagonal(M, 0)
        npos = M.sum(1)
        ok = npos > 0
        logP = np.log(P + 1e-12)
        loss = -((M * logP).sum(1)[ok] / npos[ok]).mean()
        G = np.zeros_like(P)
        G[ok] = (P[ok] - M[ok] / npos[ok, None]) / ok.sum()
        dz = (G + G.T) @ z / tau
        dh = (dz - z * (dz * z).sum(1, keepdims=True)) / nrm
        g = {'W2': a1.T @ dh, 'b2': dh.sum(0)}
        da = dh @ self.p['W2'].T * (1 - a1 ** 2)
        g['W1'] = X.T @ da; g['b1'] = da.sum(0)
        self.t += 1
        for k in self.p:
            gk = g[k] + (wd * self.p[k] if k[0] == 'W' else 0)
            self.m[k] = 0.9 * self.m[k] + 0.1 * gk
            self.v[k] = 0.999 * self.v[k] + 0.001 * gk ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t); vh = self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        return loss


def augment(X, sd_amp=0.5, sd_noise=0.05):
    """Same movement, other effort: shift all log amplitudes together; plus small feature noise."""
    X = X.copy()
    X[:, AMP] += RNG.normal(0, sd_amp, (len(X), 1))
    return X + RNG.normal(0, sd_noise, X.shape)


def train_encoder(Xs, y, w, steps=1200, per_class=40, seed=0):
    """Xs standardised features, y labels, w boolean: which windows are trusted (hits)."""
    enc = Encoder(Xs.shape[1], seed=seed)
    classes = [c for c in np.unique(y[w])]
    idx = {c: np.flatnonzero(w & (y == c)) for c in classes}
    for it in range(steps):
        b = np.concatenate([RNG.choice(idx[c], per_class, replace=len(idx[c]) < per_class) for c in classes])
        enc.step(augment(Xs[b]), y[b], lr=3e-3 if it < steps * 0.7 else 1e-3)
    return enc


# ---------------------------------------------------------------- scoring hits vs misses

def prototypes(z, y, w, classes):
    P = np.zeros((len(classes), z.shape[1]))
    for k, c in enumerate(classes):
        m = w & (y == c)
        if m.any():
            v = z[m].mean(0); P[k] = v / (np.linalg.norm(v) + 1e-8)
    return P


def hit_scores(z_test, y_test, z_tr, y_tr, w_tr, classes, at_test, at_tr, k=15):
    """Hit score per test window: kNN label agreement + own-prototype margin.

    Neighbours from the same hold within EXCL_S seconds don't count: EMG is autocorrelated,
    so a miss would otherwise vouch for itself through its own adjacent windows. A real
    contraction recurs across the hold (and in the other holds); a miss mostly doesn't."""
    P = prototypes(z_tr, y_tr, w_tr, classes)
    cos = z_test @ P.T
    ci = np.searchsorted(classes, y_test)
    own = cos[np.arange(len(y_test)), ci]
    other = cos.copy(); other[np.arange(len(y_test)), ci] = -np.inf
    margin = own - other.max(1)
    ref = np.flatnonzero(w_tr)
    sim = z_test @ z_tr[ref].T
    (seg_te, t_te), (seg_tr, t_tr) = at_test, at_tr
    near = (seg_te[:, None] == seg_tr[ref][None, :]) & (np.abs(t_te[:, None] - t_tr[ref][None, :]) < EXCL_S)
    sim[near] = -np.inf
    nn = np.argpartition(-sim, k, axis=1)[:, :k]
    knn = (y_tr[ref][nn] == y_test[:, None]).mean(1)
    return 0.5 * knn + 0.5 / (1 + np.exp(-margin / 0.05)), margin


def smooth_by_segment(score, seg, half=3):
    out = score.copy()
    for s in np.unique(seg):
        i = np.flatnonzero(seg == s)
        v = score[i]
        k = np.ones(2 * half + 1)
        num = np.convolve(v, k, 'same'); den = np.convolve(np.ones_like(v), k, 'same')
        out[i] = num / den
    return out


def folds_of(meta, k=5):
    """Fold per contiguous CHUNK_S chunk of each hold, round-robin, so neighbours share a fold."""
    f = np.zeros(len(meta), int)
    counter = {}
    for i, (_, _, seg, t, *_) in enumerate(meta):
        key = (seg, int(t // CHUNK_S))
        if key not in counter:
            counter[key] = len(counter)
        f[i] = counter[key] % k
    return f


def clean_labels(X, y, seg, tsec, fold, rounds=3, thresh=0.5, steps=1200, log=print):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    classes = np.unique(y)
    w = np.ones(len(y), bool)
    hist = []
    for r in range(rounds):
        score = np.zeros(len(y)); z_oof = np.zeros((len(y), 16))
        for f in np.unique(fold):
            tr, te = fold != f, fold == f
            enc = train_encoder(Xs[tr], y[tr], w[tr], steps=steps, seed=r * 10 + f)
            z_tr, _ = enc.forward(Xs[tr]); z_te, _ = enc.forward(Xs[te])
            score[te], _ = hit_scores(z_te, y[te], z_tr, y[tr], w[tr], classes, (seg[te], tsec[te]), (seg[tr], tsec[tr]))
            z_oof[te] = z_te
        score = smooth_by_segment(score, seg)
        w_new = score >= thresh
        for c in classes:  # never let a class vanish: keep its best third if it would
            m = y == c
            if w_new[m].sum() < 20:
                w_new[m] = score[m] >= np.quantile(score[m], 2 / 3)
        changed = (w_new != w).mean()
        w = w_new
        hist.append({'round': r, 'kept': float(w.mean()), 'changed': float(changed)})
        log(f'  round {r}: kept {w.mean():.0%} of windows, {changed:.0%} changed')
    enc = train_encoder(Xs, y, w, steps=steps, seed=99)
    return w, score, z_oof, enc, (mu, sd), hist


# ---------------------------------------------------------------- decoders

def lda_fit(X, y, shrink=0.1):
    classes = np.unique(y)
    mus = np.array([X[y == c].mean(0) for c in classes])
    Xc = X - mus[np.searchsorted(classes, y)]
    C = np.cov(Xc.T)
    C = (1 - shrink) * C + shrink * np.trace(C) / len(C) * np.eye(len(C))
    Ci = np.linalg.inv(C)
    W = mus @ Ci; b = -0.5 * (W * mus).sum(1)
    return classes, W, b


def lda_predict(model, X):
    classes, W, b = model
    return classes[np.argmax(X @ W.T + b, 1)]


def balanced_acc(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def per_class(y, p, names):
    return {names[c]: round(float(np.mean(p[y == c] == c)), 3) for c in np.unique(y)}


# ---------------------------------------------------------------- main

def main(keys):
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    X, meta, mov_names = [], [], None
    for key in keys:
        folder = find_recording(key)
        m, emg, lab = load(folder)
        mov_names = [mv['name'] for mv in m['movements']]
        Xi, mi = windows(emg, lab, folder.name)
        print(f'{folder.name}: {len(Xi)} labelled windows ({time.time() - t0:.0f}s)')
        X.append(Xi); meta += mi
    X = np.concatenate(X)
    names = mov_names + ['rest']
    y = np.array([m[0] for m in meta]); step = np.array([m[1] for m in meta])
    seg = np.array([m[2] for m in meta]); tsec = np.array([m[3] for m in meta])
    check = np.array([m[4] for m in meta])
    fold = folds_of(meta)

    # test = every hard hold + one fold of rest; train = everything else
    test = (step == 'hard') | ((step == 'rest') & (fold == 0))
    train = ~test
    counts = {names[c]: {s: int(((y == c) & (step == s)).sum()) for s in ('rest', 'hard', 'medium', 'faint', 'train')
                         if ((y == c) & (step == s)).any()} for c in np.unique(y)}
    print('windows per class/step:', counts)
    missing = [names[c] for c in np.unique(y[test]) if not ((y == c) & train).any()]
    if missing:
        print('no training windows (excluded from test):', missing)
        test &= ~np.isin(y, [names.index(n) for n in missing])

    print('cleaning labels on the training split (hard holds unseen)...')
    w, score, z_oof, enc, (mu, sd), hist = clean_labels(X[train], y[train], seg[train], tsec[train], fold[train])

    Xtr, ytr, Xte, yte = X[train], y[train], X[test], y[test]
    res = {}
    p = lda_predict(lda_fit(Xtr, ytr), Xte)
    res['lda_all_labels'] = (balanced_acc(yte, p), per_class(yte, p, names))
    p = lda_predict(lda_fit(Xtr[w], ytr[w]), Xte)
    res['lda_cleaned'] = (balanced_acc(yte, p), per_class(yte, p, names))
    coach = np.isin(check[train], ['clean'])
    if len(np.unique(ytr[coach])) == len(np.unique(ytr)):
        p = lda_predict(lda_fit(Xtr[coach], ytr[coach]), Xte)
        res['lda_coach_clean_only'] = (balanced_acc(yte, p), per_class(yte, p, names))
    hard = step == 'hard'
    if hard.any():
        first = np.zeros(len(y), bool)
        for s in np.unique(seg[hard]):
            i = np.flatnonzero(seg == s); first[i[: len(i) // 2 - 2]] = True  # gap of 2 windows
        second = np.zeros(len(y), bool)
        for s in np.unique(seg[hard]):
            i = np.flatnonzero(seg == s); second[i[len(i) // 2:]] = True
        cls_h = np.unique(y[hard])
        rest_tr, rest_te = (step == 'rest') & (fold != 0), (step == 'rest') & (fold == 0)
        a = lda_predict(lda_fit(X[first | rest_tr], y[first | rest_tr]), X[second | rest_te])
        res['ceiling_hard_half_vs_half'] = (balanced_acc(y[second | rest_te], a), per_class(y[second | rest_te], a, names))
    z_tr, _ = enc.forward((Xtr - mu) / sd); z_te, _ = enc.forward((Xte - mu) / sd)
    classes = np.unique(ytr)
    P = prototypes(z_tr, ytr, w, classes)
    p = classes[np.argmax(z_te @ P.T, 1)]
    res['contrastive_prototype'] = (balanced_acc(yte, p), per_class(yte, p, names))

    # does the hypothesis hold? misses should be more scattered than hits
    disp = {}
    for c in classes:
        hz, mz = z_tr[(ytr == c) & w], z_tr[(ytr == c) & ~w]
        spread = lambda Z: float(1 - np.linalg.norm(Z.mean(0))) if len(Z) > 1 else None  # 0 = all identical
        # the same question without the encoder (which was trained to pull hits together):
        # in standardised raw features, distance to the hits' median, and frame-to-frame jitter
        R = (Xtr[ytr == c] - mu) / sd; wc = w[ytr == c]; sc = seg[train][ytr == c]
        med = np.median(R[wc], 0) if wc.any() else R.mean(0)
        dist = np.linalg.norm(R - med, axis=1)
        same = sc[1:] == sc[:-1]
        jit = np.linalg.norm(np.diff(R, axis=0), axis=1)
        jh, jm = jit[same & wc[1:] & wc[:-1]], jit[same & ~wc[1:] & ~wc[:-1]]
        disp[names[c]] = {'hits': int(len(hz)), 'misses': int(len(mz)),
                          'hit_spread': spread(hz), 'miss_spread': spread(mz),
                          'raw_dist_hits': float(dist[wc].mean()) if wc.any() else None,
                          'raw_dist_misses': float(dist[~wc].mean()) if (~wc).any() else None,
                          'raw_jitter_hits': float(jh.mean()) if len(jh) else None,
                          'raw_jitter_misses': float(jm.mean()) if len(jm) else None}
    agree = {}
    for c in classes:
        m = (ytr == c) & (step[train] != 'rest')
        if m.any():
            agree[names[c]] = {'kept_when_coach_clean': round(float(w[m & coach].mean()), 3) if (m & coach).any() else None,
                               'kept_when_coach_other': round(float(w[m & (check[train] == 'other')].mean()), 3)
                               if (m & (check[train] == 'other')).any() else None}

    print('\nheld-out hard holds + rest, balanced accuracy:')
    for k, (acc, pc) in res.items():
        print(f'  {k:24s} {acc:.3f}   {pc}')
    print('\nhits vs misses: embedding spread (1-|mean|), raw distance to hit median, raw frame-to-frame jitter')
    f3 = lambda v: '  -  ' if v is None else f'{v:5.2f}'
    for k, v in disp.items():
        print(f'  {k:22s} n {v["hits"]:5d}/{v["misses"]:4d}   emb {f3(v["hit_spread"])} / {f3(v["miss_spread"])}'
              f'   dist {f3(v["raw_dist_hits"])} / {f3(v["raw_dist_misses"])}'
              f'   jitter {f3(v["raw_jitter_hits"])} / {f3(v["raw_jitter_misses"])}')
    print('\nkept as hit, by coach check:', agree)

    # final model on everything (hard included) for export
    print('\nfinal model on all windows...')
    wf, sf, _, encf, (muf, sdf), _ = clean_labels(X, y, seg, tsec, fold)
    Pf = prototypes(encf.forward((X - muf) / sdf)[0], y, wf, np.unique(y))
    name = '+'.join(keys)
    report = {
        'recordings': keys, 'windows': int(len(X)), 'window_ms': WIN * 1000 // FS, 'hop_ms': HOP * 1000 // FS,
        'counts': counts, 'eval': {k: {'balanced_acc': a, 'per_class': pc} for k, (a, pc) in res.items()},
        'spread': disp, 'coach_agreement': agree, 'cleaning_rounds': hist,
        'hit_fraction_final': {names[c]: round(float(wf[y == c].mean()), 3) for c in np.unique(y)},
        'segments': [{'seg': s, 'class': names[int(y[seg == s][0])], 'step': str(step[seg == s][0]),
                      'start_s': round(float(tsec[seg == s][0]), 2),
                      'hit_mask': ''.join('1' if v else '0' for v in wf[seg == s])} for s in dict.fromkeys(seg)],
        'model': {'classes': [names[c] for c in np.unique(y)], 'feature_mean': muf.tolist(), 'feature_std': sdf.tolist(),
                  'W1': encf.p['W1'].tolist(), 'b1': encf.p['b1'].tolist(),
                  'W2': encf.p['W2'].tolist(), 'b2': encf.p['b2'].tolist(), 'prototypes': Pf.tolist(),
                  'filter': '20 Hz 4th-order HP, mains notch x2 + 2x mains x2 (Q5)',
                  'features': 'per channel log MAV, log WL, ZC, SSC (1 uV threshold), log MAV minus channel mean'},
    }
    path = OUT / f'{name}.json'
    path.write_text(json.dumps(report))
    print(f'wrote {path.relative_to(ROOT)} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main(sys.argv[1:] or ['rafa1', 'rafa1.1'])
