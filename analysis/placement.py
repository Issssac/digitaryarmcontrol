"""Electrode placement audit for digit-array recordings.

Per channel: resting noise, how far each movement's hard hold lifts it above rest,
whether each movement peaks on the channel meant for it, which movements share a
pattern (confusable), which channels say the same thing (redundant), and what each
channel adds to decoding (drop-one and greedy forward selection with an LDA on
per-channel log MAV + log WL).

Decoding is scored two ways, both with whole holds kept apart in time:
  hard½→hard½ : first half of every hard hold (+ rest) trains, second half tests, and back
  medium→hard : the medium holds (+ half the rest) train, the hard holds test
Usage: python analysis/placement.py julia1 ; python analysis/placement.py rafa1 rafa1.1
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contrastive_calib as cc  # noqa: E402

AMPF = np.r_[0:16]  # log MAV (0..7), log WL (8..15)


def chan_feats(chs):
    return np.r_[chs, np.asarray(chs) + 8]


def split_halves(meta):
    """0/1 per window: first or second half (in time) of its hold."""
    seg = np.array([m[2] for m in meta]); t = np.array([m[3] for m in meta])
    half = np.zeros(len(meta), int)
    for s in np.unique(seg):
        i = seg == s
        half[i] = t[i] >= np.median(t[i])
    return half


def score(X, y, tr, te, chs):
    f = chan_feats(chs)
    mu, sd = X[tr][:, f].mean(0), X[tr][:, f].std(0) + 1e-6
    m = cc.lda_fit((X[tr][:, f] - mu) / sd, y[tr], shrink=0.2)
    return cc.balanced_acc(y[te], cc.lda_predict(m, (X[te][:, f] - mu) / sd))


def main(keys):
    X, meta = [], []
    names = None
    checks = {}
    for k in keys:
        folder = cc.find_recording(k)
        md, emg, lab = cc.load(folder)
        names = [m['name'] for m in md['movements']] + ['rest']
        sites = [p['site'] for p in md['placement']]
        Xk, mk = cc.windows(emg, lab, k)
        X.append(Xk); meta += mk
        # coach checks during medium holds, per movement
        for mv in range(8):
            sel = (lab['step'] == 'medium') & (lab['move'] == mv) & (lab['phase'] == 'record')
            if sel.any():
                c = checks.setdefault(names[mv], {})
                for v in lab['check'][sel]:
                    c[v or 'none'] = c.get(v or 'none', 0) + 1
    X = np.vstack(X)
    y = np.array([m[0] for m in meta]); step = np.array([m[1] for m in meta])
    half = split_halves(meta)
    check = np.array([m[4] for m in meta]); posture = np.array([m[6] for m in meta])
    # the rest class is the limp rest the coach kept: not the raised-arm rests
    # (a steady contraction of their own) nor samples it flagged as moving
    rest = (step == 'rest') & (posture == 'limp') & (check == 'clean')
    hard = step == 'hard'; med = step == 'medium'
    mav = np.exp(X[:, 0:8])
    out = {'recordings': keys, 'sites': sites, 'movements': names[:8]}

    rest_mav = np.median(mav[rest], 0)
    out['rest_uV'] = rest_mav.round(2).tolist()
    act = np.array([20 * np.log10(np.median(mav[hard & (y == g)], 0) / rest_mav) for g in range(8)])
    actm = np.array([20 * np.log10(np.median(mav[med & (y == g)], 0) / rest_mav) if (med & (y == g)).any() else np.full(8, np.nan) for g in range(8)])
    out['hard_dB'] = act.round(1).tolist()
    out['medium_dB'] = actm.round(1).tolist()
    out['peak_channel'] = {names[g]: int(np.argmax(act[g])) + 1 for g in range(8)}
    # selectivity: how far the peak stands above the movement's second channel
    srt = np.sort(act, 1)
    out['peak_margin_dB'] = {names[g]: round(float(srt[g, -1] - srt[g, -2]), 1) for g in range(8)}

    # confusable movements: cosine of their (hard) activation patterns, dB above rest
    P = np.clip(act, 0, None)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    C = Pn @ Pn.T
    pairs = sorted(((C[a, b], a, b) for a in range(8) for b in range(a + 1, 8)), reverse=True)
    out['most_similar_movements'] = [(names[a], names[b], round(float(c), 3)) for c, a, b in pairs[:6]]
    # same with medium patterns (what the decoder learns from)
    Pm = np.clip(np.nan_to_num(actm), 0, None)
    Pmn = Pm / (np.linalg.norm(Pm, axis=1, keepdims=True) + 1e-9)
    Cm = Pmn @ Pmn.T
    pairs = sorted(((Cm[a, b], a, b) for a in range(8) for b in range(a + 1, 8)), reverse=True)
    out['most_similar_movements_medium'] = [(names[a], names[b], round(float(c), 3)) for c, a, b in pairs[:6]]
    out['medium_vs_hard_same_movement_cos'] = {names[g]: round(float(Pmn[g] @ Pn[g]), 3) for g in range(8)}

    # redundant channels: correlation of the channels' profiles across movements,
    # and of their log MAV over every hold window
    prof = act.T  # channel x movement
    out['channel_profile_corr'] = np.corrcoef(prof).round(2).tolist()
    holds = hard | med
    out['channel_logmav_corr'] = np.corrcoef(X[holds][:, 0:8].T).round(2).tolist()

    # decoding: the hard holds are the cleanest labels
    hh = hard | rest
    full = list(range(8))
    def hh_score(chs):
        a = score(X, y, hh & (half == 0), hh & (half == 1), chs)
        b = score(X, y, hh & (half == 1), hh & (half == 0), chs)
        return (a + b) / 2
    def mh_score(chs):
        return score(X, y, med | (rest & (half == 0)), hard | (rest & (half == 1)), chs)
    out['acc_all'] = {'hard_half': round(hh_score(full), 3), 'medium_to_hard': round(mh_score(full), 3)}
    out['acc_drop_one'] = {f'CH{c + 1}': {'hard_half': round(hh_score([d for d in full if d != c]), 3),
                                          'medium_to_hard': round(mh_score([d for d in full if d != c]), 3)} for c in full}
    chosen, order = [], []
    while len(chosen) < 8:
        best = max((c for c in full if c not in chosen), key=lambda c: hh_score(chosen + [c]))
        chosen.append(best)
        order.append((f'CH{best + 1}', round(hh_score(chosen), 3)))
    out['greedy_forward_hard_half'] = order
    out['medium_checks'] = checks
    return out


if __name__ == '__main__':
    keys = sys.argv[1:] or ['julia1']
    res = main(keys)
    path = cc.OUT / f"placement_{'+'.join(keys)}.json"
    cc.OUT.mkdir(exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
