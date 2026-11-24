// Synthetic Cyton byte stream, so the whole parse → filter → control path can
// be exercised without hardware. Each channel carries a DC offset, a little
// noise, and an EMG-like noise burst scaled by its contraction level, plus mains
// hum and noise that are identical on every channel, as a shared REF electrode
// picks them up. A contraction also bleeds onto two other electrodes, the
// crosstalk that the pattern decoder has to see through.

/** Contracting channel k also drives (k + offset) % 8 by this much. */
const CROSSTALK = [[2, 0.6], [5, 0.4]];

/**
 * Digit array (bipolar pairs, in channel order: flexors from the thumb side to
 * the little-finger side, extensors in the same order, biceps, triceps): each
 * movement's share of activity on each pair. Each finger leans on its own site
 * but spills onto its neighbours, and flexing braces the extensors a little
 * (and the other way round). Thumb flex / extend, index flex / extend,
 * ring + little flex / extend, bicep, tricep (as in gesture.rs's tests).
 */
export const DIGIT_PATTERNS = [
  [0.85, 0.35, 0.05, 0.15, 0.05, 0.0, 0.0, 0.0],
  [0.1, 0.05, 0.0, 0.9, 0.35, 0.05, 0.0, 0.0],
  [0.35, 0.9, 0.35, 0.05, 0.15, 0.05, 0.0, 0.0],
  [0.05, 0.1, 0.05, 0.35, 0.9, 0.35, 0.0, 0.0],
  [0.05, 0.35, 0.9, 0.0, 0.05, 0.15, 0.0, 0.0],
  [0.0, 0.05, 0.1, 0.05, 0.35, 0.9, 0.0, 0.0],
  [0.05, 0.1, 0.05, 0.05, 0.0, 0.0, 0.9, 0.1],
  [0.0, 0.0, 0.0, 0.05, 0.05, 0.1, 0.15, 0.9],
];

/** Resting tone on each channel in each arm posture: limp, forearm raised, arm held up. */
const POSTURE_TONE = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0.035, 0.014, 0.007, 0.021, 0.028, 0.011, 0.007, 0.014],
  [0.063, 0.035, 0.028, 0.056, 0.049, 0.028, 0.025, 0.042],
];

const RATE = 250;
const UV_PER_COUNT = (4.5 / 24 / 8388607) * 1e6;
const FULL_SCALE = 8388607;

function gauss() {
  const u = 1 - Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

export class CytonSimulator {
  /**
   * differential: electrode pairs, which cancel the hum and noise common to every channel.
   * crosstalk: a channel's contraction bleeds onto two other electrodes (single-ended sites).
   * patterns: per movement, its share on each channel (setPatternLevel drives them).
   */
  constructor({ differential = false, crosstalk = true, patterns = [] } = {}) {
    this.differential = differential;
    this.crosstalk = crosstalk ? CROSSTALK : [];
    this.patterns = patterns;
    this.target = new Float32Array(8);
    this.level = new Float32Array(8);
    this.patternTarget = new Float32Array(patterns.length);
    this.patternLevel = new Float32Array(patterns.length);
    this.posture = 0;
    this.tone = new Float32Array(8);
    this.offset = Array.from({ length: 8 }, () => (Math.random() - 0.5) * 20000);
    this.seq = 0;
    this.sent = 0;
    this.start = null;
  }

  /** Contraction level 0..1 for a channel (0-based). */
  setLevel(ch, value) {
    this.target[ch] = value;
  }

  /** Tension 0..1 (1 = hard) of one of the patterns given to the constructor. */
  setPatternLevel(i, value) {
    this.patternTarget[i] = value;
  }

  /** Arm posture: 0 limp, 1 forearm raised, 2 arm held up (adds resting tone). */
  setPosture(p) {
    this.posture = Math.max(0, Math.min(POSTURE_TONE.length - 1, p | 0));
  }

  /** Returns the packets due since the last call. */
  pump(now) {
    if (this.start === null) this.start = now;
    let due = Math.floor(((now - this.start) / 1000) * RATE) - this.sent;
    if (due > RATE) {
      // tab was in the background; skip ahead instead of flooding
      this.sent += due - RATE;
      due = RATE;
    }
    const out = new Uint8Array(Math.max(due, 0) * 33);
    for (let i = 0; i < due; i++) {
      const t = this.sent / RATE;
      const p = i * 33;
      // mains a little off nominal, as the board's clock sees it
      const common = (250 * Math.sin(2 * Math.PI * 59.9 * t) + 20 * gauss()) * (this.differential ? 0.02 : 1);
      out[p] = 0xa0;
      out[p + 1] = this.seq;
      this.seq = (this.seq + 1) & 0xff;
      const drive = new Float32Array(8);
      for (let k = 0; k < 8; k++) {
        this.level[k] += (this.target[k] - this.level[k]) * 0.04;
        drive[k] += this.level[k];
        for (const [offset, gain] of this.crosstalk) drive[(k + offset) % 8] += gain * this.level[k];
      }
      this.patterns.forEach((pattern, m) => {
        this.patternLevel[m] += (this.patternTarget[m] - this.patternLevel[m]) * 0.04;
        pattern.forEach((share, ch) => { drive[ch] += share * this.patternLevel[m]; });
      });
      // moving the arm into a posture takes a moment too
      POSTURE_TONE[this.posture].forEach((v, ch) => {
        this.tone[ch] += (v - this.tone[ch]) * 0.01;
        drive[ch] += this.tone[ch];
      });
      for (let ch = 0; ch < 8; ch++) {
        const uv = this.offset[ch]
          + common
          + 2 * gauss()
          + 160 * drive[ch] * gauss();
        const counts = Math.max(-FULL_SCALE, Math.min(FULL_SCALE, Math.round(uv / UV_PER_COUNT))) & 0xffffff;
        const q = p + 2 + ch * 3;
        out[q] = (counts >> 16) & 0xff;
        out[q + 1] = (counts >> 8) & 0xff;
        out[q + 2] = counts & 0xff;
      }
      out[p + 32] = 0xc0;
      this.sent++;
    }
    return out;
  }
}
