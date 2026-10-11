// Synthetic Cyton byte stream, so the whole parse → filter → control path can
// be exercised without hardware. Each channel carries a DC offset, a little
// noise, and an EMG-like noise burst scaled by its contraction level, plus mains
// hum and noise that are identical on every channel, as a shared REF electrode
// picks them up. A contraction also bleeds onto two other electrodes, the
// crosstalk that the pattern decoder has to see through.

/** Contracting channel k also drives (k + offset) % 8 by this much. */
const CROSSTALK = [[2, 0.6], [5, 0.4]];

/**
 * Finger grid: share of each finger's activity on each site (upper row from the
 * thumb side to the pinky side, then the middle row), overlapping like real
 * forearm flexors. Thumb, index, middle, ring + little.
 */
const FINGER_PATTERNS = [
  [0.15, 0.05, 0.0, 0.0, 0.9, 0.35, 0.05, 0.0],
  [0.6, 0.85, 0.25, 0.0, 0.3, 0.55, 0.15, 0.0],
  [0.1, 0.45, 0.85, 0.25, 0.0, 0.3, 0.8, 0.2],
  [0.0, 0.1, 0.4, 0.9, 0.0, 0.05, 0.35, 0.85],
];

/** Resting tone on each grid site in each arm posture: limp, forearm raised, arm held up. */
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
  /** differential: electrode pairs, which cancel the hum and noise common to every channel. */
  constructor({ differential = false } = {}) {
    this.differential = differential;
    this.target = new Float32Array(8);
    this.level = new Float32Array(8);
    this.fingerTarget = new Float32Array(FINGER_PATTERNS.length);
    this.fingerLevel = new Float32Array(FINGER_PATTERNS.length);
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

  /** Finger tension 0..1 (1 = a hard flex) on the finger grid. */
  setFingerLevel(finger, value) {
    this.fingerTarget[finger] = value;
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
        for (const [offset, gain] of CROSSTALK) drive[(k + offset) % 8] += gain * this.level[k];
      }
      FINGER_PATTERNS.forEach((pattern, f) => {
        this.fingerLevel[f] += (this.fingerTarget[f] - this.fingerLevel[f]) * 0.04;
        pattern.forEach((share, ch) => { drive[ch] += share * this.fingerLevel[f]; });
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
