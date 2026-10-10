// Synthetic Cyton byte stream, so the whole parse → filter → control path can
// be exercised without hardware. Each channel carries a DC offset, a little
// noise, and an EMG-like noise burst scaled by its contraction level, plus mains
// hum and noise that are identical on every channel, as a shared REF electrode
// picks them up. A contraction also bleeds onto two other electrodes, the
// crosstalk that the pattern decoder has to see through.

/** Contracting channel k also drives (k + offset) % 8 by this much. */
const CROSSTALK = [[2, 0.6], [5, 0.4]];

const RATE = 250;
const UV_PER_COUNT = (4.5 / 24 / 8388607) * 1e6;
const FULL_SCALE = 8388607;

function gauss() {
  const u = 1 - Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

export class CytonSimulator {
  constructor() {
    this.target = new Float32Array(8);
    this.level = new Float32Array(8);
    this.offset = Array.from({ length: 8 }, () => (Math.random() - 0.5) * 20000);
    this.seq = 0;
    this.sent = 0;
    this.start = null;
  }

  /** Contraction level 0..1 for a channel (0-based). */
  setLevel(ch, value) {
    this.target[ch] = value;
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
      const common = 250 * Math.sin(2 * Math.PI * 59.9 * t) + 20 * gauss();
      out[p] = 0xa0;
      out[p + 1] = this.seq;
      this.seq = (this.seq + 1) & 0xff;
      const drive = new Float32Array(8);
      for (let k = 0; k < 8; k++) {
        this.level[k] += (this.target[k] - this.level[k]) * 0.04;
        drive[k] += this.level[k];
        for (const [offset, gain] of CROSSTALK) drive[(k + offset) % 8] += gain * this.level[k];
      }
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
