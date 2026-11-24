// Records every calibration run to disk through serve.py (recordings/<id>/):
//   meta.json     what was run, from which source, with which settings and prior calibration
//   raw.bin       every byte the board sent during the run, exactly as the parser got it
//   events.jsonl  the coach's status on every change and 10× a second, plus user actions
//   result.json   how it ended, the report, the resulting calibration (base64)
//   samples.csv   raw.bin decoded by the server: one row per Cyton packet, µV per channel
// Uploads are queued in order and flushed every second, so a crash loses at most that.

const FLUSH_MS = 1000;

const newId = () => {
  const d = new Date();
  const p = (n, w = 2) => String(n).padStart(w, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
    + `-${Math.random().toString(36).slice(2, 6)}`;
};

export class CalibrationRecorder {
  /** packets(): the parser's packet count, so events line up with samples.csv rows. */
  constructor({ packets, onError }) {
    this.packets = packets;
    this.onError = onError;
    this.id = null;
    this.chunks = [];
    this.bytes = 0;
    this.lines = [];
    this.queue = Promise.resolve();
    this.timer = null;
    this.failed = false;
    addEventListener('pagehide', () => this.abandon());
  }

  get active() { return this.id !== null; }

  start(meta) {
    if (this.active) this.finish({ outcome: 'superseded' });
    this.id = newId();
    this.t0 = performance.now();
    this.p0 = this.packets();
    this.failed = false;
    this.chunks = [];
    this.lines = [];
    const body = { ...meta, id: this.id, startedAt: new Date().toISOString(), startPackets: this.p0 };
    this.send('start', JSON.stringify(body), 'application/json');
    this.timer = setInterval(() => this.flush(), FLUSH_MS);
    return this.id;
  }

  /** Raw stream bytes, called for every chunk fed to the parser. */
  capture(bytes) {
    if (!this.active || !bytes.length) return;
    this.chunks.push(bytes.slice());
    this.bytes += bytes.length;
  }

  /** One line of events.jsonl; `sample` is the row in samples.csv (±1 at the very start). */
  event(type, data = {}) {
    if (!this.active) return;
    const t = Math.round(performance.now() - this.t0);
    this.lines.push(JSON.stringify({ t, sample: this.packets() - this.p0, type, ...data }));
  }

  flush() {
    if (!this.active) return;
    if (this.chunks.length) {
      const blob = new Blob(this.chunks);
      this.chunks = [];
      this.send('raw', blob, 'application/octet-stream');
    }
    if (this.lines.length) {
      const text = this.lines.join('\n') + '\n';
      this.lines = [];
      this.send('events', text, 'application/x-ndjson');
    }
  }

  /** Ends the run: flushes what's left and writes result.json. Returns when it's on disk. */
  finish(result) {
    if (!this.active) return Promise.resolve();
    this.event('end', { outcome: result.outcome });
    this.flush();
    clearInterval(this.timer);
    const done = this.send('finish', JSON.stringify({ ...result, endedAt: new Date().toISOString(),
      durationMs: Math.round(performance.now() - this.t0), samples: this.packets() - this.p0 }), 'application/json');
    this.id = null;
    return done;
  }

  /** The page is going away mid-run: best-effort beacons, so the run isn't lost. */
  abandon() {
    if (!this.active) return;
    const base = `/recordings/${this.id}`;
    this.event('end', { outcome: 'page closed' });
    if (this.chunks.length) navigator.sendBeacon(`${base}/raw`, new Blob(this.chunks));
    if (this.lines.length) navigator.sendBeacon(`${base}/events`, this.lines.join('\n') + '\n');
    navigator.sendBeacon(`${base}/finish`, JSON.stringify({ outcome: 'page closed', endedAt: new Date().toISOString() }));
    clearInterval(this.timer);
    this.id = null;
  }

  send(what, body, type) {
    const url = `/recordings/${this.id}/${what}`;
    this.queue = this.queue.then(async () => {
      try {
        const res = await fetch(url, { method: 'POST', body, headers: { 'Content-Type': type } });
        if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
      } catch (err) {
        if (!this.failed) this.onError?.(`Calibration recording ${what} failed: ${err.message ?? err}`);
        this.failed = true;
      }
    });
    return this.queue;
  }
}
