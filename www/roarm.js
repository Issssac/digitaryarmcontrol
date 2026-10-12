// Waveshare RoArm-M2-S over USB (Web Serial). The simulated arm uses the real
// arm's own joint chain, so the two share joint angles 1:1: while not following,
// the page shows the real arm's pose; while following, the real arm is sent the
// simulation's.
// Protocol as in roarm-haptic-bridge (robot_dashboard/app.py): newline-ended
// JSON, {"T":105} asks for feedback (answered with T 1051), {"T":101} moves one
// joint. The gripper's wire angle is π − its angle (0 = closed).

const BAUD = 115200;
const encoder = new TextEncoder();
const decoder = new TextDecoder();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const FTDI_VENDOR = 0x0403; // the Cyton dongle

/** Limits as the bridge enforces them (the simulation uses the same). */
export const ROARM_JOINTS = [
  { key: 'base', min: -180, max: 180 },
  { key: 'shoulder', min: -90, max: 90 },
  { key: 'elbow', min: -20, max: 169 },
  { key: 'hand', min: -10, max: 90 },
];
const HAND_ZERO = Math.PI;
/** Servo speed (steps/s, 4096 per turn: 800 ≈ 70°/s) and acceleration for each move. */
const SPEED = 800;
const ACCEL = 10;
/** Targets never move faster than this (°/s), so a Reset in the sim can't fling the arm. */
const MAX_RATE = 60;
const SEND_MS = 50;
const MIN_STEP = 0.5;
const POLL_MS = 100;
/** No feedback for this long stops following. */
const STALE_MS = 1000;
export const SERVOS_DOWN = "The controller answers but can't reach its servos. Is the arm's 12 V supply plugged in and switched on?";

export class RoArm {
  static get supported() {
    return 'serial' in navigator;
  }

  constructor({ onStatus, onClose }) {
    this.onStatus = onStatus;
    this.onClose = onClose;
    this.port = null;
    this.text = '';
    this.closing = false;
    this.writes = Promise.resolve();
    /** Latest real joint angles (°), or null before the first feedback. */
    this.real = null;
    this.feedbackAt = 0;
    /** Last time the controller said it couldn't reach a servo. */
    this.servoFailAt = -Infinity;
    this.following = false;
    this.target = null;
    this.sent = null;
    this.lastSend = 0;
  }

  async connect() {
    this.port = await navigator.serial.requestPort();
    if (this.port.getInfo().usbVendorId === FTDI_VENDOR) {
      this.port = null;
      throw new Error('That is the Cyton dongle. Pick the RoArm\'s port.');
    }
    await this.port.open({ baudRate: BAUD });
    // as the bridge does: keep DTR/RTS low so the controller isn't held in reset
    await this.port.setSignals({ dataTerminalReady: false, requestToSend: false });
    this.writer = this.port.writable.getWriter();
    this.reader = this.port.readable.getReader();
    this.loop = this.readLoop();
    this.poll = setInterval(() => this.write({ T: 105 }), POLL_MS);

    this.onStatus('Waiting for the arm to report its joints…');
    const end = performance.now() + 8000; // opening the port can reboot the controller
    while (!this.live && performance.now() < end) await sleep(50);
    if (!this.live) throw new Error(this.servosDown ? SERVOS_DOWN : 'No feedback from the arm. Is it powered on, and was the right port picked?');
    this.onStatus('Connected. The simulation shows the arm\'s pose; turn on Follow to drive it.');
  }

  write(cmd) {
    const line = encoder.encode(`${JSON.stringify(cmd)}\n`);
    this.writes = this.writes.then(() => this.writer?.write(line)).catch(() => {});
    return this.writes;
  }

  async readLoop() {
    try {
      while (true) {
        const { value, done } = await this.reader.read();
        if (done) break;
        this.text += decoder.decode(value, { stream: true });
        const lines = this.text.split('\n');
        this.text = lines.pop();
        for (const line of lines) this.parse(line);
      }
    } catch (err) {
      if (!this.closing) this.onStatus(`Arm connection lost: ${err.message}`);
    } finally {
      this.reader.releaseLock();
    }
    if (!this.closing) await this.disconnect(true);
  }

  parse(line) {
    // "Servo ID:11 status: failed." The servo bus has no power; the pose reported then is a placeholder.
    if (/^Servo ID:\d+ status: failed/.test(line)) {
      this.servoFailAt = performance.now();
      this.real = null;
      return;
    }
    let msg;
    try { msg = JSON.parse(line); } catch { return; }
    if (msg.T !== 1051 || this.servosDown) return;
    this.real = [msg.b, msg.s, msg.e, HAND_ZERO - msg.t].map((r) => (r * 180) / Math.PI);
    this.feedbackAt = performance.now();
  }

  get servosDown() {
    return performance.now() - this.servoFailAt < STALE_MS;
  }

  /** Feedback is current and real (the simulation may mirror it, and following may start). */
  get live() {
    return Boolean(this.real) && !this.servosDown && performance.now() - this.feedbackAt < STALE_MS;
  }

  /** Start or stop sending the simulation's pose. Starts from where the arm is. */
  follow(on) {
    this.following = Boolean(on) && this.live;
    if (this.following) {
      this.target = [...this.real];
      this.sent = [...this.real];
    }
  }

  /** Call every frame with the simulated joint angles (°). */
  update(simDeg, dt) {
    if (!this.following) return;
    const now = performance.now();
    if (!this.live) {
      this.following = false;
      this.onStatus(this.servosDown ? `Stopped following. ${SERVOS_DOWN}` : 'Stopped following: the arm stopped reporting its joints.');
      return;
    }
    const step = MAX_RATE * Math.min(dt, 0.1);
    ROARM_JOINTS.forEach((jt, j) => {
      const goal = Math.min(jt.max, Math.max(jt.min, simDeg[j]));
      this.target[j] += Math.max(-step, Math.min(step, goal - this.target[j]));
    });
    if (now - this.lastSend < SEND_MS) return;
    this.lastSend = now;
    ROARM_JOINTS.forEach((jt, j) => {
      if (Math.abs(this.target[j] - this.sent[j]) < MIN_STEP) return;
      this.sent[j] = this.target[j];
      const rad = (this.target[j] * Math.PI) / 180;
      this.write({ T: 101, joint: j + 1, rad: jt.key === 'hand' ? HAND_ZERO - rad : rad, spd: SPEED, acc: ACCEL });
    });
  }

  async disconnect(fromReadLoop = false) {
    if (!this.port || this.closing) return;
    this.closing = true;
    this.following = false;
    clearInterval(this.poll);
    if (!fromReadLoop) {
      try { await this.reader.cancel(); } catch { /* already released */ }
      await this.loop?.catch(() => {});
    }
    await this.writes;
    try { this.writer.releaseLock(); } catch { /* already released */ }
    try { await this.port.close(); } catch { /* already closed */ }
    this.port = null;
    this.onClose?.();
  }
}
