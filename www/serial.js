// Web Serial driver for the OpenBCI Cyton (through its RFduino USB dongle).
// Requires Chrome or Edge, served from localhost or https.

const BAUD = 115200;
const encoder = new TextEncoder();
const decoder = new TextDecoder();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// x <ch> <power 0=on> <gain 6=24x> <input 0=normal> <bias 1=include> <SRB2> <SRB1 0=off> X
// The bottom header row is each channel's N input, the top row its P input.
//   common:  one electrode per muscle on the bottom row, shared REF on the bottom
//            SRB pin (SRB2 on, so every P input sees REF).
//   bipolar: a top+bottom pair on each muscle, no SRB (SRB2 off).
const SRB2 = { common: 1, bipolar: 0 };
const channelConfig = (wiring) =>
  Array.from({ length: 8 }, (_, i) => `x${i + 1}0601${SRB2[wiring]}0X`);
// Dongle-level "is the radio link up?" query (OpenBCI_Radios v2 host command).
const SYS_STATUS = new Uint8Array([0xf0, 0x07]);

export class CytonSerial {
  static get supported() {
    return 'serial' in navigator;
  }

  constructor({ onData, onStatus, onClose, wiring = 'common' }) {
    if (!(wiring in SRB2)) throw new Error(`Unknown wiring mode: ${wiring}`);
    this.wiring = wiring;
    this.onData = onData;
    this.onStatus = onStatus;
    this.onClose = onClose;
    this.port = null;
    this.binary = false;
    this.text = '';
    this.closing = false;
  }

  async connect() {
    this.port = await navigator.serial.requestPort();
    await this.port.open({ baudRate: BAUD, bufferSize: 16384 });
    this.writer = this.port.writable.getWriter();
    this.reader = this.port.readable.getReader();
    this.loop = this.readLoop();

    this.onStatus('Resetting board…');
    await this.send('s'); // in case it was left streaming
    await sleep(300);
    this.text = '';
    await this.send('v');
    const banner = await this.waitFor('$$$', 4000);
    if (/failure|failed to poll/i.test(banner)) {
      throw new Error('The dongle answered but the board did not. Is the Cyton switched on (PC position) and the dongle switch on GPIO6?');
    }
    if (!banner.trim()) {
      throw new Error(await this.diagnoseSilence());
    }

    this.onStatus('Configuring channels for EMG…');
    for (const cmd of channelConfig(this.wiring)) {
      this.text = '';
      await this.send(cmd);
      await this.waitFor('$$$', 1000);
    }

    this.binary = true;
    await this.send('b');
    this.onStatus(`Streaming · ${banner.split('\n').find((l) => /openbci|firmware/i.test(l))?.trim() ?? 'Cyton'}`);
  }

  /** The board ignored `v`: ask the dongle itself which link is broken. */
  async diagnoseSilence() {
    this.text = '';
    await this.writer.write(SYS_STATUS);
    const status = await this.waitFor('$$$', 1500);
    if (/system is up/i.test(status)) {
      return 'Dongle and board radio are linked, but the Cyton\'s main processor is not answering. Power-cycle the board (OFF → PC) without touching PROG, and check the battery.';
    }
    if (/down|failure/i.test(status)) {
      return 'The dongle works but cannot reach the board. Is the Cyton switched to PC, and on the same radio channel?';
    }
    return 'No response from the dongle. Check its switch is on GPIO6, replug it, and make sure the right COM port was picked.';
  }

  async send(cmd) {
    await this.writer.write(encoder.encode(cmd));
  }

  /** Resolves with the text received so far once `token` appears (or on timeout). */
  async waitFor(token, timeoutMs) {
    const end = performance.now() + timeoutMs;
    while (!this.text.includes(token) && performance.now() < end) {
      await sleep(20);
    }
    return this.text;
  }

  async readLoop() {
    try {
      while (true) {
        const { value, done } = await this.reader.read();
        if (done) break;
        if (!value) continue;
        if (this.binary) this.onData(value);
        else this.text += decoder.decode(value, { stream: true });
      }
    } catch (err) {
      if (!this.closing) this.onStatus(`Connection lost: ${err.message}`);
    } finally {
      this.reader.releaseLock();
    }
    if (!this.closing) await this.disconnect(true);
  }

  async disconnect(fromReadLoop = false) {
    if (!this.port || this.closing) return;
    this.closing = true;
    try { await this.send('s'); } catch { /* port may already be gone */ }
    if (!fromReadLoop) {
      try { await this.reader.cancel(); } catch { /* already released */ }
      await this.loop?.catch(() => {});
    }
    try { this.writer.releaseLock(); } catch { /* already released */ }
    try { await this.port.close(); } catch { /* already closed */ }
    this.port = null;
    this.onClose?.();
  }
}
