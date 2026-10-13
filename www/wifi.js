// OpenBCI WiFi Shield driver for the Cyton. A page can't take the shield's TCP
// stream or call its HTTP API (no CORS), so serve.py bridges both: /shield/api
// proxies the shield's API and /shield/stream relays its raw TCP bytes.
// Same interface as CytonSerial.

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Same channel settings as serial.js: gain 24, bias on, SRB2 per wiring.
const SRB2 = { common: 1, bipolar: 0 };
const channelConfig = (wiring) =>
  Array.from({ length: 8 }, (_, i) => `x${i + 1}0601${SRB2[wiring]}0X`);
// The shield switches the Cyton to 1000 Hz; the pipeline runs at 250 Hz.
const RATE_250 = '~6';
// The shield buffers this long (µs) before each TCP write.
const LATENCY_US = 10000;

export class CytonWifi {
  constructor({ ip, onData, onStatus, onClose, wiring = 'common' }) {
    if (!(wiring in SRB2)) throw new Error(`Unknown wiring mode: ${wiring}`);
    this.ip = ip.trim();
    this.wiring = wiring;
    this.onData = onData;
    this.onStatus = onStatus;
    this.onClose = onClose;
    this.binary = false;
    this.closing = false;
    this.abort = new AbortController();
    this.bytes = 0;
  }

  async api(method, path, body) {
    let res;
    try {
      res = await fetch(`/shield/api${path}?ip=${encodeURIComponent(this.ip)}`, {
        method,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: this.abort.signal,
      });
    } catch (err) {
      if (err.name === 'AbortError') throw err;
      throw new Error('serve.py is not reachable. The WiFi Shield needs the page served by `python serve.py`.');
    }
    const text = await res.text();
    if (res.status === 502) {
      throw new Error(`No answer from the shield at ${this.ip} (${text}). Is this computer on the shield's network (OpenBCI-xxxx, shield at 192.168.4.1) or the same LAN?`);
    }
    if (!res.ok) throw new Error(`Shield ${method} ${path} failed: ${res.status} ${text}`);
    return text;
  }

  command(cmd) {
    return this.api('POST', '/command', { command: cmd });
  }

  async connect() {
    this.onStatus(`Contacting the shield at ${this.ip}…`);
    const board = JSON.parse(await this.api('GET', '/board'));
    if (!board.board_connected) {
      throw new Error('The shield is up but sees no board. Is it seated on the Cyton and the Cyton switched to PC?');
    }
    if (board.num_channels && board.num_channels !== 8) {
      throw new Error(`The shield reports a ${board.num_channels}-channel board (${board.board_type}); only the 8-channel Cyton is supported.`);
    }

    this.onStatus('Resetting board…');
    await this.api('GET', '/stream/stop').catch(() => {}); // in case it was left streaming
    const banner = await this.command('v');

    this.onStatus('Configuring channels for EMG…');
    await this.command(RATE_250);
    for (const cmd of channelConfig(this.wiring)) await this.command(cmd);

    this.onStatus('Opening the data link…');
    // Not awaited: the browser may hold the response until the first bytes,
    // and those only flow once the shield is pointed at us below.
    this.loop = this.readLoop();
    const host = JSON.parse(await (await fetch(`/shield/host?ip=${encodeURIComponent(this.ip)}`)).text());
    await this.api('POST', '/tcp', { ip: host.ip, port: host.port, output: 'raw', delimiter: true, latency: LATENCY_US });

    this.binary = true;
    await this.api('GET', '/stream/start');
    await sleep(1500);
    if (!this.bytes) {
      throw new Error(`The shield started streaming but nothing reached ${host.ip}:${host.port}. Allow Python through Windows Firewall (private networks) for TCP ${host.port}.`);
    }
    const name = banner.split('\n').find((l) => /openbci|firmware/i.test(l))?.trim() ?? 'Cyton';
    this.onStatus(`Streaming over WiFi · ${name}`);
  }

  async readLoop() {
    try {
      const res = await fetch('/shield/stream', { signal: this.abort.signal });
      const reader = res.body.getReader();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        this.bytes += value.length;
        if (this.binary) this.onData(value);
      }
    } catch (err) {
      if (!this.closing) this.onStatus(`Connection lost: ${err.message}`);
    }
    if (!this.closing) await this.disconnect(true);
  }

  async disconnect(fromReadLoop = false) {
    if (this.closing) return;
    this.closing = true;
    this.binary = false;
    const stop = new AbortController();
    setTimeout(() => stop.abort(), 2000);
    const call = (method, path) => fetch(`/shield/api${path}?ip=${encodeURIComponent(this.ip)}`, { method, signal: stop.signal }).catch(() => {});
    await call('GET', '/stream/stop');
    await call('DELETE', '/tcp');
    this.abort.abort();
    if (!fromReadLoop) await this.loop?.catch(() => {});
    this.onClose?.();
  }
}
