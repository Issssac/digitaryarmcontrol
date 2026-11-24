"""Dev server for www/ that tells the browser never to cache, so a plain reload
always picks up a fresh wasm-pack build. Usage: python serve.py [port]

GET/PUT /calibration reads and writes calibration.json next to this file: the
page's saved calibration and panel settings.

Every calibration run the page starts is recorded under recordings/<id>/
(see www/recorder.js):
  POST /recordings/<id>/start    meta.json (what was run, settings, prior calibration)
  POST /recordings/<id>/raw      appends the board's raw bytes to raw.bin
  POST /recordings/<id>/events   appends JSON lines to events.jsonl
  POST /recordings/<id>/finish   result.json, then raw.bin decoded to samples.csv,
                                 each row labelled from events.jsonl (step, movement, phase…)
  GET  /recordings               index of all runs (recordings/index.jsonl)
  GET  /recordings/<id>/<file>   any of the files above

It also bridges the OpenBCI WiFi Shield, which a page can't reach on its own:
the shield streams by opening a TCP connection *to* the host, and its HTTP API
sends no CORS headers.
  /shield/api/<path>?ip=<shield>   proxies GET/POST/DELETE to http://<shield>/<path>
  /shield/host?ip=<shield>         {"ip", "port"} the shield should stream to
  /shield/stream                   the shield's raw TCP bytes, streamed as they arrive
"""
import functools
import os
import http.server
import json
import queue
import re
import socket
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# the shield streams to this port; SHIELD_TCP_PORT moves it, e.g. for a second server
TCP_PORT = int(os.environ.get('SHIELD_TCP_PORT', 3000))
CALIBRATION = Path(__file__).parent / 'calibration.json'
RECORDINGS = Path(__file__).parent / 'recordings'
RECORDING_ID = re.compile(r'^[0-9A-Za-z_-]{1,96}$')
RECORDING_FILES = {'meta.json', 'raw.bin', 'events.jsonl', 'result.json', 'samples.csv'}
recordings_lock = threading.Lock()

# Cyton packets, as src/emg/cyton.rs parses them
PACKET_LEN = 33
UV_PER_COUNT = 4.5 / 24 / 8_388_607 * 1e6


def decode_cyton(raw):
    """Yields (seq, [8 x µV], aux bytes, footer) for each valid packet in a raw byte stream."""
    i, n = 0, len(raw)
    while i + PACKET_LEN <= n:
        if raw[i] != 0xA0:
            i += 1
            continue
        pkt = raw[i:i + PACKET_LEN]
        if not 0xC0 <= pkt[32] <= 0xC6:
            i += 1  # resync on the next header
            continue
        chans = [int.from_bytes(pkt[2 + 3 * c:5 + 3 * c], 'big', signed=True) * UV_PER_COUNT for c in range(8)]
        yield pkt[1], chans, pkt[26:32], pkt[32]
        i += PACKET_LEN


# the coach's status vector (main.js STATUS_FIELDS) and its codes, for older runs whose meta lacks them
STATUS_FIELDS = ['step', 'index', 'phase', 'progress', 'secondsLeft', 'check', 'arg', 'stuck', 'meterKind',
                 'meter', 'lo', 'hi', 'segment', 'segments', 'round', 'rounds', 'goal', 'aim', 'posture', 'waiting']
STEP_NAMES = ['rest', 'hard', 'medium', 'faint', 'train']
PHASE_NAMES = ['prepare', 'record', 'relax']
CHECK_NAMES = {-1: '', 0: 'clean', 1: 'no_contact', 2: 'moving', 3: 'weak', 4: 'strong', 5: 'other',
               6: 'not_relaxed', 7: 'fainter'}
LABEL_COLUMNS = ['step', 'step_name', 'movement', 'movement_name', 'phase', 'phase_name', 'check', 'check_name',
                 'posture', 'round', 'recording', 'stalled']


def status_labels(folder, meta):
    """[(sample, label values)] from events.jsonl's status events, in sample order."""
    path = folder / 'events.jsonl'
    if not path.exists():
        return []
    fields = meta.get('statusFields') or STATUS_FIELDS
    steps = meta.get('stepNames') or STEP_NAMES
    phases = meta.get('phaseNames') or PHASE_NAMES
    movements = [m.get('name', '') for m in meta.get('movements') or []]
    postures = meta.get('postures') or []
    at = {name: k for k, name in enumerate(fields)}
    out = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get('type') != 'status' or not isinstance(ev.get('status'), list):
            if ev.get('type') == 'end':
                out.append((ev.get('sample', 0), ['', 'ended', '', '', '', '', '', '', '', '', 0, 0]))
            continue
        st = ev['status']
        get = lambda name, default=-1: st[at[name]] if name in at and at[name] < len(st) and st[at[name]] is not None else default
        step, index, phase, check = int(get('step')), int(get('index')), int(get('phase')), int(get('check'))
        posture, rnd = int(get('posture')), int(get('round'))
        name = lambda names, k: names[k] if 0 <= k < len(names) else ''
        # recorded: the decoder is taking this window (rest, phase-1 rounds and training trials)
        recording = 1 if phase == 1 and not ev.get('stalled') else 0
        out.append((ev.get('sample', 0), [
            step, name(steps, step), index, 'rest' if step == 0 else name(movements, index),
            phase, name(phases, phase), check, CHECK_NAMES.get(check, ''),
            name(postures, posture) if step == 0 else '', rnd, recording, int(bool(ev.get('stalled')))]))
    out.sort(key=lambda e: e[0])
    return out


def write_samples_csv(folder, meta=None):
    """raw.bin → samples.csv: one row per packet, µV per channel, and the coach's labels at that sample."""
    raw = (folder / 'raw.bin').read_bytes() if (folder / 'raw.bin').exists() else b''
    labels = status_labels(folder, meta or {})
    blank = [''] * len(LABEL_COLUMNS)
    rows = 0
    dropped = 0
    prev = None
    k = -1  # the latest label at or before this row
    with open(folder / 'samples.csv', 'w', newline='') as f:
        f.write('sample,t_s,seq,dropped_before,' + ','.join(f'ch{c + 1}_uV' for c in range(8))
                + ',aux_hex,footer,' + ','.join(LABEL_COLUMNS) + '\n')
        for seq, chans, aux, footer in decode_cyton(raw):
            gap = 0 if prev is None else (seq - prev - 1) % 256
            dropped += gap
            prev = seq
            while k + 1 < len(labels) and labels[k + 1][0] <= rows:
                k += 1
            label = labels[k][1] if k >= 0 else blank
            f.write(f'{rows},{rows / 250:.3f},{seq},{gap},' + ','.join(f'{v:.3f}' for v in chans)
                    + f',{aux.hex()},{footer:#04x},' + ','.join(str(v) for v in label) + '\n')
            rows += 1
    return {'bytes': len(raw), 'packets': rows, 'dropped': dropped, 'labelled': bool(labels)}


class Relay:
    """Accepts the shield's TCP connection and fans its bytes out to page streams."""

    def __init__(self, port):
        self.port = port
        self.lock = threading.Lock()
        self.listeners = set()
        server = socket.create_server(('', port))
        threading.Thread(target=self.accept_loop, args=(server,), daemon=True).start()

    def accept_loop(self, server):
        while True:
            conn, addr = server.accept()
            print(f'Shield connected from {addr[0]}')
            threading.Thread(target=self.read_loop, args=(conn,), daemon=True).start()

    def read_loop(self, conn):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with conn:
            while chunk := conn.recv(4096):
                with self.lock:
                    for q in self.listeners:
                        q.put(chunk)
        print('Shield disconnected')

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            self.listeners.add(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.listeners.discard(q)


def host_ip_toward(ip):
    """The local address this machine uses to reach `ip` (no packet is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((ip.split(':')[0], 80))
        return s.getsockname()[0]


class Handler(http.server.SimpleHTTPRequestHandler):
    relay = None

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, fmt, *args):
        if not self.path.startswith('/shield/'):
            super().log_message(fmt, *args)

    def do_GET(self):
        if self.path == '/calibration':
            if not CALIBRATION.exists():
                return self.reply(404, b'no saved calibration')
            return self.reply(200, CALIBRATION.read_bytes())
        if self.path == '/recordings' or self.path.startswith('/recordings/'):
            return self.recording_get()
        if self.path.startswith('/shield/'):
            self.shield()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith('/recordings/'):
            return self.recording_post()
        self.shield()

    def recording_post(self):
        parts = self.path.split('?')[0].strip('/').split('/')
        if len(parts) != 3 or not RECORDING_ID.match(parts[1]):
            return self.reply(404, b'not found')
        _, rid, what = parts
        folder = RECORDINGS / rid
        body = self.rfile.read(int(self.headers.get('Content-Length') or 0))
        with recordings_lock:
            if what == 'start':
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'meta.json').write_bytes(body)
                meta = json.loads(body or b'{}')
                print(f'Calibration recording {rid} started: "{meta.get("name")}" {meta.get("scheme") or meta.get("kind")} '
                      f'{meta.get("mode")} from {meta.get("source")}')
                return self.reply(204, b'')
            if not folder.is_dir():
                return self.reply(404, b'no such recording')
            if what == 'raw':
                with open(folder / 'raw.bin', 'ab') as f:
                    f.write(body)
            elif what == 'events':
                text = body if body.endswith(b'\n') else body + b'\n'
                with open(folder / 'events.jsonl', 'ab') as f:
                    f.write(text)
            elif what == 'finish':
                result = json.loads(body or b'{}')
                meta = json.loads((folder / 'meta.json').read_bytes() or b'{}') if (folder / 'meta.json').exists() else {}
                result['decoded'] = write_samples_csv(folder, meta)
                (folder / 'result.json').write_text(json.dumps(result, indent=1))
                entry = {'id': rid, 'name': meta.get('name'), 'scheme': meta.get('scheme') or meta.get('kind'),
                         'kind': meta.get('kind'), 'mode': meta.get('mode'), 'source': meta.get('source'),
                         'wiring': meta.get('wiring'),
                         'startedAt': meta.get('startedAt'), 'endedAt': result.get('endedAt'),
                         'outcome': result.get('outcome'), 'durationMs': result.get('durationMs'), **result['decoded']}
                with open(RECORDINGS / 'index.jsonl', 'a') as f:
                    f.write(json.dumps(entry) + '\n')
                print(f'Calibration recording {rid} finished ({result.get("outcome")}): '
                      f'{result["decoded"]["packets"]} packets, {result["decoded"]["dropped"]} dropped -> {folder}')
            else:
                return self.reply(404, b'not found')
        self.reply(204, b'')

    def recording_get(self):
        parts = self.path.split('?')[0].strip('/').split('/')
        if len(parts) == 1:
            index = RECORDINGS / 'index.jsonl'
            lines = index.read_text().splitlines() if index.exists() else []
            return self.reply(200, json.dumps([json.loads(l) for l in lines if l.strip()]).encode())
        if len(parts) == 3 and RECORDING_ID.match(parts[1]) and parts[2] in RECORDING_FILES:
            path = RECORDINGS / parts[1] / parts[2]
            if path.exists():
                return self.reply(200, path.read_bytes())
        self.reply(404, b'not found')

    def do_PUT(self):
        if self.path != '/calibration':
            return self.reply(404, b'not found')
        body = self.rfile.read(int(self.headers.get('Content-Length') or 0))
        try:
            json.loads(body)
        except ValueError:
            return self.reply(400, b'not JSON')
        tmp = CALIBRATION.with_suffix('.tmp')
        tmp.write_bytes(body)
        tmp.replace(CALIBRATION)  # never leave a half-written file
        self.reply(204, b'')

    def do_DELETE(self):
        self.shield()

    def shield(self):
        url = urllib.parse.urlsplit(self.path)
        params = urllib.parse.parse_qs(url.query)
        ip = params.get('ip', [''])[0]
        try:
            if url.path == '/shield/stream':
                return self.stream()
            if url.path == '/shield/host':
                return self.reply(200, json.dumps({'ip': host_ip_toward(ip), 'port': self.relay.port}).encode())
            if url.path.startswith('/shield/api/') and ip:
                return self.proxy(ip, url.path[len('/shield/api'):])
            self.reply(404, b'unknown shield endpoint')
        except OSError as err:
            self.reply(502, f'{err}'.encode())

    def proxy(self, ip, path):
        body = self.rfile.read(int(self.headers.get('Content-Length') or 0)) or None
        req = urllib.request.Request(f'http://{ip}{path}', data=body, method=self.command)
        if body:
            req.add_header('Content-Type', 'application/json')
        try:
            with urllib.request.urlopen(req, timeout=5) as res:
                self.reply(res.status, res.read())
        except urllib.error.HTTPError as err:
            self.reply(err.code, err.read())

    def reply(self, status, body):
        self.send_response(status)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream(self):
        q = self.relay.subscribe()
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.flush()
            while True:
                try:
                    chunk = q.get(timeout=1)
                except queue.Empty:
                    chunk = b''  # nothing to send, but notice a closed page below
                if chunk:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                elif self.page_gone():
                    break
        except OSError:
            pass  # page closed the stream
        finally:
            self.relay.unsubscribe(q)

    def page_gone(self):
        self.connection.setblocking(False)
        try:
            return self.connection.recv(1, socket.MSG_PEEK) == b''
        except BlockingIOError:
            return False
        except OSError:
            return True
        finally:
            self.connection.setblocking(True)


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
Handler.relay = Relay(TCP_PORT)
handler = functools.partial(Handler, directory=str(Path(__file__).parent / 'www'))
print(f'Serving www/ on http://localhost:{port} · WiFi Shield relay on TCP :{TCP_PORT}')
http.server.ThreadingHTTPServer(('', port), handler).serve_forever()
