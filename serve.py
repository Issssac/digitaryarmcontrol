"""Dev server for www/ that tells the browser never to cache, so a plain reload
always picks up a fresh wasm-pack build. Usage: python serve.py [port]

GET/PUT /calibration reads and writes calibration.json next to this file: the
page's saved calibration and panel settings.

It also bridges the OpenBCI WiFi Shield, which a page can't reach on its own:
the shield streams by opening a TCP connection *to* the host, and its HTTP API
sends no CORS headers.
  /shield/api/<path>?ip=<shield>   proxies GET/POST/DELETE to http://<shield>/<path>
  /shield/host?ip=<shield>         {"ip", "port"} the shield should stream to
  /shield/stream                   the shield's raw TCP bytes, streamed as they arrive
"""
import functools
import http.server
import json
import queue
import socket
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TCP_PORT = 3000
CALIBRATION = Path(__file__).parent / 'calibration.json'


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
        if self.path.startswith('/shield/'):
            self.shield()
        else:
            super().do_GET()

    def do_POST(self):
        self.shield()

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
