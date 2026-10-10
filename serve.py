"""Dev server for www/ that tells the browser never to cache, so a plain reload
always picks up a fresh wasm-pack build. Usage: python serve.py [port]"""
import functools
import http.server
import sys
from pathlib import Path


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
handler = functools.partial(NoCacheHandler, directory=str(Path(__file__).parent / 'www'))
print(f'Serving www/ on http://localhost:{port}')
http.server.ThreadingHTTPServer(('', port), handler).serve_forever()
