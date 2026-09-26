"""Read-only, loopback-only model viewer. No uploads, analytics, or CDN."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit
import webbrowser


def serve_viewer(result, open_browser=True):
    result = Path(result).resolve()
    report = json.loads((result / 'report.json').read_text(encoding='utf-8'))
    if report.get('status') != 'complete' or not (result / 'model.glb').is_file():
        raise ValueError('Choose a completed faceMap result folder.')
    assets = Path(__file__).with_name('viewer')
    token = secrets.token_urlsafe(24)
    allowed = {str(p.relative_to(assets)).replace('\\', '/'): p for p in assets.rglob('*') if p.is_file()}
    allowed.update({'model.glb': result / 'model.glb', 'surface-mm.ply': result / 'surface-mm.ply', 'report.json': result / 'report.json'})

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get('Host') != f'127.0.0.1:{self.server.server_port}':
                self.send_error(403)
                return
            path = urlsplit(self.path).path
            prefix = f'/{token}/'
            if not path.startswith(prefix):
                self.send_error(404)
                return
            name = path[len(prefix):] or 'index.html'
            file = allowed.get(name)
            if not file or not file.is_file():
                self.send_error(404)
                return
            self.send_response(200)
            mime = 'text/javascript' if file.suffix in ('.js', '.mjs') else ('model/gltf-binary' if file.suffix == '.glb' else mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(file.stat().st_size))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self' blob:; worker-src 'self' blob:; frame-ancestors 'none'")
            if name in ('model.glb', 'surface-mm.ply') and urlsplit(self.path).query == 'download':
                self.send_header('Content-Disposition', f'attachment; filename="{name}"')
            self.end_headers()
            try:
                with file.open('rb') as inp:
                    while data := inp.read(1024 * 1024):
                        self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/{token}/'
    if open_browser:
        webbrowser.open(url)
    return server, url
