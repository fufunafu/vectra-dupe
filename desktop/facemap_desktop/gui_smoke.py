"""Exercise the packaged window and its worker without personal scan data."""

import json
from pathlib import Path
import tempfile
import time
import tkinter as tk
import zipfile
from urllib.request import urlopen

import cv2

from .fixtures import make_plane
from .gui import Desktop
from .viewer import serve_viewer


def run():
    # Real captures use these bundled resources even though demo scans do not.
    for name in ('haarcascade_frontalface_default.xml', 'haarcascade_profileface.xml'):
        if cv2.CascadeClassifier(cv2.data.haarcascades + name).empty():
            raise RuntimeError(f'Missing packaged OpenCV resource: {name}')
    with tempfile.TemporaryDirectory(prefix='faceMap GUI test ') as temp:
        directory = Path(temp)
        capture = make_plane(directory / 'capture')
        scan = directory / 'scan with spaces.zip'
        with zipfile.ZipFile(scan, 'w') as archive:
            for path in capture.iterdir():
                archive.write(path, path.name)
        root = tk.Tk()
        app = Desktop(root)
        app.source.set(str(scan))
        app.destination.set(str(directory / 'results with spaces'))
        app.preset.set('quick')
        try:
            root.update()
            app.start.invoke()
            deadline = time.monotonic() + 90
            while app.process and time.monotonic() < deadline:
                root.update()
                time.sleep(.03)
            if app.process:
                raise RuntimeError('Desktop worker timed out')
            report = json.loads((app.result / 'report.json').read_text())
            if report.get('status') != 'complete' or str(app.view['state']) != 'normal':
                raise RuntimeError(f'Desktop reconstruction failed: {app.status.get()}')
            if not (app.result / 'model.glb').is_file():
                raise RuntimeError('Desktop worker did not export a model')
            server, url = serve_viewer(app.result, open_browser=False)
            try:
                for name in ('', 'viewer.mjs', 'vendor/three.module.js', 'vendor/loaders/GLTFLoader.js', 'model.glb'):
                    with urlopen(url + name, timeout=5) as response:
                        if response.status != 200 or not response.read(16):
                            raise RuntimeError(f'Packaged viewer resource unavailable: {name}')
            finally:
                server.shutdown()
                server.server_close()
            app.start.invoke()
            app.cancel.invoke()
            deadline = time.monotonic() + 15
            while app.process and time.monotonic() < deadline:
                root.update()
                time.sleep(.03)
            if app.process or not app.cancelled or 'Cancelled' not in app.status.get():
                raise RuntimeError('Desktop cancellation did not finish')
            if (app.result / 'model.glb').exists():
                raise RuntimeError('Cancelled job retained a completed model')
            print(json.dumps({'gui': 'passed', 'worker': 'passed', 'cancel': 'passed', 'cascade_data': 'passed', 'viewer_assets': 'passed'}))
        finally:
            if app.process:
                app.process.kill()
                app.process.wait(timeout=10)
                app.process = None
            app.close()
    return 0
