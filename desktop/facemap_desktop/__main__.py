from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description='Reconstruct faceMap iPhone scans locally using the CPU.')
    parser.add_argument('scan', nargs='?', help='Original MOV/MP4/M4V video, scan ZIP or extracted session folder')
    parser.add_argument('--output', type=Path, help='New, unused result folder')
    parser.add_argument('--mode', choices=['auto', 'depth', 'photos'], default='auto')
    parser.add_argument('--preset', choices=['quick', 'balanced', 'detail'], default='balanced')
    parser.add_argument('--inspect', action='store_true', help='Validate and report counts without reconstruction')
    parser.add_argument('--doctor', action='store_true', help='Check runtime and CPU dependencies')
    parser.add_argument('--view', type=Path, help='Open an existing result in the local viewer')
    parser.add_argument('--json-progress', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--gui-self-test', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--self-test', action='store_true', help='Reconstruct a synthetic fixture without personal photos')
    parser.add_argument('--self-test-video', action='store_true', help='Render a generated 4K target and reconstruct it without supplied camera data')
    args = parser.parse_args()
    if args.gui_self_test:
        from .gui_smoke import run
        return run()
    if args.self_test_video:
        if args.output is None:
            parser.error('--self-test-video requires a new --output folder')
        import tempfile
        from .video_fixture import make_video
        from .reconstruct import reconstruct
        with tempfile.TemporaryDirectory(prefix='facemap-video-self-test-') as temp:
            source = make_video(Path(temp) / 'generated-target.mov')
            report = reconstruct(source, args.output, preset='quick')
            report['demo'] = True
            (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({'status': report['status'], 'demo': True, 'output': str(args.output),
                              'triangles': report['triangles'], 'registered_frames': report['camera_recovery']['registered_frames']}))
        return 0
    if args.self_test:
        if args.output is None:
            parser.error('--self-test requires a new --output folder')
        import tempfile
        from .fixtures import make_plane
        from .reconstruct import reconstruct
        with tempfile.TemporaryDirectory(prefix='facemap-self-test-') as temp:
            capture = make_plane(Path(temp) / 'capture')
            report = reconstruct(capture, args.output, preset='quick')
            print(json.dumps({'status': report['status'], 'demo': True, 'output': str(args.output)}))
        return 0
    if args.doctor:
        try:
            from . import reconstruct  # noqa: F401
            deps = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'Pillow', 'open3d', 'trimesh')}
            import cv2
            deps['opencv'] = cv2.__version__
            import av
            import pycolmap
            deps.update(av=av.__version__, pycolmap=pycolmap.__version__)
            try:
                import tkinter
                gui = {'available': True, 'tk': tkinter.TkVersion}
            except ImportError:
                gui = {'available': False, 'help': 'Install Python with Tcl/Tk for the desktop window. CLI reconstruction still works.'}
            print(json.dumps({'ok': True, 'system': platform.system(), 'machine': platform.machine(),
                              'python': platform.python_version(), 'gui': gui,
                              'cpu_threads': min(4, os.cpu_count() or 1), 'dependencies': deps}, indent=2))
        except (ImportError, OSError) as error:
            print(json.dumps({'ok': False, 'error': str(error)}))
            return 1
        return 0
    if args.view:
        from .viewer import serve_viewer
        server, url = serve_viewer(args.view)
        print(f'Local viewer: {url}\nKeep this terminal open. Press Ctrl+C to close.')
        try:
            import threading
            threading.Event().wait()
        except KeyboardInterrupt:
            server.shutdown()
        return 0
    if not args.scan:
        try:
            from .gui import launch
        except ImportError as error:
            if error.name in ('_tkinter', 'tkinter'):
                print('This Python installation lacks Tcl/Tk. Use Python 3.11 or 3.12 from python.org with Tcl/Tk installed, or install python3-tk on Linux. Command-line reconstruction is also available.', file=sys.stderr)
                return 1
            raise
        launch()
        return 0
    try:
        if args.inspect:
            if Path(args.scan).suffix.lower() in ('.mov', '.mp4', '.m4v'):
                from .video_decode import probe
                metadata, _ = probe(Path(args.scan).expanduser().resolve(strict=True))
                print(json.dumps({'capture_kind': 'video', 'metric_scale_available': False, **metadata}, indent=2))
                return 0
            from .session import open_session
            with open_session(args.scan) as session:
                print(json.dumps(session.summary(), indent=2))
            return 0
        if args.output is None:
            parser.error('--output must name a new result folder')
        from .reconstruct import reconstruct
        def progress(percent, message):
            print(json.dumps({'percent': percent, 'message': message}) if args.json_progress else f'{percent:3}%  {message}', flush=True)
        report = reconstruct(args.scan, args.output, args.mode, args.preset, progress)
        print(json.dumps({'complete': True, 'output': str(args.output.absolute()), 'warnings': report['warnings']}), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({'error': str(error) or 'Cancelled'}), flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
