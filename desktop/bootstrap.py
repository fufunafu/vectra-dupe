"""First-run setup. Installs into a private user environment, never system Python."""

from pathlib import Path
import hashlib
import os
import platform
import subprocess
import sys
import venv


def main():
    if not (3, 11) <= sys.version_info[:2] <= (3, 12):
        raise SystemExit('Install 64-bit Python 3.11 or 3.12 from https://www.python.org/downloads/ and run this launcher again. Include Tcl/Tk in the installation.')
    if sys.maxsize <= 2 ** 32:
        raise SystemExit('faceMap needs a 64-bit Python installation.')
    project = Path(__file__).resolve().parent
    appdata = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local' / 'share')) / 'faceMap'
    environment = appdata / f'desktop-py{sys.version_info.major}.{sys.version_info.minor}-{platform.machine()}'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.is_file():
        print('Preparing a private Python environment for faceMap...', flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    # Reinstall if app code or declared dependencies change, not on every start.
    digest = hashlib.sha256((project / 'pyproject.toml').read_bytes())
    for path in sorted((project / 'facemap_desktop').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            digest.update(path.relative_to(project).as_posix().encode())
            digest.update(path.read_bytes())
    signature = digest.hexdigest()
    marker = environment / 'facemap-installed.txt'
    if not marker.exists() or marker.read_text() != signature:
        print('Installing the CPU engine. This first setup needs internet and can take several minutes.', flush=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '--only-binary=:all:', str(project)], check=True)
        marker.write_text(signature)
    return subprocess.call([str(python), '-m', 'facemap_desktop', *sys.argv[1:]])


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError) as error:
        print(f'Setup could not finish: {error}\nCheck your internet connection and Python version, then run the launcher again.', file=sys.stderr)
        sys.exit(1)
