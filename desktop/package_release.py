"""Archive native builds while preserving executable bits and symlinks."""

from pathlib import Path
import platform
import shutil

root = Path(__file__).resolve().parent
bundle = root / 'dist' / 'faceMap-Desktop'
shutil.copyfile(root / 'README.md', bundle / 'README.md')
release = root / 'release'
release.mkdir(exist_ok=True)
system = platform.system()
name = f'faceMap-Desktop-{system}-{platform.machine()}'
kind = 'zip' if system == 'Windows' else 'gztar'
print(shutil.make_archive(str(release / name), kind, root_dir=bundle.parent, base_dir=bundle.name))
