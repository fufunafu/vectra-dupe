"""Bounded, local-only import of the shipping iPhone session format."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile

import numpy as np
from PIL import Image

MAX_TOTAL = 2_000_000_000
MAX_FILE = 100_000_000
MAX_FILES = 1200
FORMAT = "vectra-dupe-session/1"


class CaptureError(ValueError):
    """A capture cannot be safely or meaningfully processed."""


def filename(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value):
        raise CaptureError("A capture filename is invalid. Export the scan again from faceMap.")
    if value.split('.')[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))} or value.endswith('.'):
        raise CaptureError("A capture filename is not supported on Windows.")
    return value


def dimensions(w, h, maximum=24_000_000):
    if type(w) is not int or type(h) is not int or not (0 < w <= 16384 and 0 < h <= 16384) or w * h > maximum:
        raise CaptureError("The scan has unsupported image dimensions.")


def calibration(k, e):
    if not isinstance(k, dict) or any(type(k.get(n)) not in (int, float) or not math.isfinite(k[n]) for n in ("fx", "fy", "cx", "cy")) or min(k['fx'], k['fy']) <= 0:
        raise CaptureError("Camera calibration is missing or invalid.")
    try:
        matrix = np.asarray(e, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise CaptureError("Camera position data is invalid.") from error
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5):
        raise CaptureError("Camera position data is invalid.")
    r = matrix[:3, :3]
    if not np.allclose(r @ r.T, np.eye(3), atol=.002) or abs(np.linalg.det(r) - 1) > .002 or np.max(np.abs(matrix[:3, 3])) > 1e6:
        raise CaptureError("Camera positions are not rigid transforms in millimetres.")
    return matrix


@dataclass
class Session:
    directory: Path
    metadata: dict

    @property
    def poses(self):
        return self.metadata['poses']

    @property
    def photos(self):
        return [p for p in self.poses if p.get('color_file')] + self.metadata.get('color_frames', [])

    def summary(self):
        return {'depth_frames': len(self.poses), 'photos': len(self.photos),
                'demo': bool(self.metadata.get('is_demo')),
                'capture_app_version': self.metadata.get('app_version'),
                'capture_app_build': self.metadata.get('app_build')}


def _manifest(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8_000_000:
        raise CaptureError("A complete session.json is required.")
    def reject(value):
        raise CaptureError("The scan metadata contains a non-finite number.")
    try:
        meta = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CaptureError("The scan metadata is damaged.") from error
    if not isinstance(meta, dict) or meta.get('format') != FORMAT:
        raise CaptureError("Choose a ZIP exported using Sessions > Export scan in faceMap.")
    if meta.get('state') not in (None, 'saved'):
        raise CaptureError("Save this capture in faceMap before exporting it.")
    poses, photos = meta.get('poses'), meta.get('color_frames', [])
    if not isinstance(poses, list) or not isinstance(photos, list) or len(poses) > 100 or len(photos) > 1000 or not poses and not photos:
        raise CaptureError("The scan is empty or has too many frames.")
    if meta.get('capture_kind') not in (None, 'photo_only') or meta.get('capture_kind') == 'photo_only' and poses:
        raise CaptureError("The scan's capture type is inconsistent.")
    return meta


def validate(directory: Path) -> Session:
    meta = _manifest(directory / 'session.json')
    references = ['session.json']
    names = set()
    for is_depth, entries in ((True, meta['poses']), (False, meta.get('color_frames', []))):
        for entry in entries:
            if not isinstance(entry, dict):
                raise CaptureError("A frame has invalid metadata.")
            name = filename(entry.get('name')).casefold()
            if name in names:
                raise CaptureError("The scan contains duplicate frame names.")
            names.add(name)
            if is_depth:
                dimensions(entry.get('width'), entry.get('height'), 4_000_000)
                calibration(entry.get('intrinsics'), entry.get('world_to_camera'))
                unit = entry.get('depth_unit_mm', 1)
                if type(unit) not in (int, float) or not math.isfinite(unit) or not 0 < unit <= 1000:
                    raise CaptureError("The scan has invalid depth units.")
                depth_name = filename(entry.get('depth_file'))
                references.append(depth_name)
                path = directory / depth_name
                if path.is_symlink() or not path.is_file() or path.stat().st_size != entry['width'] * entry['height'] * 4:
                    raise CaptureError("A depth frame is missing or truncated.")
                depth = np.fromfile(path, dtype='<f4')
                if not np.isfinite(depth).all() or np.any(depth < 0):
                    raise CaptureError("A depth frame contains damaged samples.")
            if not is_depth or entry.get('color_file'):
                photo = filename(entry.get('color_file'))
                references.append(photo)
                dimensions(entry.get('rgb_width'), entry.get('rgb_height'))
                calibration(entry.get('rgb_intrinsics'), entry.get('rgb_world_to_camera', entry.get('world_to_camera')))
                path = directory / photo
                if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE:
                    raise CaptureError("A photograph is missing or too large.")
                try:
                    with Image.open(path) as image:
                        if image.size != (entry['rgb_width'], entry['rgb_height']) or image.format not in ('JPEG', 'PNG'):
                            raise CaptureError("A photo's dimensions do not match its calibration.")
                        image.verify()
                except (OSError, SyntaxError, Image.DecompressionBombError) as error:
                    raise CaptureError("A photograph is damaged.") from error
    if len({name.casefold() for name in references}) != len(references):
        raise CaptureError("The scan references duplicate files.")
    if sum((directory / name).stat().st_size for name in references) > MAX_TOTAL:
        raise CaptureError("This scan exceeds the 2 GB import limit.")
    return Session(directory, meta)


def _unpack(source: Path, dest: Path):
    try:
        archive = zipfile.ZipFile(source)
    except (OSError, zipfile.BadZipFile) as error:
        raise CaptureError("This file is not a readable scan ZIP.") from error
    with archive:
        entries = archive.infolist()
        if len(entries) > MAX_FILES * 2 or sum(p.file_size for p in entries) > MAX_TOTAL:
            raise CaptureError("This ZIP is too large to import.")
        usable = []
        names = set()
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in entry.filename or ':' in entry.filename or '\0' in entry.filename:
                raise CaptureError("The ZIP contains an unsafe path.")
            if stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
                raise CaptureError("Encrypted ZIPs and symbolic links are not supported.")
            if entry.file_size > MAX_FILE:
                raise CaptureError("A file in the ZIP is too large.")
            if entry.is_dir() or '__MACOSX' in path.parts or path.name.startswith('._') or path.name == '.DS_Store':
                continue
            key = str(path).casefold()
            if key in names:
                raise CaptureError("The ZIP contains duplicate filenames.")
            names.add(key)
            usable.append((entry, path))
        manifests = [p for _, p in usable if p.name == 'session.json']
        if len(manifests) != 1:
            raise CaptureError("The ZIP must contain exactly one faceMap scan.")
        parent = manifests[0].parent
        total = 0
        for entry, path in usable:
            if path.parent != parent:
                raise CaptureError("The ZIP contains files outside its scan folder.")
            target = dest / filename(path.name)
            with archive.open(entry) as inp, target.open('xb') as out:
                size = 0
                while chunk := inp.read(1024 * 1024):
                    size += len(chunk)
                    total += len(chunk)
                    if size > MAX_FILE or total > MAX_TOTAL:
                        raise CaptureError("The ZIP exceeds the import size limit.")
                    out.write(chunk)
                if size != entry.file_size:
                    raise CaptureError("The ZIP is truncated.")


@contextmanager
def open_session(source, scratch=None):
    """Own a private snapshot so later source changes cannot affect processing."""
    source = Path(source).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix='facemap-import-', dir=scratch) as temp:
        dest = Path(temp)
        if source.is_dir():
            entries = list(source.iterdir())
            if len(entries) > MAX_FILES or any(p.is_symlink() or not p.is_file() for p in entries):
                raise CaptureError("Choose a scan folder containing regular files only.")
            if any(p.stat().st_size > MAX_FILE for p in entries) or sum(p.stat().st_size for p in entries) > MAX_TOTAL:
                raise CaptureError("This scan exceeds the import size limit.")
            for path in entries:
                if path.name != '.DS_Store':
                    shutil.copyfile(path, dest / filename(path.name))
        else:
            _unpack(source, dest)
        yield validate(dest)
