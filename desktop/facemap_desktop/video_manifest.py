"""Bounded on-disk assembly of the workspace's original video parts."""

import json
from pathlib import Path
import re
import shutil

FORMAT = 'facemap-video-upload/1'
MAX_BYTES = 2 * 1024 ** 3
PART_BYTES = 16 * 1024 ** 2


def validate(raw, meta=None):
    raw = Path(raw)
    manifest = raw / 'session.json'
    if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > 1024 ** 2:
        raise ValueError('The video manifest is missing or too large.')
    meta = json.loads(manifest.read_text(encoding='utf-8')) if meta is None else meta
    if not isinstance(meta, dict) or meta.get('format') != FORMAT or meta.get('capture_kind') != 'video' or meta.get('state') != 'saved':
        raise ValueError('Unsupported video upload format.')
    video = meta.get('video')
    if not isinstance(video, dict) or not isinstance(video.get('file'), str) or not re.fullmatch(r'capture\.(mov|mp4|m4v)', video['file']):
        raise ValueError('Invalid video filename.')
    size, parts = video.get('bytes'), video.get('parts')
    if type(size) is not int or not 0 < size <= MAX_BYTES:
        raise ValueError('Video exceeds the 2 GB limit or is empty.')
    if not isinstance(parts, list) or len(parts) != (size + PART_BYTES - 1) // PART_BYTES:
        raise ValueError('Video parts are incomplete.')
    expected_names = {'session.json'}
    for index, part in enumerate(parts):
        name = f'video_{index:04d}.bin'
        length = min(PART_BYTES, size - index * PART_BYTES)
        if not isinstance(part, dict) or part.get('name') != name or type(part.get('bytes')) is not int or part['bytes'] != length:
            raise ValueError('Video parts are damaged or out of order.')
        path = raw / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != length:
            raise ValueError('A video part is missing or truncated.')
        expected_names.add(name)
    if {path.name for path in raw.iterdir()} != expected_names:
        raise ValueError('Video upload contains unexpected files.')
    return meta


def assemble(raw, destination):
    meta = validate(raw)
    target = Path(destination) / meta['video']['file']
    if shutil.disk_usage(target.parent).free < meta['video']['bytes'] + 512 * 1024 ** 2:
        raise ValueError('The processing computer needs more free disk space for this video.')
    with target.open('xb') as output:
        for part in meta['video']['parts']:
            with (Path(raw) / part['name']).open('rb') as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
    if target.stat().st_size != meta['video']['bytes']:
        raise ValueError('The assembled video size does not match its upload.')
    return target
