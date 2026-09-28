"""Same-frame colour/depth validation and measured-depth gates for stereo."""
import json
import math
import numpy as np

FORMAT = 'facemap-rgbd-session/1'


def is_rgbd(metadata):
    return metadata.get('format') == FORMAT


def assess(metadata):
    from .readiness import _usable, _view, _number
    poses = metadata.get('poses', [])
    views = [v for p in poses if (v := _view(p)) is not None]
    distinct = []
    for yaw, pitch, *_ in views:
        if all((yaw-a)**2 + (pitch-b)**2 >= 4 for a, b in distinct):
            distinct.append((yaw, pitch))
    missing = [name for name, found in (
        ('front', any(abs(v[0]) <= 15 and abs(v[1]) <= 15 for v in views)),
        ('left', any(v[0] <= -35 for v in views)), ('right', any(v[0] >= 35 for v in views)),
        ('above', any(v[1] >= 12 for v in views)), ('below', any(v[1] <= -12 for v in views))) if not found]
    issues = []
    if len(distinct) < 20:
        issues.append(f'Record at least 20 distinct sharp synchronized depth views; {len(distinct)} qualify.')
    if missing:
        issues.append('Record the missing views: ' + ', '.join(missing) + '.')
    duration = (metadata.get('rgbd') or {}).get('duration_seconds', 0)
    if not _number(duration) or duration < 8:
        issues.append('Record a slower pass lasting at least 8 seconds.')
    if any(p.get('tracking_state') != 'normal' or not _usable(p.get('photo_quality'))
           or not p.get('confidence_file') or not _number((p.get('quality') or {}).get('valid_depth_fraction'))
           or not .1 <= p['quality']['valid_depth_fraction'] <= 1 for p in poses):
        issues.append('Some frames lack reliable tracking, facial sharpness, or LiDAR confidence.')
    times = [p.get('frame_timestamp_seconds') for p in poses]
    if (any(not _number(t) for t in times) or not times
            or any(b <= a or b-a > 2.5 for a, b in zip(sorted(times), sorted(times)[1:]))
            or (_number(duration) and (min(times) > 2.5 or duration-max(times) > 2.5))):
        issues.append('Too much of the pass lacked usable synchronized frames. Move slowly in brighter light.')
    if any(isinstance(w, str) and w.startswith('Capture interrupted:') for w in metadata.get('warnings', []) or []):
        issues.append('Tracking or subject movement interrupted this capture. Record a new continuous pass.')
    return {'status': 'blocked' if issues else 'passed', 'issues': issues, 'policy_version': 1,
            'usable_distinct_photos': len(distinct), 'missing_regions': missing, 'capture_kind': 'video_depth'}


def validate_manifest(meta):
    from .session import CaptureError, dimensions
    r = meta.get('rgbd')
    if meta.get('capture_kind') != 'video_depth' or meta.get('color_frames') or not isinstance(r, dict):
        raise CaptureError('The synchronized capture type is inconsistent.')
    if r.get('depth_source') != 'arkit-sceneDepth' or r.get('depth_association') != 'same-arframe' or r.get('timeline_file') != 'camera-frames.jsonl':
        raise CaptureError('Synchronized depth association and a camera timeline are required.')
    duration = r.get('duration_seconds')
    if type(duration) not in (float, int) or not math.isfinite(duration) or not 0 < duration <= 91:
        raise CaptureError('The synchronized recording duration is invalid.')
    for key, low, high in [('video_frames_written', 1, 6000), ('video_frames_dropped', 0, 10000)]:
        if type(r.get(key)) is not int or not low <= r[key] <= high:
            raise CaptureError('The synchronized recording frame counts are invalid.')
    dimensions(r.get('color_width'), r.get('color_height'))


def validate_files(meta, directory):
    from .session import CaptureError, filename, calibration
    from .readiness import _number
    validate_manifest(meta)
    r = meta['rgbd']
    references = [r['timeline_file']]
    for p in meta['poses']:
        filename(p.get('color_file'))
        name = filename(p.get('confidence_file')); references.append(name)
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != p['width'] * p['height']:
            raise CaptureError('A LiDAR confidence map is missing or truncated.')
        if np.any(np.fromfile(path, dtype=np.uint8) > 2):
            raise CaptureError('A LiDAR confidence map contains invalid values.')
        if (p.get('tracking_state') != 'normal' or not _number(p.get('frame_timestamp_seconds'))
                or not 0 <= p['frame_timestamp_seconds'] <= r['duration_seconds']
                or p.get('rgb_world_to_camera') != p.get('world_to_camera')):
            raise CaptureError('Colour and depth are not paired with the same tracked camera frame.')
        if (p.get('rgb_width'), p.get('rgb_height')) != (r['color_width'], r['color_height']):
            raise CaptureError('The synchronized camera resolution changed during recording.')
        k = p['rgb_intrinsics']; d = p['intrinsics']
        sx, sy = p['width'] / p['rgb_width'], p['height'] / p['rgb_height']
        if any(abs(d[key] - k[key]*scale) > .01 for key, scale in [('fx', sx), ('fy', sy), ('cx', sx), ('cy', sy)]):
            raise CaptureError('Depth and colour calibration do not match the same camera frame.')
    path = directory / r['timeline_file']
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 6_000_000:
        raise CaptureError('The camera timeline is missing or invalid.')
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
        if len(lines) != r['video_frames_written']:
            raise CaptureError('The camera timeline is incomplete.')
        last = -1
        for line in lines:
            f = json.loads(line)
            t = f.get('timestamp_seconds')
            if (not _number(t) or not 0 <= t <= r['duration_seconds'] or t <= last
                    or f.get('tracking_state') not in ('normal', 'limited') or type(f.get('has_depth')) is not bool
                    or (f.get('rgb_width'), f.get('rgb_height')) != (r['color_width'], r['color_height'])):
                raise CaptureError('The camera timeline does not match its recording.')
            calibration(f.get('intrinsics'), f.get('camera_to_world_ar_m'))
            last = t
    except (ValueError, TypeError, AttributeError, UnicodeError) as error:
        raise CaptureError('The camera timeline is damaged or inconsistent.') from error
    return references


def read_depth(session, entry):
    """Return a private confidence-masked copy. Never modify sensor files."""
    from .session import CaptureError
    depth = np.fromfile(session.directory / entry['depth_file'], dtype='<f4').reshape(entry['height'], entry['width'])
    depth = depth * entry.get('depth_unit_mm', 1)
    if is_rgbd(session.metadata):
        confidence = np.fromfile(session.directory / entry['confidence_file'], dtype=np.uint8).reshape(depth.shape)
        if np.any(confidence > 2):
            raise CaptureError('Invalid confidence map.')
        depth = np.where(confidence >= 1, depth, 0)
    return depth.astype(np.float32)


def gate_stereo(session, depth, k, e, entries, tolerance_mm=10.):
    """Keep stereo samples only where BOTH measured maps support their 3D position.

    Projection uses each measured camera, not a resized/blurred depth image.
    No measured depth is invented across holes or low-confidence samples.
    """
    from .reconstruct import matrix_k, extrinsic
    y, x = np.indices(depth.shape)
    points = np.stack(((x-k[0, 2])*depth/k[0, 0], (y-k[1, 2])*depth/k[1, 1], depth), axis=-1)
    world = (points-e[:3, 3]) @ e[:3, :3]
    keep = np.isfinite(depth) & (depth > 0)
    for entry in entries:
        measured = read_depth(session, entry)
        ek, kk = extrinsic(entry, False), matrix_k(entry, depth=True)
        camera = world @ ek[:3, :3].T + ek[:3, 3]
        projected = camera @ kk.T
        z = camera[..., 2]
        u = np.rint(projected[..., 0] / np.maximum(z, 1e-6)).astype(int)
        v = np.rint(projected[..., 1] / np.maximum(z, 1e-6)).astype(int)
        valid = (z > 0) & (u >= 0) & (v >= 0) & (u < measured.shape[1]) & (v < measured.shape[0])
        samples = measured[np.clip(v, 0, measured.shape[0]-1), np.clip(u, 0, measured.shape[1]-1)]
        keep &= valid & (samples > 0) & (np.abs(z-samples) <= tolerance_mm)
    return np.where(keep, depth, 0).astype(np.float32)


def alignment_pairs(session):
    """Bounded temporal and loop-neighbour graph, with every frame represented."""
    from .reconstruct import extrinsic, camera_center
    centers = np.array([camera_center(extrinsic(p)) for p in session.poses])
    pairs = set()
    for i in range(len(centers)):
        neighbors = set(range(max(0, i-3), min(len(centers), i+4)))
        neighbors.update(np.argsort(np.linalg.norm(centers-centers[i], axis=1))[:7].tolist())
        for j in neighbors:
            if i != j: pairs.add(tuple(sorted((i, j))))
    return pairs
