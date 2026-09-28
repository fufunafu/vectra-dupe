"""Capture completeness checks before reconstruction, not an accuracy score.

Use the same guided views, photo screens and coverage grid as the iPhone.
File validation runs separately. A saved flag or high photo count is not proof
of coverage. Camera positions are only recorded evidence, not verified alignment.
"""

import math
import numpy as np

GUIDED = ('front', 'left_half', 'left', 'right_half', 'right', 'brow', 'jaw')


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _usable(quality):
    if not isinstance(quality, dict):
        return False
    sharp, mean, clipped = (quality.get(k) for k in ('sharpness', 'mean_luma', 'clipped_fraction'))
    region = quality.get('region')
    if not isinstance(region, dict) or not all(_number(region.get(k)) for k in ('x', 'y', 'width', 'height')):
        return False
    return (all(_number(v) for v in (sharp, mean, clipped))
            and sharp >= 8 and 20 <= mean <= 235 and 0 <= clipped < .5
            and min(region['x'], region['y']) >= 0 and min(region['width'], region['height']) > 0
            and region['x'] + region['width'] <= 1.000001 and region['y'] + region['height'] <= 1.000001
            and ('exposure_duration_seconds' not in quality or (_number(quality['exposure_duration_seconds'])
                 and 0 < quality['exposure_duration_seconds'] <= 10)))


def _view(entry):
    try:
        e = np.asarray(entry['world_to_camera'], dtype=float)
        if e.shape != (4, 4) or not np.isfinite(e).all():
            return None
        r = e[:3, :3]
        if not np.allclose(r @ r.T, np.eye(3), atol=.002) or abs(np.linalg.det(r) - 1) > .002:
            return None
        x, y, z = np.linalg.inv(e)[:3, 3]
        if np.linalg.norm([x, y, z]) < 1:
            return None
        yaw = math.degrees(math.atan2(-x, z))
        pitch = math.degrees(math.atan2(y, math.hypot(x, z)))
        # Swift rounds ties away from zero.
        yi = int(math.copysign(math.floor(abs(yaw / 15) + .5), yaw))
        pi = min(range(3), key=lambda i: abs(pitch - (-25, 0, 25)[i]))
        return yaw, pitch, yi, pi
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return None


def assess(metadata):
    """Return actionable blockers. Synthetic demos never claim real readiness."""
    if metadata.get('format') == 'facemap-rgbd-session/1':
        from .rgbd import assess as assess_rgbd
        return assess_rgbd(metadata)
    if metadata.get('is_demo') is True:
        return {'status': 'demo', 'issues': [], 'policy_version': 1}
    issues = []
    poses = metadata.get('poses', [])
    colors = metadata.get('color_frames', []) or []
    photo_only = metadata.get('capture_kind') == 'photo_only'
    guided = {p.get('name') for p in poses if p.get('color_file')}
    if photo_only:
        guided = {p.get('name', '')[4:] for p in colors if p.get('name', '').startswith('key_')}
    missing = [name.replace('_', ' ') for name in GUIDED if name not in guided]
    if missing:
        issues.append('Retake the missing guided views: ' + ', '.join(missing) + '.')
    if not photo_only:
        weak = []
        for pose in poses:
            q = pose.get('quality')
            if not isinstance(q, dict):
                q = {}
            values = [q.get(k) for k in ('accepted_frames', 'valid_depth_fraction', 'maximum_motion',
                                        'maximum_translation_mm', 'maximum_rotation_degrees')]
            if (not all(_number(v) for v in values) or not isinstance(q.get('accepted_frames'), int)
                    or values[0] < 3 or not 0 < values[1] <= 1 or not 0 <= values[2] < 55
                    or not 0 <= values[3] <= 2 or not 0 <= values[4] <= 1):
                weak.append(pose.get('name', 'unknown').replace('_', ' '))
        if weak:
            issues.append('Depth stability is insufficient or unverified. Retake: ' + ', '.join(weak) + '.')
    cells, views = set(), []
    rejected = 0
    for photo in colors:
        if not photo.get('name', '').startswith('orbit_'):
            continue
        view = _view(photo)
        if not _usable(photo.get('photo_quality')) or view is None:
            rejected += 1
            continue
        yaw, pitch, yi, pi = view
        if not -6 <= yi <= 6 or abs(pitch - (-25, 0, 25)[pi]) > 18:
            continue
        cells.add((yi, pi))
        if all((yaw - a) ** 2 + (pitch - b) ** 2 >= 16 for a, b in views):
            views.append((yaw, pitch))
    minimum = 20 if photo_only else 12
    if len(views) < minimum:
        issues.append(f'Capture at least {minimum} sharp automatic photos from distinct nearby angles; {len(views)} qualify.')
    missing_regions = []
    for band, height in enumerate(('lower', 'level', 'upper')):
        for side, indices in (('left', range(-6, -1)), ('front', range(-1, 2)), ('right', range(2, 7))):
            if not any((i, band) in cells for i in indices):
                missing_regions.append(f'{height} {side}')
    if missing_regions:
        issues.append('Add sharp overlapping photos for: ' + ', '.join(missing_regions) + '.')
    if rejected:
        issues.append(f'{rejected} automatic photos lack usable quality or camera-position evidence. Retake those views.')
    # Older captures persisted a gap warning instead of structured readiness.
    for warning in metadata.get('warnings', []) or []:
        if isinstance(warning, str) and ('Missing coverage:' in warning or 'Recovered unfinished scan' in warning):
            issues.append('This scan was saved incomplete. ' + warning.split('\n')[0])
    return {'status': 'blocked' if issues else 'passed', 'issues': issues,
            'policy_version': 1, 'usable_distinct_photos': len(views), 'missing_regions': missing_regions}


def message(result):
    return 'Scan is not ready for reconstruction.\n\n' + '\n'.join(result['issues']) + '\n\nKeep the original scan. Capture the missing views in the guided pass, or record a new scan if the subject has moved.'
