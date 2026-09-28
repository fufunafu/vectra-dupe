"""Check recorded poses against the capture, then recover supported rigid drift.

Only image correspondences backed by this scan's depth may move a depth view.
The front view fixes the reference frame. Unconnected or contradictory views
fail closed. No template, invented depth, or nonrigid face fitting is used.
"""

from dataclasses import dataclass
import copy
import heapq
import math

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .session import CaptureError, Session
from .rgbd import read_depth, is_rgbd, alignment_pairs


MAX_SHIFT_MM = 300.
MAX_ROTATION_DEG = 25.
MATCH_ERROR_MM = 10.


@dataclass
class Features:
    pixels: np.ndarray
    descriptors: np.ndarray
    points: np.ndarray | None
    size: tuple[int, int]


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def rigid_fit(source, target):
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((target - b).T @ (source - a))
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, b - rotation @ a
    return matrix


def _rotation_degrees(matrix):
    return math.degrees(Rotation.from_matrix(matrix[:3, :3]).magnitude())


def _bounded(matrix):
    return (np.isfinite(matrix).all() and np.linalg.norm(matrix[:3, 3]) <= MAX_SHIFT_MM
            and _rotation_degrees(matrix) <= MAX_ROTATION_DEG)


def _face_mask(image):
    """Optional feature mask, not a geometry model. Try sensor orientations.

    A missing detection falls back to the captured depth envelope. The cascade
    files ship with OpenCV; no model or patient data is downloaded.
    """
    classifiers = [cv2.CascadeClassifier(cv2.data.haarcascades + name) for name in
                   ('haarcascade_frontalface_default.xml', 'haarcascade_profileface.xml')]
    scale = min(1., 640 / max(image.shape[:2]))
    small = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    for turns in (3, 0, 1, 2):
        upright = np.ascontiguousarray(np.rot90(small, turns))
        gray = cv2.cvtColor(upright, cv2.COLOR_RGB2GRAY)
        choices = []
        for index, classifier in enumerate(classifiers):
            for mirror in ((False,) if index == 0 else (False, True)):
                work = gray[:, ::-1].copy() if mirror else gray
                boxes = classifier.detectMultiScale(work, 1.08, 4, minSize=(65, 65))
                for x, y, w, h in boxes:
                    if mirror:
                        x = gray.shape[1] - x - w
                    choices.append((x, y, w, h))
            if choices:
                break
        if choices:
            x, y, w, h = max(choices, key=lambda box: box[2] * box[3])
            mask = np.zeros(gray.shape, np.uint8)
            mask[max(0, int(y - .15*h)):min(gray.shape[0], int(y + 1.05*h)),
                 max(0, int(x - .1*w)):min(gray.shape[1], int(x + 1.1*w))] = 255
            return cv2.resize(np.ascontiguousarray(np.rot90(mask, -turns)),
                              (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return None


def _features(session, entry, *, with_depth):
    # Imported at call time to share exactly the fusion pipeline's calibration
    # and avoid a second interpretation of the capture coordinate convention.
    from .reconstruct import matrix_k, extrinsic, photo

    scale = min(1., 1600 / max(entry['rgb_width'], entry['rgb_height']))
    size = (round(entry['rgb_width'] * scale), round(entry['rgb_height'] * scale))
    image = photo(session, entry, size)
    mask = _face_mask(image) if not session.metadata.get('is_demo') else None
    keys, descriptors = cv2.SIFT_create(nfeatures=6000, contrastThreshold=.008).detectAndCompute(
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), mask)
    pixels = np.asarray([key.pt for key in keys], dtype=float).reshape(-1, 2)
    if descriptors is None:
        descriptors = np.empty((0, 128), np.float32)
    if not with_depth or not len(pixels):
        return Features(pixels, descriptors, np.empty((0, 3)) if with_depth else None, size)

    depth = read_depth(session, entry)
    kd, kr = matrix_k(entry, depth=True), matrix_k(entry, size)
    ed, er = extrinsic(entry, False), extrinsic(entry)
    if np.allclose(ed, er, atol=1e-5):
        locations = (pixels - kr[:2, 2]) / np.diag(kr)[:2] * np.diag(kd)[:2] + kd[:2, 2]
        xy = np.rint(locations).astype(int)
        good = (xy[:, 0] >= 0) & (xy[:, 0] < depth.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < depth.shape[0])
        z = depth[np.clip(xy[:, 1], 0, depth.shape[0] - 1), np.clip(xy[:, 0], 0, depth.shape[1] - 1)]
    else:
        # A later RGB exposure has its own pose. Project captured depth into
        # that image rather than pretending the two camera positions coincide.
        y, x = np.indices(depth.shape)
        camera = np.stack(((x-kd[0, 2])*depth/kd[0, 0], (y-kd[1, 2])*depth/kd[1, 1], depth), axis=-1)
        points = transform(camera.reshape(-1, 3), np.linalg.inv(ed))
        rgb = transform(points, er)
        good_depth = (depth.ravel() > 120) & (depth.ravel() < 1200) & (rgb[:, 2] > 1)
        rgb = rgb[good_depth]
        if not len(rgb):
            return Features(pixels[:0], descriptors[:0], np.empty((0, 3)), size)
        projected = rgb[:, :2] / rgb[:, 2:] * np.diag(kr)[:2] + kr[:2, 2]
        distance, nearest = cKDTree(projected).query(pixels)
        z = rgb[nearest, 2]
        good = distance < max(2., 1.5 * kr[0, 0] / kd[0, 0])
    camera = np.column_stack(((pixels - kr[:2, 2]) / np.diag(kr)[:2] * z[:, None], z))
    world = transform(camera, np.linalg.inv(er))
    good &= (z > 120) & (z < 1200) & (np.linalg.norm(world, axis=1) < 350)
    return Features(pixels[good], descriptors[good], world[good], size)


def _matches(a, b):
    if len(a.descriptors) < 2 or len(b.descriptors) < 2:
        return np.empty(0, int), np.empty(0, int)
    pairs = cv2.BFMatcher().knnMatch(a.descriptors, b.descriptors, k=2)
    candidates = sorted((m for pair in pairs if len(pair) == 2
                         for m, n in [pair] if m.distance < .8 * n.distance), key=lambda m: m.distance)
    # Multiple SIFT scales at the same pixel are not independent observations.
    used_a, used_b, accepted = set(), set(), []
    for match in candidates:
        pa = tuple(np.round(a.pixels[match.queryIdx] / 2).astype(int))
        pb = tuple(np.round(b.pixels[match.trainIdx] / 2).astype(int))
        if pa not in used_a and pb not in used_b:
            used_a.add(pa)
            used_b.add(pb)
            accepted.append(match)
    return (np.array([m.queryIdx for m in accepted], int), np.array([m.trainIdx for m in accepted], int))


def _fit_matches(a, b, seed, adaptive=False):
    if len(a) < 8:
        return None
    rng = np.random.default_rng(seed)
    best = np.zeros(len(a), bool)
    iterations = 2000
    for attempt in range(2000):
        if adaptive and attempt >= iterations: break
        sample = rng.choice(len(a), 3, replace=False)
        # Reject collinear triples before estimating a rigid orientation.
        if np.linalg.norm(np.cross(a[sample[1]]-a[sample[0]], a[sample[2]]-a[sample[0]])) < 10:
            continue
        matrix = rigid_fit(a[sample], b[sample])
        if not _bounded(matrix):
            continue
        inlier = np.linalg.norm(transform(a, matrix)-b, axis=1) < MATCH_ERROR_MM
        if inlier.sum() > best.sum():
            best = inlier
            if adaptive:
                success = min(.999999, float(best.mean()) ** 3)
                iterations = min(iterations, max(64, math.ceil(math.log(.001)/math.log(max(1e-9, 1-success)))))
    if best.sum() < 8 or best.mean() < .18:
        return None
    matrix = rigid_fit(a[best], b[best])
    best &= np.linalg.norm(transform(a, matrix)-b, axis=1) < MATCH_ERROR_MM
    if best.sum() < 8 or not _bounded(matrix):
        return None
    # Matches must cover two dimensions of the surface, not one tiny crease.
    spread = np.linalg.svd(a[best]-a[best].mean(0), compute_uv=False) / np.sqrt(best.sum())
    if spread[1] < 8:
        return None
    return matrix, a[best], b[best]


def _geometry_connected(session):
    """Conservative fallback for depth views with no accompanying photos."""
    from .reconstruct import matrix_k, extrinsic
    clouds = []
    for entry in session.poses:
        depth = read_depth(session, entry)
        k, e = matrix_k(entry, depth=True), extrinsic(entry, False)
        y, x = np.indices(depth.shape)
        camera = np.stack(((x-k[0, 2])*depth/k[0, 0], (y-k[1, 2])*depth/k[1, 1], depth), axis=-1)
        points = transform(camera.reshape(-1, 3), np.linalg.inv(e))
        keep = (depth.ravel() > 120) & (depth.ravel() < 1200) & (np.linalg.norm(points, axis=1) < 240)
        points = points[keep][::4]
        if len(points) < 100:
            return False
        clouds.append(points)
    connected = {0}
    while True:
        added = set()
        for i in range(len(clouds)):
            if i in connected:
                continue
            for j in connected:
                distance, _ = cKDTree(clouds[j]).query(clouds[i])
                if np.mean(distance < 10.) >= .3:
                    added.add(i)
                    break
        if not added:
            return len(connected) == len(clouds)
        connected.update(added)


def _relocalize_photos(session, features, corrections, diagnostics, progress):
    from .reconstruct import matrix_k, extrinsic, camera_center
    kept = []
    records = []
    for number, entry in enumerate(session.metadata.get('color_frames', [])):
        progress(35 + int(15 * number / max(1, len(session.metadata['color_frames']))),
                 f'Aligning photograph {number + 1} of {len(session.metadata["color_frames"])}')
        target = _features(session, entry, with_depth=False)
        raw = extrinsic(entry)
        candidates = sorted(features, key=lambda i: _rotation_degrees(raw @ np.linalg.inv(extrinsic(session.poses[i]))))[:3]
        solutions = []
        for index in candidates:
            reference = features[index]
            ia, ib = _matches(reference, target)
            if len(ia) < 12:
                continue
            points = transform(reference.points[ia], corrections[index])
            pixels = target.pixels[ib]
            k = matrix_k(entry, target.size)
            cv2.setRNGSeed(7201 + number * 31 + index)
            ok, r, t, inliers = cv2.solvePnPRansac(points, pixels, k, None,
                iterationsCount=1000, reprojectionError=6., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
            if not ok or inliers is None or len(inliers) < 12 or len(inliers) / len(ia) < .25:
                continue
            chosen = inliers.ravel()
            r, t = cv2.solvePnPRefineLM(points[chosen], pixels[chosen], k, None, r, t)
            e = np.eye(4)
            e[:3, :3], e[:3, 3] = cv2.Rodrigues(r)[0], t.ravel()
            projection, _ = cv2.projectPoints(points[chosen], r, t, k, None)
            error = np.linalg.norm(projection.reshape(-1, 2) - pixels[chosen], axis=1)
            if (np.median(error) > 4 or np.percentile(error, 90) > 7
                    or _rotation_degrees(e @ np.linalg.inv(raw)) > MAX_ROTATION_DEG
                    or np.linalg.norm(camera_center(e)-camera_center(raw)) > MAX_SHIFT_MM
                    or np.any(transform(points[chosen], e)[:, 2] <= 0)):
                continue
            solutions.append((len(chosen), float(np.median(error)), e))
        if solutions:
            count, error, e = max(solutions, key=lambda value: value[0] / max(1., value[1]))
            fixed = copy.deepcopy(entry)
            fixed['world_to_camera'] = e.tolist()
            if 'rgb_world_to_camera' in fixed:
                fixed['rgb_world_to_camera'] = e.tolist()
            kept.append(fixed)
            records.append({'name': entry['name'], 'status': 'aligned', 'inliers': count, 'median_pixel_error': round(error, 3)})
        else:
            records.append({'name': entry['name'], 'status': 'excluded', 'reason': 'No reliable alignment to captured depth.'})
    diagnostics['photographs'] = records
    diagnostics['aligned_sweep_photos'] = len(kept)
    diagnostics['excluded_sweep_photos'] = len(records) - len(kept)
    return kept


def align_depth_session(session, diagnostics, progress):
    """Return a corrected in-memory manifest; never alter source capture files."""
    from .reconstruct import extrinsic
    diagnostics.update(method='photo_depth_rigid', status='checking', views=[], pairs=[])
    if len(session.poses) < 2:
        raise CaptureError('At least two depth views are needed to verify alignment.')
    if len(session.poses) > (90 if is_rgbd(session.metadata) else 24):
        raise CaptureError('This capture has too many depth views for desktop alignment. Export one guided scan at a time.')
    features = {}
    for i, entry in enumerate(session.poses):
        progress(3 + int(12 * i / len(session.poses)), f'Checking alignment of depth view {i + 1} of {len(session.poses)}')
        if entry.get('color_file'):
            features[i] = _features(session, entry, with_depth=True)
    if not features:
        if not _geometry_connected(session):
            diagnostics['status'] = 'rejected'
            raise CaptureError('The depth views do not overlap consistently. Camera tracking drift or subject movement may have displaced them. Capture again with the subject still.')
        diagnostics.update(method='recorded_depth_overlap', status='recorded', note='No depth photographs were available for visual pose verification.')
        return session

    edges = []
    candidate_pairs = alignment_pairs(session) if is_rgbd(session.metadata) else None
    for i in features:
        for j in features:
            if j >= i or (candidate_pairs is not None and (j, i) not in candidate_pairs):
                continue
            ia, ib = _matches(features[i], features[j])
            fitted = _fit_matches(features[i].points[ia], features[j].points[ib], 81 + i * 31 + j, adaptive=is_rgbd(session.metadata))
            if fitted is None:
                continue
            matrix, a, b = fitted
            record = {'views': [session.poses[i]['name'], session.poses[j]['name']], 'inliers': len(a),
                      'recorded_error_mm': round(float(np.median(np.linalg.norm(a-b, axis=1))), 3)}
            diagnostics['pairs'].append(record)
            edges.append((i, j, matrix, a, b, record))

    anchor = next((i for i, entry in enumerate(session.poses) if entry['name'] == 'front'), 0)
    corrections = {anchor: np.eye(4)}
    distances = {anchor: 0.}
    queue = [(0., anchor)]
    while queue:
        cost, current = heapq.heappop(queue)
        if cost != distances[current]:
            continue
        for i, j, matrix, a, b, _ in edges:
            if current not in (i, j):
                continue
            other = j if current == i else i
            next_cost = cost + 1. / math.sqrt(len(a))
            if next_cost >= distances.get(other, math.inf):
                continue
            distances[other] = next_cost
            corrections[other] = corrections[current] @ (np.linalg.inv(matrix) if current == i else matrix)
            heapq.heappush(queue, (next_cost, other))
    if len(corrections) != len(session.poses):
        missing = [p['name'] for i, p in enumerate(session.poses) if i not in corrections]
        diagnostics.update(status='rejected', unverified_views=missing)
        raise CaptureError('The scan views could not be aligned reliably (' + ', '.join(missing) + '). '
                           'A 3D model was not saved. Capture again with the subject still, moving the phone slowly through overlapping views.')

    before = np.concatenate([np.linalg.norm(a-b, axis=1) for _, _, _, a, b, _ in edges])
    diagnostics['recorded_median_error_mm'] = round(float(np.median(before)), 3)
    if np.percentile(before, 90) <= 5.:
        diagnostics['status'] = 'recorded'
        return session

    ids = [i for i in sorted(corrections) if i != anchor]
    initial = np.array([np.r_[Rotation.from_matrix(corrections[i][:3, :3]).as_rotvec(), corrections[i][:3, 3]] for i in ids])
    def matrices(values):
        result = {anchor: np.eye(4)}
        for i, value in zip(ids, values.reshape(-1, 6)):
            e = np.eye(4)
            e[:3, :3], e[:3, 3] = Rotation.from_rotvec(value[:3]).as_matrix(), value[3:]
            result[i] = e
        return result
    optimization_edges = edges
    sparse_options = {}
    if is_rgbd(session.metadata):
        from scipy.sparse import lil_matrix
        optimization_edges = []
        for i, j, matrix, a, b, record in edges:
            sample = np.unique(np.linspace(0, len(a)-1, min(128, len(a))).round().astype(int))
            optimization_edges.append((i, j, matrix, a[sample], b[sample], record))
        pattern = lil_matrix((sum(len(a)*3 for _, _, _, a, _, _ in optimization_edges), len(ids)*6), dtype=np.int8)
        columns = {view:index*6 for index,view in enumerate(ids)}
        row = 0
        for i, j, _, a, _, _ in optimization_edges:
            for view in (i, j):
                if view in columns: pattern[row:row+len(a)*3,columns[view]:columns[view]+6] = 1
            row += len(a)*3
        sparse_options = {'jac_sparsity': pattern.tocsr(), 'tr_solver': 'lsmr'}
    def residual(values):
        poses = matrices(values)
        return np.concatenate([(transform(a, poses[i])-transform(b, poses[j])).ravel() for i, j, _, a, b, _ in optimization_edges])
    optimized = least_squares(residual, initial.ravel(), loss='soft_l1', f_scale=4., max_nfev=80, **sparse_options)
    corrections = matrices(optimized.x)
    for i, j, _, a, b, record in edges:
        error = np.linalg.norm(transform(a, corrections[i])-transform(b, corrections[j]), axis=1)
        record['aligned_median_error_mm'] = round(float(np.median(error)), 3)
        record['aligned_p90_error_mm'] = round(float(np.percentile(error, 90)), 3)
        if np.percentile(error, 90) > 12:
            diagnostics['status'] = 'rejected'
            raise CaptureError('The scan contains inconsistent views that cannot form one rigid surface. Capture again with a still subject and steady tracking.')
    if not all(_bounded(e) for e in corrections.values()):
        diagnostics['status'] = 'rejected'
        raise CaptureError('The scan needs a camera-position correction beyond the supported recovery range. Please capture it again.')

    if is_rgbd(session.metadata) and any(np.linalg.norm(e[:3, 3]) > 30 or _rotation_degrees(e) > 5 for e in corrections.values()):
        diagnostics['status'] = 'rejected'
        raise CaptureError('The continuous recording needs excessive pose correction. Record a new pass with the subject still.')
    metadata = copy.deepcopy(session.metadata)
    for i, entry in enumerate(metadata['poses']):
        inverse = np.linalg.inv(corrections[i])
        original = session.poses[i]
        entry['world_to_camera'] = (extrinsic(original, False) @ inverse).tolist()
        if original.get('color_file'):
            entry['rgb_world_to_camera'] = (extrinsic(original) @ inverse).tolist()
        diagnostics['views'].append({'name': entry['name'], 'correction_mm': round(float(np.linalg.norm(corrections[i][:3, 3])), 3),
                                     'correction_degrees': round(_rotation_degrees(corrections[i]), 3),
                                     'world_correction': corrections[i].tolist()})
    metadata['color_frames'] = _relocalize_photos(session, features, corrections, diagnostics, progress)
    diagnostics.update(status='recovered', reference_view=session.poses[anchor]['name'],
                       aligned_median_error_mm=round(float(np.median(np.linalg.norm(residual(optimized.x).reshape(-1, 3), axis=1))), 3))
    return Session(session.directory, metadata)
