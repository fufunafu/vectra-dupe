"""Bounded CPU depth fusion and calibrated stereo for iPhone exports.

No Apple SDK, model downloads, CUDA, or connection to the production server.
All geometry is reconstructed from the current capture. No face template fills
missing anatomy. Photo-only reconstruction is experimental and may reject a scan.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

# Limit native thread pools before importing numerical libraries in a worker.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, str(min(4, os.cpu_count() or 1)))

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import trimesh

from .session import CaptureError, Session, open_session


@dataclass(frozen=True)
class Settings:
    image_size: int
    max_views: int
    voxel_mm: float
    max_triangles: int


PRESETS = {
    'quick': Settings(640, 12, 2.5, 50000),
    'balanced': Settings(960, 24, 1.5, 100000),
    'detail': Settings(1280, 36, 1.0, 150000),
}


def matrix_k(entry, size=None, depth=False):
    k = entry['intrinsics' if depth else 'rgb_intrinsics']
    w, h = (entry['width'], entry['height']) if depth else (entry['rgb_width'], entry['rgb_height'])
    sx, sy = (size[0] / w, size[1] / h) if size else (1, 1)
    # OpenCV resize uses pixel-centre mapping.
    return np.array([[k['fx'] * sx, 0, (k['cx'] + .5) * sx - .5],
                     [0, k['fy'] * sy, (k['cy'] + .5) * sy - .5], [0, 0, 1]], dtype=np.float64)


def extrinsic(entry, rgb=True):
    return np.asarray(entry.get('rgb_world_to_camera', entry['world_to_camera']) if rgb else entry['world_to_camera'], dtype=np.float64)


def camera_center(e):
    return -e[:3, :3].T @ e[:3, 3]


def photo(session, entry, size):
    with Image.open(session.directory / entry['color_file']) as image:
        return np.asarray(image.convert('RGB').resize(size, Image.Resampling.LANCZOS)).copy()


def crop_depth(depth, k, e, radius=240.):
    """The iPhone stores the subject near the world origin, in millimetres."""
    y, x = np.indices(depth.shape)
    points = np.stack(((x - k[0, 2]) * depth / k[0, 0],
                       (y - k[1, 2]) * depth / k[1, 1], depth), axis=-1)
    world = (points - e[:3, 3]) @ e[:3, :3]
    valid = np.isfinite(depth) & (depth >= 120) & (depth <= 1200)
    valid &= np.linalg.norm(world, axis=-1) <= radius
    return np.where(valid, depth, 0).astype(np.float32)


def _rgb_for_depth(session, entry, depth):
    """Project via the RGB pose too, since camera handoffs can move the phone."""
    if not entry.get('color_file'):
        return np.full((*depth.shape, 3), 175, dtype=np.uint8)
    h, w = depth.shape
    scale = min(1, 1280 / max(entry['rgb_width'], entry['rgb_height']))
    size = (round(entry['rgb_width'] * scale), round(entry['rgb_height'] * scale))
    rgb = photo(session, entry, size)
    k = matrix_k(entry, depth=True)
    y, x = np.indices((h, w))
    p = np.stack(((x - k[0, 2]) * depth / k[0, 0], (y - k[1, 2]) * depth / k[1, 1], depth), axis=-1)
    world = (p - extrinsic(entry, False)[:3, 3]) @ extrinsic(entry, False)[:3, :3]
    ergb = extrinsic(entry)
    camera = world @ ergb[:3, :3].T + ergb[:3, 3]
    q = camera @ matrix_k(entry, size).T
    z = np.maximum(q[..., 2], 1e-6)
    return cv2.remap(rgb, (q[..., 0] / z).astype('float32'), (q[..., 1] / z).astype('float32'),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(175, 175, 175))


def integrate(volume, depth, rgb, k, e):
    h, w = depth.shape
    image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb, dtype=np.uint8)),
        o3d.geometry.Image(np.ascontiguousarray(depth, dtype=np.float32)),
        depth_scale=1., depth_trunc=1200., convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
    volume.integrate(image, intrinsic, e)


def stereo_pairs(photos, max_views):
    """Choose moderate-baseline pairs by pose, not filename or capture order."""
    if len(photos) < 3:
        raise CaptureError("Photo reconstruction needs at least three overlapping views. Capture a slower pass around the face.")
    matrices = [extrinsic(p) for p in photos]
    centers = np.array([camera_center(e) for e in matrices])
    refs = np.unique(np.linspace(0, len(photos) - 1, min(max_views, len(photos))).round().astype(int))
    pairs, used = [], set()
    for i in refs:
        choices = []
        for j in range(len(photos)):
            if j == i or tuple(sorted((i, j))) in used:
                continue
            baseline = np.linalg.norm(centers[i] - centers[j])
            angle = math.degrees(math.acos(np.clip((np.trace(matrices[j][:3, :3] @ matrices[i][:3, :3].T) - 1) / 2, -1, 1)))
            if 15 <= baseline <= 110 and angle <= 24:
                score = abs(baseline - 45) + angle * 1.5
                choices.append((score, j))
        if choices:
            j = min(choices)[1]
            used.add(tuple(sorted((i, j))))
            pairs.append((int(i), j))
    if len(pairs) < 2:
        raise CaptureError("These photos do not have enough overlapping camera positions for CPU reconstruction. Capture more closely spaced views.")
    return pairs


def rectified_pair(session, left, right, image_size):
    scale = min(1., image_size / max(left['rgb_width'], left['rgb_height']))
    size = (round(left['rgb_width'] * scale), round(left['rgb_height'] * scale))
    k1, k2 = matrix_k(left, size), matrix_k(right, size)
    e1, e2 = extrinsic(left), extrinsic(right)
    relative = e2 @ np.linalg.inv(e1)
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(k1, None, k2, None, size,
        relative[:3, :3], relative[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    vertical = bool(abs(p2[1, 3]) > abs(p2[0, 3]))
    axis = 1 if vertical else 0
    if p2[axis, 3] > 0:
        return rectified_pair(session, right, left, image_size)
    if not 0 < p1[0, 0] < image_size * 4:
        raise CaptureError("This camera pair has too little shared field of view.")
    maps1 = cv2.initUndistortRectifyMap(k1, None, r1, p1[:3, :3], size, cv2.CV_32FC1)
    maps2 = cv2.initUndistortRectifyMap(k2, None, r2, p2[:3, :3], size, cv2.CV_32FC1)
    rgb1 = cv2.remap(photo(session, left, size), *maps1, cv2.INTER_LINEAR)
    rgb2 = cv2.remap(photo(session, right, size), *maps2, cv2.INTER_LINEAR)
    valid = (maps1[0] >= 1) & (maps1[0] < size[0] - 2) & (maps1[1] >= 1) & (maps1[1] < size[1] - 2)
    valid2 = (maps2[0] >= 1) & (maps2[0] < size[0] - 2) & (maps2[1] >= 1) & (maps2[1] < size[1] - 2)
    er = np.eye(4)
    er[:3, :3] = r1
    return rgb1, rgb2, valid, valid2, p1[:3, :3], er @ e1, q, vertical, abs(p2[axis, 3])


def stereo_depth(session, left, right, image_size):
    rgb1, rgb2, valid, valid2, k, e, q, vertical, fb = rectified_pair(session, left, right, image_size)
    a, b = cv2.cvtColor(rgb1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(rgb2, cv2.COLOR_RGB2GRAY)
    if vertical:
        a, b, valid, valid2 = a.T.copy(), b.T.copy(), valid.T.copy(), valid2.T.copy()
    minimum = max(0, int(fb / 1200) - 4)
    number = min(256, int(math.ceil((fb / 160 - minimum + 8) / 16)) * 16)
    number = min(number, ((a.shape[1] - minimum - 16) // 16) * 16)
    if number < 16:
        raise CaptureError("This camera pair cannot be matched at the selected resolution.")
    matcher = cv2.StereoSGBM_create(minDisparity=minimum, numDisparities=number,
        blockSize=5, P1=8 * 25, P2=32 * 25, disp12MaxDiff=1, uniquenessRatio=12,
        speckleWindowSize=80, speckleRange=2, preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = matcher.compute(a, b).astype(np.float32) / 16
    reverse = matcher.compute(np.fliplr(b).copy(), np.fliplr(a).copy()).astype(np.float32)[:, ::-1] / 16
    y, x = np.indices(d.shape)
    xr = np.rint(x - d).astype(int)
    in_bounds = (xr >= 0) & (xr < d.shape[1])
    xr = np.clip(xr, 0, d.shape[1] - 1)
    reverse_at = reverse[y, xr]
    valid &= in_bounds & valid2[y, xr] & (d > max(.5, minimum)) & (d < minimum + number - 1)
    valid &= (reverse_at > max(.5, minimum)) & (np.abs(d - reverse_at) <= 1.5)
    # Reject flat patches where a smooth disparity is unsupported by texture.
    af = a.astype(np.float32)
    variance = cv2.boxFilter(af * af, -1, (7, 7)) - cv2.boxFilter(af, -1, (7, 7)) ** 2
    valid &= variance > 6
    if vertical:
        d, valid = d.T.copy(), valid.T.copy()
    points = cv2.reprojectImageTo3D(d, q)
    depth = crop_depth(np.where(valid, points[..., 2], 0), k, e)
    count = int(np.count_nonzero(depth))
    if count < 300:
        raise CaptureError("This photo pair has too few reliable stereo matches.")
    return depth, rgb1, k, e, {'matched_pixels': count, 'vertical_stereo': vertical}


def _clean_mesh(mesh, settings):
    if len(mesh.triangles) < 100:
        raise CaptureError("Not enough consistent surface could be reconstructed. Try a slower capture with even lighting.")
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    labels, counts, _ = mesh.cluster_connected_triangles()
    counts = np.asarray(counts)
    largest_fraction = float(counts.max() / counts.sum())
    if largest_fraction < .65:
        raise CaptureError("The scan did not form one consistent surface. Camera tracking drift or subject movement may have displaced the views. Capture again with the subject still and overlapping photos.")
    mesh.remove_triangles_by_mask(counts[np.asarray(labels)] < max(30, counts.max() * .025))
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > settings.max_triangles:
        mesh = mesh.simplify_quadric_decimation(settings.max_triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    if not np.isfinite(np.asarray(mesh.vertices)).all():
        raise CaptureError('Surface reconstruction produced invalid coordinates. Try Quick mode or capture the scan again.')
    # A mild non-shrinking display filter suppresses voxel-scale noise. It
    # neither fills holes nor fits a generic face. The report records it.
    smoothed = mesh.filter_smooth_taubin(number_of_iterations=8)
    smoothing_applied = bool(np.isfinite(np.asarray(smoothed.vertices)).all())
    if smoothing_applied:
        smoothing_applied = bool(np.max(np.linalg.norm(np.asarray(smoothed.vertices) - np.asarray(mesh.vertices), axis=1)) < settings.voxel_mm * 2)
    if smoothing_applied:
        mesh = smoothed
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh, largest_fraction, smoothing_applied


def _texture(session, mesh, progress):
    """Photo-project visible triangles into a bounded atlas, retaining holes."""
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    centers = vertices[faces].mean(axis=1)
    normals = np.asarray(mesh.triangle_normals)
    photos = session.photos
    if not photos:
        return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=np.asarray(mesh.vertex_colors), process=False), 0.
    indexes = np.unique(np.linspace(0, len(photos) - 1, min(12, len(photos))).round().astype(int))
    entries = [photos[i] for i in indexes]
    cell, columns = 1024, 4
    atlas = Image.new('RGB', (cell * columns, cell * 4), (175, 175, 175))
    best = np.full(len(faces), -1., dtype=np.float64)
    uv = np.tile([(.5 + 3 * cell) / (4 * cell), 1 - (.5 + 3 * cell) / (4 * cell)], (len(faces), 3, 1))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    for i, entry in enumerate(entries):
        progress(78 + int(i / len(entries) * 14), f'Adding photograph {i + 1} of {len(entries)}')
        scale = min(cell / entry['rgb_width'], cell / entry['rgb_height'])
        size = (max(1, round(entry['rgb_width'] * scale)), max(1, round(entry['rgb_height'] * scale)))
        image = photo(session, entry, size)
        ox, oy = (i % columns) * cell, (i // columns) * cell
        atlas.paste(Image.fromarray(image), (ox, oy))
        e, k = extrinsic(entry), matrix_k(entry, size)
        p = vertices @ e[:3, :3].T + e[:3, 3]
        q = p @ k.T
        coords = q[:, :2] / np.maximum(q[:, 2:], 1e-6)
        in_image = (p[:, 2] > 1) & (coords[:, 0] > 1) & (coords[:, 0] < size[0] - 2) & (coords[:, 1] > 1) & (coords[:, 1] < size[1] - 2)
        c = camera_center(e)
        direction = c - centers
        distance = np.linalg.norm(direction, axis=1)
        facing = np.maximum(0, (normals * direction / np.maximum(distance[:, None], 1)).sum(axis=1))
        rays = np.column_stack((np.broadcast_to(c, centers.shape), centers - c)).astype('float32')
        hit = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        visible = np.isfinite(hit) & (np.abs(hit - 1) * distance < 3)
        score = facing ** 2 / np.maximum(distance, 1) ** 2
        choose = in_image[faces].all(axis=1) & visible & (facing > .12) & (score > best)
        best[choose] = score[choose]
        projected = coords[faces[choose]]
        uv[choose, :, 0] = (projected[:, :, 0] + .5 + ox) / atlas.width
        uv[choose, :, 1] = 1 - (projected[:, :, 1] + .5 + oy) / atlas.height
    model = trimesh.Trimesh(vertices=vertices[faces].reshape(-1, 3), faces=np.arange(len(faces) * 3).reshape(-1, 3), process=False)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=atlas, roughnessFactor=1., metallicFactor=0., doubleSided=True)
    model.visual = trimesh.visual.TextureVisuals(uv=uv.reshape(-1, 2), material=material)
    return model, float(np.mean(best >= 0))


def reconstruct(source, output, mode='auto', preset='balanced', progress=None):
    progress = progress or (lambda percent, message: None)
    if mode not in ('auto', 'depth', 'photos') or preset not in PRESETS:
        raise ValueError('Unknown reconstruction mode or preset.')
    output = Path(output).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'processing', 'engine': 'facemap-cpu', 'version': '0.1.0',
              'preset': preset, 'measurement_validated': False, 'warnings': [], 'pairs': [],
              'surface_smoothing': {'method': 'taubin', 'iterations': 8}}
    started = time.monotonic()
    cv2.setNumThreads(min(4, os.cpu_count() or 1))
    def save_report():
        report['elapsed_seconds'] = round(time.monotonic() - started, 2)
        temp = output / 'report.json.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temp.replace(output / 'report.json')
    save_report()
    try:
        progress(1, 'Checking and importing scan')
        with open_session(source, output) as session:
            report.update(session.summary())
            settings = PRESETS[preset]
            selected = 'depth' if mode == 'auto' and session.poses else ('photos' if mode == 'auto' else mode)
            report['mode'] = selected
            if selected == 'depth' and not session.poses:
                raise CaptureError('This scan has no LiDAR depth. Select Automatic or Photos mode.')
            if selected == 'depth':
                from .alignment import align_depth_session
                report['depth_alignment'] = {}
                session = align_depth_session(session, report['depth_alignment'], progress)
                if report['depth_alignment']['status'] == 'recovered':
                    report['warnings'].append('Camera positions were recovered from matching photographs and captured depth. Inspect the recovered surface for residual seams or gaps.')
                    excluded = report['depth_alignment'].get('excluded_sweep_photos', 0)
                    if excluded:
                        report['warnings'].append(f'{excluded} photographs could not be aligned reliably and were excluded from texture projection. The original export is unchanged.')
                report['texture_source_photos'] = len(session.photos)
                save_report()
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=settings.voxel_mm, sdf_trunc=max(10. if selected == 'depth' else 6., settings.voxel_mm * 4),
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                depth_sampling_stride=2)
            integrated = 0
            if selected == 'depth':
                for i, entry in enumerate(session.poses):
                    progress(50 + int(18 * i / len(session.poses)), f'Combining depth view {i + 1} of {len(session.poses)}')
                    depth = np.fromfile(session.directory / entry['depth_file'], dtype='<f4').reshape(entry['height'], entry['width'])
                    depth = depth * entry.get('depth_unit_mm', 1)
                    # Smooth captured samples only; never turn missing depth
                    # into measured data. Reject isolated large discontinuities.
                    median = cv2.medianBlur(depth.astype(np.float32), 3)
                    depth = np.where((depth > 0) & (median > 0) & (np.abs(depth - median) <= 20), median, 0)
                    k, e = matrix_k(entry, depth=True), extrinsic(entry, False)
                    depth = crop_depth(depth, k, e)
                    if np.count_nonzero(depth) < 100:
                        report['warnings'].append(f'Depth view {i + 1} contained too little usable face data.')
                        continue
                    integrate(volume, depth, _rgb_for_depth(session, entry, depth), k, e)
                    integrated += 1
                if integrated < 2:
                    raise CaptureError('At least two usable depth views are needed. Complete the guided face capture and export it again.')
                report['warnings'].append('LiDAR surface detail is limited by the captured depth resolution.')
            else:
                report['warnings'].append('Photo-only CPU stereo is experimental. It can leave gaps and is not equivalent to Apple Object Capture.')
                pairs = stereo_pairs(session.photos, settings.max_views)
                for i, (a, b) in enumerate(pairs):
                    progress(8 + int(60 * i / len(pairs)), f'Matching photo pair {i + 1} of {len(pairs)}')
                    record = {'frames': [a, b]}
                    try:
                        depth, rgb, k, e, details = stereo_depth(session, session.photos[a], session.photos[b], settings.image_size)
                        integrate(volume, depth, rgb, k, e)
                        integrated += 1
                        record.update(status='integrated', **details)
                    except CaptureError as error:
                        record.update(status='rejected', reason=str(error))
                    report['pairs'].append(record)
                    save_report()
                if integrated < 2:
                    raise CaptureError('Not enough reliable photo matches. Use a slower capture, a still subject and even lighting, or capture with a LiDAR iPhone.')
            report['integrated_views'] = integrated
            progress(70, 'Building the surface')
            mesh, main_fraction, smoothing_applied = _clean_mesh(volume.extract_triangle_mesh(), settings)
            report['surface_smoothing']['applied'] = smoothing_applied
            if not smoothing_applied:
                report['warnings'].append('Surface smoothing was skipped because it exceeded the geometry safeguards.')
            model, coverage = _texture(session, mesh, progress)
            # The capture's subject frame already uses Y-up and Z toward the
            # front. Preserve it: sensor-image rotation is not head rotation.
            transform = np.diag([.001, .001, .001, 1.])
            transform[:3, 3] = -np.asarray(mesh.vertices).mean(axis=0) * .001
            model.apply_transform(transform)
            progress(95, 'Saving the 3D model')
            model.export(output / 'model.glb', file_type='glb')
            if not o3d.io.write_triangle_mesh(str(output / 'surface-mm.ply'), mesh):
                raise RuntimeError('Could not save the surface file.')
            report.update(status='complete', vertices=len(mesh.vertices), triangles=len(mesh.triangles),
                          texture_coverage=round(coverage, 4), largest_component_fraction=round(main_fraction, 4),
                          glb_units='metres', ply_units='millimetres', native_to_glb=transform.tolist())
            report['warnings'].append('Inspect the surface for gaps and distortion. This model is not a validated clinical measurement.')
            if coverage < .7:
                report['warnings'].append('Some surface areas have no reliable photograph and are shown in grey.')
            save_report()
            progress(100, 'Model saved. Inspect it before use.')
            return report
    except BaseException as error:
        report.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        # A failed export must never leave a file that looks like a finished model.
        for name in ('model.glb', 'surface-mm.ply'):
            (output / name).unlink(missing_ok=True)
        save_report()
        raise
