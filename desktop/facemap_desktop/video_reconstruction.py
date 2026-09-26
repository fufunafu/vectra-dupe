"""Video SfM and CPU surface reconstruction, with explicitly unknown metric scale."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from .session import CaptureError, Session
from .video_input import DEFAULT_REGION, prepare_video


@dataclass(frozen=True)
class VideoSettings:
    frames: int
    feature_size: int


VIDEO_PRESETS = {
    'quick': VideoSettings(36, 1280),
    'balanced': VideoSettings(64, 1600),
    'detail': VideoSettings(100, 2000),
}


def recover_cameras(images, work, settings, region, progress):
    import pycolmap
    threads = min(4, os.cpu_count() or 1)
    proxies = work / 'features'
    proxies.mkdir()
    paths = sorted(images.glob('frame_*.png'))
    for path in paths:
        with Image.open(path) as image:
            image.thumbnail((settings.feature_size, settings.feature_size), Image.Resampling.LANCZOS)
            image.save(proxies / path.name)
            size = image.size
    width, height = size
    mask = np.zeros((height, width), np.uint8)
    x, y, w, h = region
    mask[round(y * height):round((y + h) * height), round(x * width):round((x + w) * width)] = 255
    mask_path = work / 'subject-mask.png'
    Image.fromarray(mask).save(mask_path)
    database = work / 'features.db'
    # COLMAP 4 has no Intel Mac wheel. The final 3.12 wheel uses the same
    # reconstruction operations with different feature option names.
    modern = hasattr(pycolmap, 'FeatureExtractionOptions')
    extraction = {'max_image_size': settings.feature_size, 'num_threads': threads, 'use_gpu': False}
    extraction.update({'sift': {'max_num_features': 6000}} if modern else {'max_num_features': 6000})
    progress(27, 'Finding matching details in the video')
    pycolmap.extract_features(str(database), str(proxies), camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options={'camera_model': 'SIMPLE_RADIAL', 'camera_mask_path': str(mask_path)},
        **{'extraction_options' if modern else 'sift_options': extraction}, device=pycolmap.Device.cpu)
    progress(35, 'Matching overlapping views')
    matching = {'num_threads': threads, 'use_gpu': False}
    pairing = {'overlap': 8, 'quadratic_overlap': True, 'loop_detection': False, 'num_threads': threads}
    pycolmap.match_sequential(str(database), device=pycolmap.Device.cpu,
        **({'matching_options': matching, 'pairing_options': pairing} if modern
           else {'sift_options': matching, 'matching_options': pairing}))
    progress(43, 'Recovering the camera positions')
    mapping = {
        'num_threads': threads, 'multiple_models': False, 'min_model_size': 12,
        'min_focal_length_ratio': .35, 'max_focal_length_ratio': 3.,
        'max_extra_param': .5, 'ba_local_max_num_iterations': 30, 'ba_global_max_num_iterations': 60,
        'mapper': {'init_min_tri_angle': 5., 'init_min_num_inliers': 70}}
    if modern:
        mapping.update(random_seed=0, max_runtime_seconds=900)
    else:
        pycolmap.set_random_seed(0)
    deadline = time.monotonic() + 900

    def check_mapping_time():
        if time.monotonic() > deadline:
            raise CaptureError('Camera recovery took too long. Try a shorter, clearer video.')

    with (pycolmap.Database.open(str(database)) if modern else pycolmap.Database(str(database))) as db:
        image_ids = [image.image_id for image in sorted(db.read_all_images(), key=lambda image: image.name)]
    if len(image_ids) < 12:
        raise CaptureError('Too few usable video frames remain for camera recovery.')
    # A nearly adjacent automatic starting pair can produce a two-view dead end.
    # Retry with separated, already matched views. Keep the same inlier,
    # coverage and geometry requirements; never accept that partial model.
    gap = min(8, len(image_ids) - 1)
    starts = [max(0, min(len(image_ids) - gap - 1, center - gap // 2))
              for center in (len(image_ids) // 2, len(image_ids) // 3)]
    initial_pairs = [None, *dict.fromkeys((image_ids[start], image_ids[start + gap]) for start in starts)]
    required_views = max(12, math.ceil(len(paths) * .6))
    model = None
    attempts = 0
    for attempts, initial_pair in enumerate(initial_pairs, 1):
        check_mapping_time()
        options = dict(mapping)
        if modern:
            options.update(random_seed=attempts - 1, max_runtime_seconds=max(1, int(deadline - time.monotonic())))
        else:
            pycolmap.set_random_seed(attempts - 1)
        if initial_pair:
            options.update(init_image_id1=initial_pair[0], init_image_id2=initial_pair[1])
            progress(43, 'Trying another pair of overlapping views')
        maps = pycolmap.incremental_mapping(str(database), str(proxies), str(work / f'sparse-{attempts}'), options=options,
            initial_image_pair_callback=check_mapping_time, next_image_callback=check_mapping_time)
        if maps:
            candidate = max(maps.values(), key=lambda item: item.num_reg_images())
            if model is None or candidate.num_reg_images() > model.num_reg_images():
                model = candidate
            if model.num_reg_images() >= required_views:
                break
    if model is None:
        raise CaptureError('Camera positions could not be recovered. Keep the face still and move the phone slowly around it in even light.')
    registered = sorted(model.images.values(), key=lambda image: image.name)
    if len(registered) < required_views:
        raise CaptureError('Too few video views align into one model. Record a slower continuous pass with more overlap.')
    points = np.array([point.xyz for point in model.points3D.values()
                       if point.track.length() >= 3 and point.error < 2.5])
    if len(points) < 100 or not np.isfinite(points).all():
        raise CaptureError('The video has too few reliable 3D matches to reconstruct a surface.')
    error = float(model.compute_mean_reprojection_error())
    if not math.isfinite(error) or error > 2.5:
        raise CaptureError('The recovered camera positions are inconsistent. Keep the subject still while recording.')
    center = np.median(points, axis=0)
    centers = np.array([image.projection_center() for image in registered])
    distance = float(np.median(np.linalg.norm(centers - center, axis=1)))
    if not math.isfinite(distance) or distance < 1e-8:
        raise CaptureError('The video does not establish a usable camera-to-subject distance.')
    spread = np.linalg.norm(centers - np.median(centers, axis=0), axis=1).max() / distance
    if spread < .035:
        raise CaptureError('The camera moved too little for 3D reconstruction. Move around the face rather than filming from one position.')
    # Internal numerical conditioning only. 500 is NOT a measured distance.
    # No metric session is written and output scale remains explicitly unknown.
    scale = 500. / distance
    midpoint = registered[len(registered) // 2].cam_from_world().matrix()
    axes = np.diag([1., -1., -1.]) @ midpoint[:3, :3]
    undistorted = work / 'undistorted'
    undistorted.mkdir()
    photos = []
    for index, image in enumerate(registered):
        camera = model.cameras[image.camera_id]
        if camera.model.name != 'SIMPLE_RADIAL' or not np.isfinite(camera.params).all():
            raise CaptureError('Recovered camera calibration is unsupported.')
        with Image.open(images / image.name) as source:
            pixels = np.asarray(source.convert('RGB'))
        h, w = pixels.shape[:2]
        # COLMAP uses corner-origin pixel coordinates; OpenCV uses pixel centers.
        k = camera.calibration_matrix().copy()
        k[0] *= w / camera.width
        k[1] *= h / camera.height
        k[0, 2] -= .5
        k[1, 2] -= .5
        distortion = np.array([camera.params[3], 0., 0., 0.])
        pixels = cv2.undistort(pixels, k, distortion, None, k)
        Image.fromarray(pixels).save(undistorted / image.name, compress_level=3)
        e = image.cam_from_world().matrix()
        transform = np.eye(4)
        transform[:3, :3] = e[:3, :3] @ axes.T
        transform[:3, 3] = scale * (e[:3, :3] @ center + e[:3, 3])
        photos.append({'name': f'video_{index:04d}', 'color_file': image.name,
            'rgb_width': w, 'rgb_height': h,
            'rgb_intrinsics': {'fx': k[0, 0], 'fy': k[1, 1], 'cx': k[0, 2], 'cy': k[1, 2]},
            'world_to_camera': transform.tolist(), 'calibration_source': 'estimated_from_video'})
    details = {'engine': 'pycolmap', 'version': pycolmap.__version__,
        'initialization_attempts': attempts,
        'registered_frames': len(photos), 'selected_frames': len(paths), 'reliable_points': len(points),
        'mean_reprojection_error_pixels': error, 'camera_spread_relative_to_distance': float(spread),
        'scale_source': 'arbitrary_numerical_normalization', 'metric_scale_available': False,
        'normalization': {'origin': center.tolist(), 'axes': axes.tolist(), 'scale': scale}}
    return Session(undistorted, {'format': 'facemap-estimated-video/1', 'poses': [], 'color_frames': photos}), details


def reconstruct_video(source, output, preset='balanced', progress=None, region=DEFAULT_REGION):
    from .reconstruct import PRESETS, _clean_mesh, _texture, integrate, stereo_depth, stereo_pairs
    from . import __version__
    if preset not in VIDEO_PRESETS:
        raise ValueError('Unknown video reconstruction preset')
    progress = progress or (lambda percent, message: None)
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if source.is_relative_to(output.resolve()):
        raise CaptureError('The result folder must not contain the original video.')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'status': 'processing', 'version': __version__, 'engine': 'facemap-video-cpu',
        'mode': 'video', 'preset': preset, 'capture_kind': 'video', 'display_only': True,
        'measurement_validated': False, 'metric_scale_available': False,
        'glb_units': 'arbitrary', 'ply_units': 'arbitrary', 'surface_file': 'surface.ply',
        'calibration_source': 'estimated_from_video', 'pairs': [],
        'warnings': ['Video reconstruction has no measured scale. Do not use this model for distances or volumes.',
                     'Inspect the surface for gaps, movement artifacts and distortion.']}

    def save_report():
        report['elapsed_seconds'] = round(time.monotonic() - started, 2)
        temp = output / 'report.json.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temp.replace(output / 'report.json')

    save_report()
    try:
        cv2.setNumThreads(min(4, os.cpu_count() or 1))
        with tempfile.TemporaryDirectory(prefix='facemap-import-video-', dir=output) as temp:
            work = Path(temp)
            def video_progress(event):
                if event['phase'] == 'analyzing':
                    progress(3 + int(15 * event['frame'] / max(1, event['total_frames'])), 'Selecting clear video frames')
                else:
                    progress(20, 'Extracting the selected video frames')
            progress(1, 'Checking the original video')
            prepared = prepare_video(source, work / 'video', maximum=VIDEO_PRESETS[preset].frames,
                                     region=region, progress=video_progress, backend='av')
            report['video'] = {key: prepared[key] for key in ('source', 'summary', 'settings', 'source_unchanged')}
            report['warnings'].extend(prepared['warnings'])
            shutil.copyfile(work / 'video' / 'video_input.json', output / 'video-input.json')
            if prepared['status'] != 'ready_for_review':
                raise CaptureError('Too few clear, distinct video frames. Record a slower pass with the face still and well lit.')
            session, cameras = recover_cameras(work / 'video' / 'images', work, VIDEO_PRESETS[preset], region, progress)
            report.update(camera_recovery=cameras, photos=len(session.photos), depth_frames=0)
            save_report()
            settings = PRESETS[preset]
            volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=settings.voxel_mm,
                sdf_trunc=max(6., settings.voxel_mm * 4),
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8, depth_sampling_stride=2)
            pairs = stereo_pairs(session.photos, settings.max_views)
            integrated = 0
            for index, (a, b) in enumerate(pairs):
                progress(55 + int(25 * index / len(pairs)), f'Reconstructing surface view {index + 1} of {len(pairs)}')
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
            if integrated < 3:
                raise CaptureError('Not enough video views support a consistent surface. Try a slower pass in brighter, even lighting.')
            progress(82, 'Building and colouring the 3D model')
            mesh, fraction, smoothed = _clean_mesh(volume.extract_triangle_mesh(), settings)
            model, coverage = _texture(session, mesh, lambda value, message: progress(84 + int((value - 70) * .3), message))
            vertices = np.asarray(mesh.vertices)
            size = float(np.ptp(vertices, axis=0).max())
            if not math.isfinite(size) or size <= 0:
                raise CaptureError('The reconstructed surface is empty or invalid.')
            transform = np.eye(4)
            transform[:3, :3] /= size
            transform[:3, 3] = -vertices.mean(axis=0) / size
            model.apply_transform(transform)
            model.metadata.update(metric_scale_available=False, display_only=True, source='video')
            progress(95, 'Saving the model')
            model.export(output / 'model.glb', file_type='glb')
            mesh.transform(transform)
            if not o3d.io.write_triangle_mesh(str(output / 'surface.ply'), mesh):
                raise RuntimeError('Could not save the reconstructed surface.')
            report.update(status='complete', vertices=len(mesh.vertices), triangles=len(mesh.triangles),
                integrated_views=integrated, texture_coverage=round(coverage, 4),
                largest_component_fraction=round(fraction, 4), native_to_glb=transform.tolist(),
                surface_smoothing={'method': 'taubin', 'iterations': 8, 'applied': smoothed})
            if coverage < .7:
                report['warnings'].append('Some areas have no reliable photograph and are shown in grey.')
            save_report()
        progress(100, 'Model saved. Open the viewer to inspect it.')
        return report
    except BaseException as error:
        report.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        for name in ('model.glb', 'surface.ply'):
            (output / name).unlink(missing_ok=True)
        save_report()
        raise
