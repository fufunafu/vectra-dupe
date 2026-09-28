"""Deterministic camera/depth fixture. Contains no person or patient data."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .rgbd import FORMAT


def make_rgbd_plane(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    angles = ([(y, 0) for y in range(0, -41, -5)] + [(y, 0) for y in range(-35, 41, 5)]
              + [(y, 0) for y in range(35, -1, -5)] + [(0, p) for p in range(5, 21, 5)]
              + [(0, p) for p in range(15, -21, -5)])
    w, h, dw, dh, focal = 640, 480, 160, 120, 620.
    k = {'fx': focal, 'fy': focal, 'cx': w/2, 'cy': h/2}
    kd = {key: value/4 for key, value in k.items()}
    rng = np.random.default_rng(8642)
    texture = cv2.GaussianBlur(rng.integers(30, 235, (1400, 1400, 3), dtype=np.uint8), (3, 3), .6)
    poses, timeline = [], []
    for i, (yaw, pitch) in enumerate(angles):
        yaw, pitch = np.radians([yaw, pitch])
        c = 500*np.array([-np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
        z = -c/np.linalg.norm(c); x = np.cross(z, [0., 1., 0.]); x /= np.linalg.norm(x); y = np.cross(z, x)
        e = np.eye(4); e[:3, :3] = [x, y, z]; e[:3, 3] = -e[:3, :3] @ c
        def rays(width, height, intrinsics):
            v, u = np.indices((height, width), dtype=float)
            ray = np.stack(((u-intrinsics['cx'])/intrinsics['fx'], (v-intrinsics['cy'])/intrinsics['fy'], np.ones_like(u)), axis=-1) @ e[:3, :3]
            depth = -c[2]/ray[..., 2]
            return depth.astype('<f4'), c+ray*depth[..., None]
        _, world = rays(w, h, k)
        photo = cv2.remap(texture, (world[..., 0]*2+700).astype('float32'), (world[..., 1]*2+700).astype('float32'), cv2.INTER_LINEAR)
        rgb_name, depth_name, confidence_name = f'frame_{i}.jpg', f'depth_{i}.bin', f'confidence_{i}.bin'
        Image.fromarray(photo).save(directory/rgb_name, quality=98)
        depth, _ = rays(dw, dh, kd); depth.tofile(directory/depth_name)
        np.full((dh, dw), 2, dtype=np.uint8).tofile(directory/confidence_name)
        poses.append({'name': f'frame_{i}', 'depth_file': depth_name, 'width': dw, 'height': dh,
            'intrinsics': kd, 'world_to_camera': e.tolist(), 'depth_unit_mm': 1,
            'color_file': rgb_name, 'rgb_width': w, 'rgb_height': h, 'rgb_intrinsics': k,
            'rgb_world_to_camera': e.tolist(), 'confidence_file': confidence_name,
            'frame_timestamp_seconds': i*.4, 'tracking_state': 'normal',
            'quality': {'accepted_frames': 1, 'valid_depth_fraction': 1},
            'photo_quality': {'sharpness': 50, 'mean_luma': 128, 'clipped_fraction': 0,
                'region': {'x': .1, 'y': .1, 'width': .8, 'height': .8}}})
        timeline.append({'timestamp_seconds': i*.4, 'tracking_state': 'normal', 'has_depth': True,
            'rgb_width': w, 'rgb_height': h, 'intrinsics': k, 'camera_to_world_ar_m': np.eye(4).tolist()})
    meta = {'format': FORMAT, 'capture_kind': 'video_depth', 'state': 'saved', 'is_demo': False,
        'poses': poses, 'color_frames': [], 'rgbd': {'depth_source': 'arkit-sceneDepth', 'depth_association': 'same-arframe',
            'timeline_file': 'camera-frames.jsonl', 'duration_seconds': len(poses)*.4,
            'video_frames_written': len(poses), 'video_frames_dropped': 0, 'color_width': w, 'color_height': h}}
    (directory/'session.json').write_text(json.dumps(meta))
    (directory/'camera-frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in timeline))
    return directory

