"""Generated calibration fixture. Contains no patient imagery or identity."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def make_plane(directory, *, vertical=False, with_depth=True, blank=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Integer ground-truth disparity isolates transform/unit correctness from
    # SGBM's subpixel quantisation on resampled random textures.
    width, height, focal, distance = 320, 240, 300., 500.
    rng = np.random.default_rng(7401)
    texture = cv2.GaussianBlur(rng.integers(30, 235, (800, 800, 3), dtype=np.uint8), (3, 3), .7)
    if blank:
        texture[:] = 140
    y, x = np.indices((height, width), dtype=np.float32)
    poses, colors = [], []
    for i, shift in enumerate((-40., 0., 40.)):
        cx, cy = (0., shift) if vertical else (shift, 0.)
        wx = (x - (width - 1) / 2) * distance / focal + cx
        wy = (y - (height - 1) / 2) * distance / focal + cy
        rgb = cv2.remap(texture, wx + 400, wy + 400, cv2.INTER_LINEAR)
        rgb_name = f'photo_{i}.jpg'
        Image.fromarray(rgb).save(directory / rgb_name, quality=98)
        e = np.eye(4)
        e[:3, 3] = [-cx, -cy, distance]
        k = {'fx': focal, 'fy': focal, 'cx': (width - 1) / 2, 'cy': (height - 1) / 2}
        color = {'name': f'photo_{i}', 'color_file': rgb_name, 'rgb_width': width, 'rgb_height': height,
                 'rgb_intrinsics': k, 'world_to_camera': e.tolist()}
        colors.append(color)
        if with_depth:
            # Separate photo names avoid duplicate manifest references.
            depth_name = f'depth_{i}.bin'
            np.full((height, width), distance, dtype='<f4').tofile(directory / depth_name)
            poses.append({'name': f'depth_{i}', 'depth_file': depth_name, 'width': width, 'height': height,
                          'intrinsics': k, 'world_to_camera': e.tolist(), 'depth_unit_mm': 1.})
    manifest = {'format': 'vectra-dupe-session/1', 'state': 'saved', 'is_demo': True, 'poses': poses, 'color_frames': colors}
    (directory / 'session.json').write_text(json.dumps(manifest), encoding='utf-8')
    return directory
