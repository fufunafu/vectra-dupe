"""Native 4K render of a textured ellipsoid for video pipeline verification.

This is a generated geometric target, not a face or a reconstruction-quality claim.
No camera poses or calibration are supplied to the reconstruction engine.
"""

from pathlib import Path
import av
import cv2
import numpy as np


def make_video(path, *, frames=36, width=3840, height=2160, blank=False):
    path = Path(path)
    if path.exists():
        raise ValueError('The test video already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7109)
    texture = rng.integers(35, 230, (512, 1024, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), 1.)
    texture = cv2.resize(texture, (2048, 1024), interpolation=cv2.INTER_CUBIC)
    if blank:
        texture[:] = 140
    radii = np.array([110., 140., 90.], dtype=np.float32)
    focal = width * 1.05
    x = (np.arange(width, dtype=np.float32) - (width - 1) / 2) / focal
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
        stream.options = {'crf': '16', 'preset': 'ultrafast', 'threads': '2'}
        for index, angle in enumerate(np.linspace(-38, 38, frames)):
            radians = np.deg2rad(angle)
            camera = np.array([600 * np.sin(radians), 18 * np.sin(radians * 2), 600 * np.cos(radians)], dtype=np.float32)
            forward = -camera / np.linalg.norm(camera)
            right = np.cross(forward, [0., 1., 0.]).astype(np.float32)
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            output = np.full((height, width, 3), 105, dtype=np.uint8)
            origin = camera / radii
            for start in range(0, height, 64):
                end = min(start + 64, height)
                y = (np.arange(start, end, dtype=np.float32) - (height - 1) / 2) / focal
                directions = forward + x[None, :, None] * right + y[:, None, None] * down
                ray = directions / radii
                a = np.sum(ray * ray, axis=-1)
                b = 2 * np.sum(ray * origin, axis=-1)
                c = np.dot(origin, origin) - 1
                disc = b * b - 4 * a * c
                visible = disc > 0
                distance = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
                point = camera + distance[..., None] * directions
                unit = point / radii
                u = ((np.arctan2(unit[..., 0], unit[..., 2]) / (2 * np.pi) + .5) * (texture.shape[1] - 1)).astype(np.float32)
                v = ((.5 - np.arcsin(np.clip(unit[..., 1], -1, 1)) / np.pi) * (texture.shape[0] - 1)).astype(np.float32)
                pixels = cv2.remap(texture, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                normal = point / (radii * radii)
                normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
                light = np.maximum(.55, .75 + .25 * normal[..., 2])
                pixels = np.clip(pixels * light[..., None], 0, 255).astype(np.uint8)
                output[start:end][visible] = pixels[visible]
            frame = av.VideoFrame.from_ndarray(output, format='rgb24')
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
