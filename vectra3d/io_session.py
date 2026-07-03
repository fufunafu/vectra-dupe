"""On-disk capture session format — the contract between capture clients
(iOS app, simulator) and the processing pipeline.

A session directory contains `session.json` plus one depth file per pose:

    session.json
    depth_front.bin     raw little-endian float32, row-major, millimeters,
    depth_left.bin      0 = invalid pixel
    depth_right.bin
    color_front.jpg     (optional, for texturing later)

`world_to_camera` is a 4x4 OpenCV-convention extrinsic in millimeters.
`.npy` depth files are accepted as well as `.bin`.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from PIL import Image

FORMAT_ID = "vectra-dupe-session/1"
# Guided DEPTH keyframes: front + a 3/4 and a near-profile each side, plus
# brow/jaw (pitched up/down) and a near-ear view each side for jawline and
# under-chin coverage. `front` must stay first — the pipeline anchors the world
# frame on poses[0]. These drive the metric geometry (TSDF + bounded ICP), so
# the count stays modest. (Only the synthetic capture in capture.py relies on
# this tuple; load_session reads whatever poses a real session.json lists.)
POSE_NAMES = ("front", "left_half", "left", "right_half", "right",
              "brow", "jaw", "ear_left", "ear_right")


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def to_o3d(self, width: int, height: int) -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(
            width, height, self.fx, self.fy, self.cx, self.cy)


@dataclass
class PoseCapture:
    name: str
    depth: np.ndarray  # float32 (h, w), mm, 0 = invalid
    intrinsics: Intrinsics
    world_to_camera: np.ndarray  # 4x4
    # Optional color photo for texturing: raw RGB (camera-native landscape) plus
    # its own intrinsics (the RGB image is usually a different resolution than
    # the depth map, so it must be resampled onto the depth grid before use).
    color: np.ndarray | None = None        # uint8 (H, W, 3)
    rgb_intrinsics: Intrinsics | None = None


@dataclass
class ColorFrame:
    """A colour-only capture frame (no depth): one of the many quality-gated
    photos auto-harvested over a free orbit. It carries a pose so it can be
    projected onto the fused mesh for texture, but never enters TSDF or ICP —
    so the dense RGB set can be large without slowing the (bounded) geometry
    path. Mirrors PoseCapture's `color`/`rgb_intrinsics` so the texture builders
    can iterate poses and colour frames through the same code path."""
    name: str
    color: np.ndarray              # uint8 (H, W, 3)
    rgb_intrinsics: Intrinsics
    world_to_camera: np.ndarray    # 4x4, raw ARKit pose (no ICP refinement)


def _rgb_entry(name: str, color: np.ndarray, rgb: Intrinsics,
               world_to_camera: np.ndarray) -> tuple[dict, str]:
    """JSON metadata + filename for one colour image. Shared by poses (optional
    colour) and colour frames."""
    color_file = f"color_{name}.jpg"
    return {
        "color_file": color_file,
        "rgb_width": int(color.shape[1]),
        "rgb_height": int(color.shape[0]),
        "rgb_intrinsics": {"fx": rgb.fx, "fy": rgb.fy, "cx": rgb.cx, "cy": rgb.cy},
        "world_to_camera": np.asarray(world_to_camera).tolist(),
    }, color_file


def write_session(session_dir: str, poses: list[PoseCapture], label: str,
                  device: str = "simulator", captured_at: str = "",
                  color_frames: list[ColorFrame] | None = None) -> None:
    os.makedirs(session_dir, exist_ok=True)
    meta = {"format": FORMAT_ID, "label": label, "device": device,
            "captured_at": captured_at, "poses": []}
    for pose in poses:
        depth_file = f"depth_{pose.name}.bin"
        pose.depth.astype("<f4").tofile(os.path.join(session_dir, depth_file))
        entry = {
            "name": pose.name,
            "depth_file": depth_file,
            "width": int(pose.depth.shape[1]),
            "height": int(pose.depth.shape[0]),
            "intrinsics": {"fx": pose.intrinsics.fx, "fy": pose.intrinsics.fy,
                           "cx": pose.intrinsics.cx, "cy": pose.intrinsics.cy},
            "world_to_camera": np.asarray(pose.world_to_camera).tolist(),
            "depth_unit_mm": 1.0,
        }
        if pose.color is not None and pose.rgb_intrinsics is not None:
            rgb_meta, cfile = _rgb_entry(pose.name, pose.color,
                                         pose.rgb_intrinsics, pose.world_to_camera)
            Image.fromarray(pose.color).save(os.path.join(session_dir, cfile),
                                             quality=90)
            entry["color_file"] = cfile
            entry["rgb_intrinsics"] = rgb_meta["rgb_intrinsics"]
        meta["poses"].append(entry)
    for cf in (color_frames or []):
        rgb_meta, cfile = _rgb_entry(cf.name, cf.color, cf.rgb_intrinsics,
                                     cf.world_to_camera)
        Image.fromarray(cf.color).save(os.path.join(session_dir, cfile), quality=90)
        meta.setdefault("color_frames", []).append({"name": cf.name, **rgb_meta})
    with open(os.path.join(session_dir, "session.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _load_depth(path: str, width: int, height: int, unit_mm: float) -> np.ndarray:
    if path.endswith(".npy"):
        depth = np.load(path)
    else:
        depth = np.fromfile(path, dtype="<f4").reshape(height, width)
    depth = depth.astype(np.float32)
    if unit_mm != 1.0:
        depth = depth * unit_mm
    return depth


def _load_color(session_dir: str, entry: dict) -> tuple[np.ndarray | None,
                                                        Intrinsics | None]:
    """Load the optional colour image + its intrinsics for a pose/colour-frame
    entry. Returns (None, None) when absent or missing on disk."""
    cfile = entry.get("color_file")
    rk = entry.get("rgb_intrinsics")
    if not (cfile and rk):
        return None, None
    cpath = os.path.join(session_dir, cfile)
    if not os.path.exists(cpath):
        return None, None
    color = np.asarray(Image.open(cpath).convert("RGB"))
    return color, Intrinsics(rk["fx"], rk["fy"], rk["cx"], rk["cy"])


def load_session(session_dir: str) -> tuple[list[PoseCapture], list[ColorFrame], dict]:
    """Load a capture session.

    Returns (depth_poses, color_frames, meta). `color_frames` is the optional
    top-level "color_frames" list — the dense, depth-less RGB harvested over the
    orbit — and is empty for older 5-pose sessions, in which case the pipeline
    behaves exactly as before.
    """
    with open(os.path.join(session_dir, "session.json")) as f:
        meta = json.load(f)
    if meta.get("format") != FORMAT_ID:
        raise ValueError(f"unknown session format: {meta.get('format')!r}")
    poses = []
    for p in meta["poses"]:
        depth = _load_depth(os.path.join(session_dir, p["depth_file"]),
                            p["width"], p["height"], p.get("depth_unit_mm", 1.0))
        k = p["intrinsics"]
        color, rgb_intr = _load_color(session_dir, p)
        poses.append(PoseCapture(
            name=p["name"], depth=depth,
            intrinsics=Intrinsics(k["fx"], k["fy"], k["cx"], k["cy"]),
            world_to_camera=np.array(p["world_to_camera"], dtype=np.float64),
            color=color, rgb_intrinsics=rgb_intr))
    color_frames = []
    for c in meta.get("color_frames", []):
        color, rgb_intr = _load_color(session_dir, c)
        if color is None or rgb_intr is None:
            continue
        color_frames.append(ColorFrame(
            name=c["name"], color=color, rgb_intrinsics=rgb_intr,
            world_to_camera=np.array(c["world_to_camera"], dtype=np.float64)))
    return poses, color_frames, meta
