"""Simulated depth capture: render z-depth maps of a mesh and add sensor noise.

Noise model for TrueDepth-class sensors, three components:
- white per-pixel noise (averages out over frames),
- a DEVICE-FIXED low-frequency calibration error field (same in every capture
  on the same phone — cancels in before/after differencing),
- a smaller per-capture low-frequency field (temperature, distance, pose
  dependence — this is what limits the differential noise floor).
"""

import numpy as np
import open3d as o3d
from scipy.ndimage import zoom

from . import cameras
from .io_session import POSE_NAMES, ColorFrame, Intrinsics, PoseCapture

WHITE_SIGMA_MM = 0.8
DEVICE_FIXED_SIGMA_MM = 0.5
PER_CAPTURE_SIGMA_MM = 0.25

# Off-axis (yaw, pitch) views for the synthetic dense RGB harvest. These mirror
# the iOS free-orbit capture: colour-only frames at wide angles the depth
# keyframes don't cover, used to exercise the (depth-decoupled) texture path.
COLOR_FRAME_POSES_DEG = (
    (-50.0, 15.0), (50.0, 15.0), (-50.0, -15.0), (50.0, -15.0),
    (-20.0, 28.0), (20.0, 28.0), (0.0, -28.0),
)
SKIN_RGB = (228, 188, 168)


def _lowfreq_field(rng: np.random.Generator, sigma_mm: float) -> np.ndarray:
    coarse = rng.normal(scale=sigma_mm, size=(6, 8))
    field = zoom(coarse, (cameras.HEIGHT / 6, cameras.WIDTH / 8), order=3)
    return field[: cameras.HEIGHT, : cameras.WIDTH]


class Sensor:
    """One physical depth sensor with its fixed calibration error."""

    def __init__(self, rng: np.random.Generator):
        self.fixed_field = _lowfreq_field(rng, DEVICE_FIXED_SIGMA_MM)


def render_depth(mesh: o3d.geometry.TriangleMesh, extrinsic: np.ndarray) -> np.ndarray:
    """Render a z-depth map (float32 mm, 0 = no hit) from one camera pose."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = scene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(cameras.intrinsic_matrix()),
        extrinsic_matrix=o3d.core.Tensor(extrinsic),
        width_px=cameras.WIDTH, height_px=cameras.HEIGHT)
    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()
    rays_np = rays.numpy()
    hit = np.isfinite(t_hit)

    # Hit points in world coords -> camera frame -> z-depth.
    origins, dirs = rays_np[..., :3], rays_np[..., 3:]
    points = origins + dirs * np.where(hit, t_hit, 0.0)[..., None]
    r, t = extrinsic[:3, :3], extrinsic[:3, 3]
    z = (points @ r.T + t)[..., 2]
    depth = np.where(hit, z, 0.0).astype(np.float32)
    return depth


def render_color(mesh: o3d.geometry.TriangleMesh, extrinsic: np.ndarray,
                 skin: tuple[int, int, int] = SKIN_RGB) -> np.ndarray:
    """Render a plausible RGB photo (uint8 H,W,3) from one camera pose by
    Lambert-shading the surface normals. Stands in for a real iPhone photo so
    the synthetic test can exercise the colour-frame texture path; not used by
    the real pipeline, which gets actual JPEGs."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = scene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(cameras.intrinsic_matrix()),
        extrinsic_matrix=o3d.core.Tensor(extrinsic),
        width_px=cameras.WIDTH, height_px=cameras.HEIGHT)
    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()
    nrm = ans["primitive_normals"].numpy()
    rays_np = rays.numpy()
    hit = np.isfinite(t_hit)
    origins, dirs = rays_np[..., :3], rays_np[..., 3:]
    points = origins + dirs * np.where(hit, t_hit, 0.0)[..., None]
    cam_center = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
    view = cam_center[None, None, :] - points
    view /= np.linalg.norm(view, axis=-1, keepdims=True) + 1e-9
    shade = np.clip(np.abs(np.sum(nrm * view, axis=-1)), 0.0, 1.0)  # (H, W)
    img = (shade[..., None] * np.array(skin, dtype=np.float64)).astype(np.uint8)
    img[~hit] = 0
    return img


def capture_color_frames(mesh: o3d.geometry.TriangleMesh) -> list[ColorFrame]:
    """Synthetic dense RGB harvest: colour-only frames at off-axis angles,
    carrying their (clean) poses. Mirrors the iOS free-orbit capture."""
    intr = Intrinsics(cameras.FX, cameras.FY, cameras.CX, cameras.CY)
    frames = []
    for i, (yaw, pitch) in enumerate(COLOR_FRAME_POSES_DEG):
        ext = cameras.orbit_extrinsic(yaw, pitch)
        frames.append(ColorFrame(
            name=f"c{i:03d}", color=render_color(mesh, ext),
            rgb_intrinsics=intr, world_to_camera=ext))
    return frames


def capture_session(mesh: o3d.geometry.TriangleMesh, sensor: Sensor,
                    rng: np.random.Generator,
                    pose_drift: bool = True, frames_per_pose: int = 8):
    """Capture the guided depth keyframes of one session (see POSE_NAMES).

    Per pose the sensor delivers a short burst of frames; averaging them cuts
    the white noise by sqrt(frames). The device-fixed and per-capture
    low-frequency fields are applied once per pose. Returns PoseCapture
    objects whose world_to_camera matrices carry simulated ARKit pose drift
    on the side views.
    """
    true_ext = cameras.capture_extrinsics()
    intr = Intrinsics(cameras.FX, cameras.FY, cameras.CX, cameras.CY)
    poses = []
    for name, e in zip(POSE_NAMES, true_ext):
        clean = render_depth(mesh, e)
        valid = clean > 0
        depth = clean.copy()
        white = rng.normal(scale=WHITE_SIGMA_MM / np.sqrt(frames_per_pose),
                           size=int(valid.sum()))
        lowfreq = sensor.fixed_field + _lowfreq_field(rng, PER_CAPTURE_SIGMA_MM)
        depth[valid] += (white + lowfreq[valid]).astype(np.float32)
        assumed = e if (name == "front" or not pose_drift) \
            else cameras.perturb_extrinsic(e, rng)
        poses.append(PoseCapture(name=name, depth=depth, intrinsics=intr,
                                 world_to_camera=assumed))
    return poses
