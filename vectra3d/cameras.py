"""Camera model for the simulated depth keyframe capture: front, a 3/4 view and
a near-profile (~72 deg) on each side, plus pitched brow/jaw and near-ear views.

The wide near-profile views model the phone-orbit capture flow (subject holds
the head still, the phone swings round to the side) which is the only way to
reach a true side profile — ARKit's face anchor drops tracking past ~±40 deg of
head turn, so a head-turn capture can never see the nose in profile.

Conventions: OpenCV pinhole camera (x right, y down, z forward).
`extrinsic` is world-to-camera, as expected by Open3D's TSDF/RGBD utilities.
"""

import numpy as np
import open3d as o3d

# Approximates the iPhone TrueDepth depth stream (640x480).
WIDTH = 640
HEIGHT = 480
FX = FY = 580.0
CX = (WIDTH - 1) / 2.0
CY = (HEIGHT - 1) / 2.0

CAPTURE_DISTANCE = 350.0  # mm, middle of the TrueDepth working range
# Expanded depth keyframe set, as (yaw_deg, pitch_deg) of the camera orbiting
# the head. The 3/4 views bridge the front and the profile so ICP can chain
# across the gap (a 72 deg profile shares almost no surface with the front view
# alone). brow/jaw add pitched coverage of the forehead and under-chin; the
# near-ear views (±80, not 90, where TrueDepth grazes) reach the jawline. Order
# must match io_session.POSE_NAMES; front stays first (the world anchor).
VIEW_POSES_DEG = (
    (0.0, 0.0),     # front
    (-35.0, 0.0),   # left_half
    (-72.0, 0.0),   # left
    (35.0, 0.0),    # right_half
    (72.0, 0.0),    # right
    (0.0, 30.0),    # brow  — camera above, looking down at the forehead
    (0.0, -30.0),   # jaw   — camera below, looking up under the chin
    (-80.0, 0.0),   # ear_left
    (80.0, 0.0),    # ear_right
)


def intrinsic_o3d() -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, FX, FY, CX, CY)


def intrinsic_matrix() -> np.ndarray:
    return np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)


def look_at_extrinsic(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-to-camera matrix for a camera at `cam_pos` looking at `target`."""
    up = np.array([0.0, 1.0, 0.0])
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(-up, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    cam_to_world = np.eye(4)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = down
    cam_to_world[:3, 2] = forward
    cam_to_world[:3, 3] = cam_pos
    return np.linalg.inv(cam_to_world)


def orbit_extrinsic(yaw_deg: float, pitch_deg: float,
                    distance: float = CAPTURE_DISTANCE) -> np.ndarray:
    """World-to-camera for a camera orbiting the head (at the origin) at the
    given yaw/pitch and looking at it. The face looks along +z, so cameras sit
    on a +z hemisphere; yaw sweeps around the vertical axis, pitch lifts above
    (+) or below (-) the eye line."""
    yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    pos = distance * np.array([
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
        np.cos(pitch) * np.cos(yaw)])
    return look_at_extrinsic(pos, np.zeros(3))


def capture_extrinsics() -> list[np.ndarray]:
    """Extrinsics for the depth keyframe poses, orbiting the head at the origin."""
    return [orbit_extrinsic(yaw, pitch) for yaw, pitch in VIEW_POSES_DEG]


def perturb_extrinsic(
    extrinsic: np.ndarray, rng: np.random.Generator,
    rot_deg: float = 1.0, trans_mm: float = 2.0,
) -> np.ndarray:
    """Simulate ARKit pose drift: a small random rigid error on a view pose."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rot_deg) * rng.uniform(0.5, 1.0)
    delta = np.eye(4)
    delta[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    delta[:3, 3] = rng.normal(scale=trans_mm / np.sqrt(3), size=3)
    return delta @ extrinsic
