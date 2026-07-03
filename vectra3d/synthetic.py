"""Synthetic head with a known-volume "filler" bump for ground-truth testing.

The head is a deformed ellipsoid with nose/brow/chin features so that ICP has
geometry to lock onto (a plain sphere is rotationally symmetric and would make
registration degenerate). The face looks along +z.
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class BumpSpec:
    center: np.ndarray  # point on the head surface, before-frame coords
    amplitude_mm: float
    sigma_mm: float


def _displace(mesh: o3d.geometry.TriangleMesh, center: np.ndarray,
              amplitude: float, sigma: float) -> None:
    """Displace vertices along their normals by a Gaussian falloff bump."""
    mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices)
    n = np.asarray(mesh.vertex_normals)
    d = np.linalg.norm(v - center, axis=1)
    disp = amplitude * np.exp(-(d ** 2) / (2 * sigma ** 2))
    mesh.vertices = o3d.utility.Vector3dVector(v + n * disp[:, None])


def _surface_point(mesh: o3d.geometry.TriangleMesh, direction: np.ndarray) -> np.ndarray:
    """Point on the mesh surface in the given direction from the origin."""
    direction = direction / np.linalg.norm(direction)
    v = np.asarray(mesh.vertices)
    idx = np.argmax(v @ direction / np.linalg.norm(v, axis=1).clip(min=1e-9))
    return v[idx].copy()


def make_head(resolution: int = 160) -> o3d.geometry.TriangleMesh:
    head = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    head.scale(1.0, center=np.zeros(3))
    v = np.asarray(head.vertices)
    v[:, 0] *= 75.0   # half-width
    v[:, 1] *= 105.0  # half-height
    v[:, 2] *= 85.0   # half-depth
    head.vertices = o3d.utility.Vector3dVector(v)

    # Facial features (all on the +z side) so the surface is asymmetric.
    _displace(head, _surface_point(head, np.array([0.0, -0.1, 1.0])), 14.0, 18.0)   # nose
    _displace(head, _surface_point(head, np.array([0.0, 0.35, 1.0])), 5.0, 25.0)    # brow
    _displace(head, _surface_point(head, np.array([0.0, -0.75, 0.8])), 6.0, 16.0)   # chin
    _displace(head, _surface_point(head, np.array([0.45, -0.25, 0.85])), 3.0, 20.0) # cheekbone R
    _displace(head, _surface_point(head, np.array([-0.45, -0.25, 0.85])), 3.0, 20.0)
    head.compute_vertex_normals()
    return head


def cheek_bump_spec(head: o3d.geometry.TriangleMesh,
                    amplitude_mm: float = 2.0, sigma_mm: float = 8.0) -> BumpSpec:
    """A filler-like bump on the left cheek (subject's left, +x)."""
    center = _surface_point(head, np.array([0.5, -0.35, 0.8]))
    return BumpSpec(center=center, amplitude_mm=amplitude_mm, sigma_mm=sigma_mm)


def apply_bump(head: o3d.geometry.TriangleMesh, bump: BumpSpec) -> o3d.geometry.TriangleMesh:
    out = o3d.geometry.TriangleMesh(head)
    _displace(out, bump.center, bump.amplitude_mm, bump.sigma_mm)
    out.compute_vertex_normals()
    return out


def closed_mesh_volume(mesh: o3d.geometry.TriangleMesh) -> float:
    """Exact volume of a watertight mesh via the divergence theorem (mm^3)."""
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))


def random_session_transform(rng: np.random.Generator,
                             rot_deg: float = 5.0, trans_mm: float = 10.0) -> np.ndarray:
    """How differently the subject sits in front of the camera between sessions."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    t = np.eye(4)
    t[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(
        axis * np.deg2rad(rot_deg) * rng.uniform(0.5, 1.0))
    t[:3, 3] = rng.normal(scale=trans_mm / np.sqrt(3), size=3)
    return t
