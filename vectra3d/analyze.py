"""Signed surface distance, ROI volume integration, and heatmap output.

For every vertex of the before-mesh we cast a ray along its normal into the
aligned after-mesh; the signed hit distance is the local surface change.
Volume = sum(distance_i * vertex_area_i) over the region of interest.
"""

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

RAY_BACKOFF_MM = 15.0  # start rays behind the surface so near hits register
MAX_ABS_DIST_MM = 10.0  # reject hits further than any plausible soft-tissue change


@dataclass
class DistanceField:
    vertices: np.ndarray
    distances: np.ndarray  # signed mm, NaN where no valid hit
    vertex_areas: np.ndarray  # mm^2


def adjacency(mesh: o3d.geometry.TriangleMesh):
    import scipy.sparse as sp
    t = np.asarray(mesh.triangles)
    n = len(mesh.vertices)
    i = np.concatenate([t[:, 0], t[:, 1], t[:, 2], t[:, 1], t[:, 2], t[:, 0]])
    j = np.concatenate([t[:, 1], t[:, 2], t[:, 0], t[:, 0], t[:, 1], t[:, 2]])
    a = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    a.data[:] = 1.0
    return a


def boundary_band_mask(mesh: o3d.geometry.TriangleMesh, rings: int = 6) -> np.ndarray:
    """Vertices within `rings` graph hops of the open scan boundary.

    Boundary areas are covered by a single view and carry the worst sensor
    error, so they are excluded from analysis.
    """
    t = np.asarray(mesh.triangles)
    edges = np.sort(np.concatenate(
        [t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, idx, counts = np.unique(edges, axis=0, return_index=True, return_counts=True)
    boundary_edges = edges[idx[counts == 1]]
    band = np.zeros(len(mesh.vertices), dtype=bool)
    band[np.unique(boundary_edges)] = True
    adj = adjacency(mesh)
    for _ in range(rings):
        band = band | (adj @ band > 0)
    return band


def vertex_areas(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    cross = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    tri_area = 0.5 * np.linalg.norm(cross, axis=1)
    areas = np.zeros(len(v))
    for k in range(3):
        np.add.at(areas, t[:, k], tri_area / 3.0)
    return areas


def signed_distance_field(before: o3d.geometry.TriangleMesh,
                          after_aligned: o3d.geometry.TriangleMesh) -> DistanceField:
    before.compute_vertex_normals()
    v = np.asarray(before.vertices)
    n = np.asarray(before.vertex_normals)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(after_aligned))
    origins = v - n * RAY_BACKOFF_MM
    rays = o3d.core.Tensor(
        np.hstack([origins, n]).astype(np.float32))
    t_hit = scene.cast_rays(rays)["t_hit"].numpy()

    dist = t_hit - RAY_BACKOFF_MM
    dist[~np.isfinite(t_hit)] = np.nan
    dist[np.abs(dist) > MAX_ABS_DIST_MM] = np.nan
    dist[boundary_band_mask(before)] = np.nan
    return DistanceField(vertices=v, distances=dist, vertex_areas=vertex_areas(before))


def subtract_bias_field(field: DistanceField,
                        exclude_center: np.ndarray | None = None,
                        exclude_radius_mm: float = 0.0,
                        exclude_mask: np.ndarray | None = None) -> DistanceField:
    """Remove the smooth systematic component of the distance field.

    Residual registration error and low-frequency sensor error both show up as
    a smooth field over the face, which integrates into phantom milliliters.
    Estimate that field by Gaussian-kernel averaging (Nadaraya-Watson) of the
    distances on STABLE vertices only — the treated ROI is excluded and the
    estimate is interpolated across it — with outlier trimming, then subtract.
    Kernel averaging is a convex combination of observed values, so unlike an
    RBF least-squares fit it cannot oscillate when bridging the ROI hole.
    """
    v, d = field.vertices, field.distances
    stable = np.isfinite(d)
    if exclude_center is not None:
        stable &= np.linalg.norm(v - exclude_center, axis=1) > exclude_radius_mm
    if exclude_mask is not None:
        stable &= ~exclude_mask

    bandwidth = 25.0  # mm; must stay wider than any plausible treatment effect
    sample_idx = np.flatnonzero(stable)[::  max(1, int(stable.sum()) // 3000)]

    def kernel_average(targets: np.ndarray, src_pts: np.ndarray,
                       src_val: np.ndarray) -> np.ndarray:
        out = np.empty(len(targets))
        for start in range(0, len(targets), 8192):
            block = targets[start:start + 8192]
            sq = ((block[:, None, :] - src_pts[None, :, :]) ** 2).sum(axis=2)
            w = np.exp(-sq / (2 * bandwidth ** 2))
            out[start:start + 8192] = (w @ src_val) / w.sum(axis=1).clip(min=1e-12)
        return out

    keep = np.ones(len(sample_idx), dtype=bool)
    bias = None
    for _ in range(2):
        src = sample_idx[keep]
        bias = kernel_average(v, v[src], d[src])
        residual_at_samples = d[sample_idx] - bias[sample_idx]
        keep = np.abs(residual_at_samples) < 2.5 * np.std(residual_at_samples)

    corrected = DistanceField(
        vertices=v, distances=d - bias, vertex_areas=field.vertex_areas)
    return corrected


def roi_volume_mm3(field: DistanceField, center: np.ndarray, radius_mm: float) -> float:
    in_roi = np.linalg.norm(field.vertices - center, axis=1) <= radius_mm
    valid = in_roi & np.isfinite(field.distances)
    return float(np.sum(field.distances[valid] * field.vertex_areas[valid]))


def rms_distance_mm(field: DistanceField, exclude_center=None,
                    exclude_radius_mm: float = 0.0) -> float:
    valid = np.isfinite(field.distances)
    if exclude_center is not None:
        valid &= np.linalg.norm(field.vertices - exclude_center, axis=1) > exclude_radius_mm
    return float(np.sqrt(np.mean(field.distances[valid] ** 2)))


def save_colored_mesh(before: o3d.geometry.TriangleMesh, field: DistanceField,
                      path: str, vmax_mm: float = 2.5) -> None:
    cmap = plt.get_cmap("RdBu_r")
    norm = np.clip((np.nan_to_num(field.distances) + vmax_mm) / (2 * vmax_mm), 0, 1)
    colors = cmap(norm)[:, :3]
    colors[~np.isfinite(field.distances)] = 0.55
    out = o3d.geometry.TriangleMesh(before)
    out.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(path, out)


def save_heatmap_png(field: DistanceField, path: str, title: str,
                     vmax_mm: float = 2.5) -> None:
    """Front-view (x up-screen-right, y up) orthographic heatmap."""
    front = field.vertices[:, 2] > 0  # face looks along +z
    v = field.vertices[front]
    d = field.distances[front]
    fig, ax = plt.subplots(figsize=(7, 8))
    sc = ax.scatter(v[:, 0], v[:, 1], c=d, cmap="RdBu_r",
                    vmin=-vmax_mm, vmax=vmax_mm, s=2, linewidths=0)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(sc, ax=ax, label="surface change (mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
