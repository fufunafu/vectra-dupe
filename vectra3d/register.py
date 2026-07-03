"""Cross-session registration: align an "after" scan onto a "before" scan.

Critical rule (same as the real VECTRA workflow): the final alignment must use
only STABLE regions. If the treated area is allowed to participate, ICP will
absorb part of the volume change into the pose and the measurement shrinks.

Geometry + Texture refinement (Artec "Geometry + Texture" Global Registration
analog): faces are smooth, so geometry-only ICP can slide tangentially over the
stable region and lock in a slightly wrong pose — quietly absorbing real volume
change into the alignment. After the geometry fine pass nails the surface, an
optional colored-ICP pass adds a photometric term that pins the alignment on
skin texture (freckles, brow, lip/nostril edges). It is applied only as a
*bounded refinement on top of* the validated geometry solution: if colour is
unusable, the colored solve fails, or it drifts farther than a real residual
could, we keep the geometry-only result. Toggle with VECTRA_COLORED_REG=0.
"""

import os

import numpy as np
import open3d as o3d

USE_COLORED_REG = os.environ.get("VECTRA_COLORED_REG", "1") != "0"
# A colored refinement sits on top of an already-converged geometry alignment,
# so the residual it corrects is small (sub-mm tangential slide). A larger move
# means the photometric solve wandered on a low-texture patch; reject it and
# keep the geometry result. Measured against the centroid of the stable region.
MAX_FINE_TRANS_MM = 6.0
MAX_FINE_ROT_DEG = 3.0


def _to_cloud(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals
    if mesh.has_vertex_colors():
        pcd.colors = mesh.vertex_colors
    return pcd


def _has_usable_color(cloud: o3d.geometry.PointCloud) -> bool:
    """Colored ICP needs spatial colour variation to lock onto. A flat (single
    skin-tone) cloud gives the photometric term no gradient, so it degenerates to
    geometry-only — skip the extra cost there."""
    if not cloud.has_colors():
        return False
    c = np.asarray(cloud.colors)
    return c.size > 0 and float(c.std()) > 1e-3


def _delta_ok(fine: np.ndarray, geom: np.ndarray, probe: np.ndarray) -> bool:
    """True if `fine` is within a drift-sized delta of the geometry solution
    `geom`, measured as the physical move of the stable-region centroid `probe`
    plus the rotation between the two transforms."""
    p = np.r_[probe, 1.0]
    trans_mm = float(np.linalg.norm((fine @ p)[:3] - (geom @ p)[:3]))
    dr = geom[:3, :3].T @ fine[:3, :3]
    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(dr) - 1.0) / 2.0, -1.0, 1.0))))
    return trans_mm <= MAX_FINE_TRANS_MM and rot_deg <= MAX_FINE_ROT_DEG


def _colored_refine(source: o3d.geometry.PointCloud,
                    target: o3d.geometry.PointCloud,
                    init: np.ndarray) -> np.ndarray | None:
    """Coarse-to-fine colored ICP refinement starting from `init`. Returns the
    refined transform, or None if colour is unusable or the solve fails."""
    if not (USE_COLORED_REG and _has_usable_color(source) and _has_usable_color(target)):
        return None
    transform = init
    try:
        for voxel, max_corr, iters in ((4.0, 8.0, 50), (2.0, 4.0, 30), (1.0, 2.0, 20)):
            s = source.voxel_down_sample(voxel)
            t = target.voxel_down_sample(voxel)
            for c in (s, t):
                c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel * 2.0, max_nn=30))
            transform = o3d.pipelines.registration.registration_colored_icp(
                s, t, max_corr, transform,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=iters)).transformation
    except RuntimeError:
        return None
    return transform


def register_with_exclusion(after: o3d.geometry.TriangleMesh,
                            before: o3d.geometry.TriangleMesh,
                            exclude_fn=None,
                            init: np.ndarray | None = None) -> np.ndarray:
    """Return the rigid transform that maps `after` into `before`'s frame.

    `exclude_fn(points) -> bool array` marks treated points (evaluated in the
    before frame after coarse alignment); they are dropped from the fine
    alignment pass.
    """
    source = _to_cloud(after)
    target = _to_cloud(before)

    # Coarse pass on everything (geometry only) to absorb the gross misalignment.
    transform = np.eye(4) if init is None else init
    for max_corr in (30.0, 10.0):
        result = o3d.pipelines.registration.registration_icp(
            source, target, max_corr, transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        transform = result.transformation

    # Fine pass on stable regions only.
    if exclude_fn is not None:
        moved = np.asarray(o3d.geometry.PointCloud(source).transform(transform).points)
        stable_source = source.select_by_index(np.flatnonzero(~exclude_fn(moved)))
    else:
        stable_source = source

    # Geometry fine pass first — locks the surface (validated ±0.2 mL path).
    for max_corr in (5.0, 2.0):
        result = o3d.pipelines.registration.registration_icp(
            stable_source, target, max_corr, transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        transform = result.transformation

    # Then a bounded colored refinement that pins residual tangential slide using
    # skin texture. Accepted only if it stays within a drift-sized delta of the
    # geometry solution; otherwise the geometry result stands.
    refined = _colored_refine(stable_source, target, transform)
    if refined is not None:
        probe = np.asarray(stable_source.points).mean(axis=0)
        if _delta_ok(refined, transform, probe):
            transform = refined
    return transform


def register_after_to_before(after: o3d.geometry.TriangleMesh,
                             before: o3d.geometry.TriangleMesh,
                             exclude_center: np.ndarray,
                             exclude_radius_mm: float) -> np.ndarray:
    """Spherical-exclusion convenience wrapper around register_with_exclusion."""
    def exclude_fn(points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points - exclude_center, axis=1) <= exclude_radius_mm

    return register_with_exclusion(after, before, exclude_fn)
