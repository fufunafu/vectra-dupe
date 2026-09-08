"""Conservative engineering screens, not a clinical accuracy certification.

Thresholds reject obvious reconstruction/alignment failures. Passing them does
not establish a minimum detectable volume or real-patient repeatability.
"""

import hashlib

import numpy as np
import open3d as o3d

QUALITY_VERSION = 1
MIN_FRONT_COVERAGE = 0.85
MAX_LAYERED_FRACTION = 0.10
MIN_ALIGNMENT_FITNESS = 0.85
MAX_ALIGNMENT_RMSE_MM = 1.0
MIN_DISTANCE_COVERAGE = 0.85


class QualityError(ValueError):
    """A scan or comparison did not pass the measurement safety screens."""


def mesh_fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_mesh(mesh: o3d.geometry.TriangleMesh) -> dict:
    """Check the central frontal surface in the canonical millimetre frame.

    Rays with multiple distinct hits in the front hemisphere expose overlapping
    sheets that a texture or smoothing pass can conceal. This deliberately
    avoids the ears, hairline and back of the head. Coincident triangle-edge
    hits within 0.5 mm are counted once.
    """
    report = {"version": QUALITY_VERSION, "eligible": False, "reasons": [],
              "note": "Engineering checks only; real-patient accuracy is unvalidated."}
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if (len(vertices) < 3 or len(triangles) < 1
            or not np.isfinite(vertices).all()
            or triangles.min() < 0 or triangles.max() >= len(vertices)):
        report["reasons"].append("Measurement mesh is empty or invalid.")
        return report
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    x, y = np.meshgrid(np.arange(-60, 61, 2), np.arange(-50, 51, 2))
    rays = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 200),
                            np.zeros(x.size), np.zeros(x.size), -np.ones(x.size)))
    hits = scene.list_intersections(o3d.core.Tensor(rays.astype(np.float32)))
    ids, distances = hits["ray_ids"].numpy(), hits["t_hit"].numpy()
    front = np.isfinite(distances) & (distances > 0) & (distances < 200)
    ids, distances = ids[front], distances[front]
    order = np.lexsort((distances, ids))
    ids, distances = ids[order], distances[order]
    counts = np.zeros(x.size, dtype=int)
    if len(ids):
        distinct = np.r_[True, (ids[1:] != ids[:-1]) | (np.diff(distances) > 0.5)]
        counts = np.bincount(ids[distinct], minlength=x.size)
    coverage = float(np.mean(counts > 0))
    layered = float(np.sum(counts > 1) / max(1, np.sum(counts > 0)))
    report.update(front_coverage=round(coverage, 4),
                  layered_fraction=round(layered, 4),
                  limits={"min_front_coverage": MIN_FRONT_COVERAGE,
                          "max_layered_fraction": MAX_LAYERED_FRACTION})
    if coverage < MIN_FRONT_COVERAGE:
        report["reasons"].append(f"Incomplete frontal surface ({coverage:.0%} coverage).")
    if layered > MAX_LAYERED_FRACTION:
        report["reasons"].append(f"Overlapping surface layers ({layered:.0%} of frontal rays).")
    report["eligible"] = not report["reasons"]
    return report


def require_mesh(mesh: o3d.geometry.TriangleMesh, label: str) -> dict:
    report = inspect_mesh(mesh)
    if not report["eligible"]:
        raise QualityError(f"{label}: " + " ".join(report["reasons"]))
    return report


def inspect_alignment(before, after_aligned) -> dict:
    """Check BOTH directions before bias subtraction can hide misalignment."""
    clouds = []
    for mesh in (before, after_aligned):
        cloud = o3d.geometry.PointCloud()
        cloud.points = mesh.vertices
        clouds.append(cloud.voxel_down_sample(1.0))
    metrics = []
    for source, target in (clouds, clouds[::-1]):
        result = o3d.pipelines.registration.evaluate_registration(
            source, target, 2.0, np.eye(4))
        metrics.append({"fitness": float(result.fitness),
                        "rmse_mm": float(result.inlier_rmse)})
    failed = any(m["fitness"] < MIN_ALIGNMENT_FITNESS
           or not np.isfinite(m["rmse_mm"])
           or m["rmse_mm"] > MAX_ALIGNMENT_RMSE_MM for m in metrics)
    return {"directions": metrics, "eligible": not failed,
            "limits": {"min_fitness": MIN_ALIGNMENT_FITNESS,
                       "max_rmse_mm": MAX_ALIGNMENT_RMSE_MM}}


def require_alignment(before, after_aligned) -> dict:
    report = inspect_alignment(before, after_aligned)
    if not report["eligible"]:
        raise QualityError("Before/after alignment or shared coverage is inadequate; "
                           "no volume result was produced.")
    return report


def require_distance_coverage(field, eligible_mask, label="Comparison") -> float:
    areas = field.vertex_areas
    total = float(areas[eligible_mask].sum())
    valid = eligible_mask & np.isfinite(field.distances)
    coverage = float(areas[valid].sum() / total) if total > 0 else 0.0
    if coverage < MIN_DISTANCE_COVERAGE:
        raise QualityError(f"{label}: only {coverage:.0%} valid surface coverage; "
                           "no volume result was produced.")
    return coverage
