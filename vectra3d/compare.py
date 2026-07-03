"""Full session comparison with automatic changed-region detection.

Two-pass scheme, because registration and bias correction must exclude the
treated regions but we don't know where they are yet:

  pass 1: register on everything -> distance field -> rough bias removal ->
          detect candidate change regions (smoothed |d| threshold + connected
          components on the mesh graph)
  pass 2: re-register and re-fit the bias field EXCLUDING those regions ->
          final per-region volumes

Sign convention: positive distance = "after" surface sits outside "before"
(volume gain, e.g. filler).
"""

from dataclasses import dataclass, field as dataclass_field

import numpy as np
import open3d as o3d
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from . import analyze, register

DETECT_THRESHOLD_MM = 0.35   # ~1.5x the repeat-scan RMS measured in phase 0
MIN_REGION_AREA_MM2 = 150.0
MIN_REGION_VOLUME_MM3 = 40.0
ROI_MARGIN_MM = 8.0
MAX_ROI_RADIUS_MM = 30.0     # cap so exclusion holes stay bridgeable
EXCLUDE_MARGIN_MM = 4.0
MAX_REGIONS = 6
NOISE_FLOOR_MM3 = 200.0      # ~1 sigma of a repeat-scan measurement


@dataclass
class ChangeRegion:
    center: list  # [x, y, z] in the before frame
    radius_mm: float
    volume_mm3: float
    area_mm2: float
    peak_mm: float
    mean_mm: float

    def to_dict(self) -> dict:
        return {"center": [round(c, 2) for c in self.center],
                "radius_mm": round(self.radius_mm, 1),
                "volume_mm3": round(self.volume_mm3, 1),
                "volume_ml": round(self.volume_mm3 / 1000.0, 3),
                "area_mm2": round(self.area_mm2, 1),
                "peak_mm": round(self.peak_mm, 2),
                "mean_mm": round(self.mean_mm, 3),
                "significant": bool(abs(self.volume_mm3) >= NOISE_FLOOR_MM3)}


@dataclass
class CompareResult:
    field: analyze.DistanceField
    regions: list = dataclass_field(default_factory=list)
    transform: np.ndarray = None
    aligned_after: o3d.geometry.TriangleMesh = None


def _smooth_on_mesh(adj: sp.csr_matrix, values: np.ndarray,
                    iterations: int = 8) -> np.ndarray:
    """Neighborhood-average smoothing that tolerates NaNs."""
    valid = np.isfinite(values).astype(float)
    s = np.nan_to_num(values)
    w = valid.copy()
    for _ in range(iterations):
        s = s + adj @ s
        w = w + adj @ w
        s = np.where(w > 0, s / w.clip(min=1e-12), 0.0)
        w = (w > 0).astype(float)
    s[valid == 0] = np.nan
    return s


def detect_change_regions(mesh: o3d.geometry.TriangleMesh,
                          field: analyze.DistanceField) -> list[ChangeRegion]:
    adj = analyze.adjacency(mesh)
    smooth = _smooth_on_mesh(adj, field.distances)
    candidate = np.isfinite(smooth) & (np.abs(smooth) > DETECT_THRESHOLD_MM)
    if not candidate.any():
        return []

    idx = np.flatnonzero(candidate)
    sub = adj[idx][:, idx]
    n_comp, labels = connected_components(sub, directed=False)

    regions = []
    v = field.vertices
    for comp in range(n_comp):
        comp_idx = idx[labels == comp]
        area = float(field.vertex_areas[comp_idx].sum())
        if area < MIN_REGION_AREA_MM2:
            continue
        weights = field.vertex_areas[comp_idx]
        center = (v[comp_idx] * weights[:, None]).sum(axis=0) / weights.sum()
        radius = float(min(
            np.linalg.norm(v[comp_idx] - center, axis=1).max() + ROI_MARGIN_MM,
            MAX_ROI_RADIUS_MM))
        volume = analyze.roi_volume_mm3(field, center, radius)
        if abs(volume) < MIN_REGION_VOLUME_MM3:
            continue
        d_comp = field.distances[comp_idx]
        regions.append(ChangeRegion(
            center=center.tolist(), radius_mm=radius, volume_mm3=volume,
            area_mm2=area, peak_mm=float(np.nanmax(np.abs(d_comp))),
            mean_mm=float(np.nanmean(d_comp))))

    regions.sort(key=lambda r: -abs(r.volume_mm3))
    return regions[:MAX_REGIONS]


def _exclusion_mask(points: np.ndarray, regions: list[ChangeRegion]) -> np.ndarray:
    mask = np.zeros(len(points), dtype=bool)
    for r in regions:
        mask |= (np.linalg.norm(points - np.asarray(r.center), axis=1)
                 <= r.radius_mm + EXCLUDE_MARGIN_MM)
    return mask


def compare_sessions(before: o3d.geometry.TriangleMesh,
                     after: o3d.geometry.TriangleMesh) -> CompareResult:
    # Pass 1: blind registration and rough field to locate change regions.
    t1 = register.register_with_exclusion(after, before)
    aligned = o3d.geometry.TriangleMesh(after).transform(t1)
    rough = analyze.signed_distance_field(before, aligned)
    rough = analyze.subtract_bias_field(rough)
    regions = detect_change_regions(before, rough)

    if not regions:
        return CompareResult(field=rough, regions=[], transform=t1,
                             aligned_after=aligned)

    # Pass 2: redo registration and bias fit with the regions excluded, then
    # re-measure volumes over the FROZEN pass-1 region geometry. Re-detecting
    # here would feed back: bigger holes -> worse bias interpolation -> more
    # suprathreshold area -> bigger holes.
    t2 = register.register_with_exclusion(
        after, before, exclude_fn=lambda pts: _exclusion_mask(pts, regions),
        init=t1)
    aligned = o3d.geometry.TriangleMesh(after).transform(t2)
    fine = analyze.signed_distance_field(before, aligned)
    fine = analyze.subtract_bias_field(
        fine, exclude_mask=_exclusion_mask(fine.vertices, regions))

    final_regions = []
    for r in regions:
        center = np.asarray(r.center)
        volume = analyze.roi_volume_mm3(fine, center, r.radius_mm)
        if abs(volume) < MIN_REGION_VOLUME_MM3:
            continue
        in_roi = (np.linalg.norm(fine.vertices - center, axis=1) <= r.radius_mm)
        d_roi = fine.distances[in_roi]
        final_regions.append(ChangeRegion(
            center=r.center, radius_mm=r.radius_mm, volume_mm3=volume,
            area_mm2=r.area_mm2,
            peak_mm=float(np.nanmax(np.abs(d_roi))) if np.isfinite(d_roi).any() else 0.0,
            mean_mm=float(np.nanmean(d_roi)) if np.isfinite(d_roi).any() else 0.0))
    final_regions.sort(key=lambda r: -abs(r.volume_mm3))
    return CompareResult(field=fine, regions=final_regions, transform=t2,
                         aligned_after=aligned)
