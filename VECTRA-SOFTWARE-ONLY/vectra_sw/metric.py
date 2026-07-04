"""Metric scale + facial measurements (Plan 2).

VGGT geometry is metric-consistent only up to an unknown global scale. We fix the
scale from inter-pupillary distance: detect iris centers on the frontal view, map
them to 3D via the per-view world-point maps, and scale so IPD matches a known
value. Then report standard facial anthropometrics in millimetres.

NOTE: with no physical scale reference in the photos, IPD is assumed from a
population average (adult female ~63 mm), so absolute sizes carry ~5% uncertainty.
Relative change between visits (after the same scaling) is far more reliable.
"""
from __future__ import annotations
import os, json, subprocess
import numpy as np
import cv2

from .render_views import canonical_basis

MP_PYTHON = os.path.join(os.path.dirname(__file__), "..", ".venv-mp", "bin", "python")
LM_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "tools", "landmark_detect.py")
ASSUMED_IPD_MM = 63.0
# Horizontal visible iris diameter is remarkably constant across adults —
# an independent second prior to cross-check the IPD-derived scale.
IRIS_DIAMETER_MM = 11.7
MAX_SCALE_VIEWS = 5          # how many near-frontal views vote on the scale
MAX_SCALE_YAW_DEG = 40.0     # landmark quality collapses past this yaw
# MediaPipe iris refinement points: [center, right, top, left, bottom] per eye
LEFT_IRIS_H = (469, 471)     # horizontal boundary pair, left eye
RIGHT_IRIS_H = (474, 476)

# MediaPipe FaceMesh landmark indices
IDX = {"nose_tip": 1, "menton": 152, "nasion": 168,
       "zyg_r": 234, "zyg_l": 454, "alar_r": 48, "alar_l": 278,
       "mouth_r": 61, "mouth_l": 291, "eye_out_r": 33, "eye_out_l": 263}


def _lookup_3d(world_points, conf, x, y, win=4):
    """Median world point in a small window around (x,y) over confident pixels."""
    H, W = conf.shape
    x, y = int(round(x)), int(round(y))
    x0, x1 = max(0, x - win), min(W, x + win + 1)
    y0, y1 = max(0, y - win), min(H, y + win + 1)
    wp = world_points[y0:y1, x0:x1].reshape(-1, 3)
    c = conf[y0:y1, x0:x1].reshape(-1)
    good = np.isfinite(wp).all(1) & (c >= np.median(c))
    if good.sum() == 0:
        return None
    return np.median(wp[good], 0)


def _detect_landmarks(image_bgr, out_dir, up=3):
    # upscale the small VGGT frame for reliable detection, then map coords back
    big = cv2.resize(image_bgr, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    p = os.path.join(out_dir, "_frontal_for_landmarks.png")
    cv2.imwrite(p, big)
    res = subprocess.run([MP_PYTHON, LM_SCRIPT, p], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"landmark detect failed: {res.stderr[-500:]}")
    out = json.loads(res.stdout.strip().splitlines()[-1])
    if out.get("ok"):
        out["left_iris"] = [c / up for c in out["left_iris"]]
        out["right_iris"] = [c / up for c in out["right_iris"]]
        out["landmarks"] = [[x / up, y / up] for x, y in out["landmarks"]]
    return out


def measure(out_dir: str) -> dict:
    d = np.load(os.path.join(out_dir, "vggt_perview.npz"), allow_pickle=True)
    wp, conf, imgs = d["world_points"], d["conf"], d["images"]
    extr = [np.vstack([e, [0, 0, 0, 1]]) for e in d["extrinsic"]]

    # Rank views by how directly they face the front; the most frontal one
    # carries the anthropometric measurements, and the top MAX_SCALE_VIEWS
    # each cast an independent vote on the metric scale (median rejects the
    # odd bad landmark fit — single-view scale wobbled ~±2% across visits).
    B = canonical_basis(extr)
    front = B[:, 2]
    view_dirs = np.array([T[:3, :3][2, :] for T in extr])   # cam z-axis in world
    facing = view_dirs @ front
    ranked = np.argsort(facing)                              # most frontal first
    fi = int(ranked[0])
    print(f"[metric] frontal view = index {fi} ({d['names'][fi]})", flush=True)

    ipds, iris_diams = [], []
    lm = None
    for vi in ranked[:MAX_SCALE_VIEWS]:
        vi = int(vi)
        lmi = _detect_landmarks(cv2.cvtColor(imgs[vi], cv2.COLOR_RGB2BGR), out_dir)
        if not lmi.get("ok"):
            continue
        if vi == fi:
            lm = lmi                                # keep for measurements below
        L = lmi["landmarks"]

        def p3v(xy, _vi=vi):
            return _lookup_3d(wp[_vi], conf[_vi], xy[0], xy[1])

        li, ri = p3v(lmi["left_iris"]), p3v(lmi["right_iris"])
        if li is not None and ri is not None:
            ipds.append(float(np.linalg.norm(li - ri)))
        for a, b in (LEFT_IRIS_H, RIGHT_IRIS_H):
            if a < len(L) and b < len(L):
                pa, pb = p3v(L[a]), p3v(L[b])
                if pa is not None and pb is not None:
                    iris_diams.append(float(np.linalg.norm(pa - pb)))
    if not ipds:
        raise RuntimeError("could not map irises to 3D in any frontal view")
    if lm is None:
        img = cv2.cvtColor(imgs[fi], cv2.COLOR_RGB2BGR)
        lm = _detect_landmarks(img, out_dir)
        if not lm.get("ok"):
            raise RuntimeError("no face detected on frontal view")

    ipd_units = float(np.median(ipds))
    ipd_spread = (float(np.ptp(ipds) / ipd_units) if len(ipds) > 1 else None)
    scale = ASSUMED_IPD_MM / ipd_units      # mm per world unit
    iris_scale = (IRIS_DIAMETER_MM / float(np.median(iris_diams))
                  if iris_diams else None)
    agree_pct = (round(100 * abs(iris_scale - scale) / scale, 1)
                 if iris_scale else None)
    print(f"[metric] IPD={ipd_units:.4f} units over {len(ipds)} views "
          f"(spread {100 * (ipd_spread or 0):.1f}%) -> scale {scale:.2f} mm/unit; "
          f"iris-diameter scale {iris_scale and round(iris_scale, 2)} "
          f"(disagrees {agree_pct}%)", flush=True)
    if agree_pct is not None and agree_pct > 5:
        print(f"[metric] WARNING: IPD and iris-diameter scales disagree by "
              f"{agree_pct}% — treat absolute mm with caution", flush=True)

    def p3(name_or_xy):
        xy = lm[name_or_xy] if isinstance(name_or_xy, str) else name_or_xy
        return _lookup_3d(wp[fi], conf[fi], xy[0], xy[1])

    L = lm["landmarks"]
    pt = {k: p3(L[i]) for k, i in IDX.items()}

    def dist(a, b):
        if pt[a] is None or pt[b] is None:
            return None
        return round(float(np.linalg.norm(pt[a] - pt[b])) * scale, 1)

    measures_mm = {
        "interpupillary_distance": round(ASSUMED_IPD_MM, 1),
        "bizygomatic_face_width": dist("zyg_r", "zyg_l"),
        "morphological_face_height": dist("nasion", "menton"),
        "nose_length": dist("nasion", "nose_tip"),
        "nose_width": dist("alar_r", "alar_l"),
        "mouth_width": dist("mouth_r", "mouth_l"),
        "outer_eye_width": dist("eye_out_r", "eye_out_l"),
    }
    result = {"scale_mm_per_unit": scale, "ipd_units": ipd_units,
              "scale_views": len(ipds), "scale_spread_pct":
              round(100 * ipd_spread, 2) if ipd_spread is not None else None,
              "iris_scale_mm_per_unit":
              round(iris_scale, 4) if iris_scale else None,
              "scale_disagreement_pct": agree_pct,
              "frontal_view": d["names"][fi].item(), "measures_mm": measures_mm}
    with open(os.path.join(out_dir, "measurements.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("[metric] measurements (mm):", flush=True)
    for k, v in measures_mm.items():
        print(f"    {k:28s} {v}", flush=True)
    return result


if __name__ == "__main__":
    import sys
    measure(sys.argv[1])
