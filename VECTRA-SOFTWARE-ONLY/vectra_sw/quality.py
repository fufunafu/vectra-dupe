"""Quantitative quality gate: our renders vs VECTRA's reference screenshots.

For each canonical render, finds the best-matching VECTRA screenshot (their
export order includes up/down views and doesn't match our azimuth sweep), then
scores: silhouette IoU (centroid+scale aligned), MediaPipe landmark residual
(2D similarity-aligned, normalized by inter-ocular distance), and masked SSIM
on luminance. Plus intrinsic mesh stats. -> quality_report.json

Acceptance bars (PLAN): frontal IoU >= 0.90, profiles >= 0.85,
landmark error <= 3% of inter-ocular distance.
"""
from __future__ import annotations
import glob
import json
import os
import subprocess

import cv2
import numpy as np

MP_PYTHON = os.path.join(os.path.dirname(__file__), "..", ".venv-mp", "bin", "python")
LM_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "tools", "landmark_detect.py")

SIL_THRESH = 55           # luminance above this = subject (both use dark backdrops)
WORK_H = 700              # common working height for alignment/scoring


def _silhouette(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fg = (gray > SIL_THRESH).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return fg
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == big).astype(np.uint8)
    ff = m * 255
    h, w = m.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return ((m * 255) | cv2.bitwise_not(ff)) // 255


def _norm_mask(mask: np.ndarray, img: np.ndarray, out_hw=(WORK_H, WORK_H)):
    """Center the silhouette and scale it to a canonical area; apply the same
    transform to the (grayscale) image. Returns (mask, gray) at out_hw."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 100:
        return None, None
    cy, cx = ys.mean(), xs.mean()
    area = float(len(ys))
    target_area = 0.28 * out_hw[0] * out_hw[1]
    s = np.sqrt(target_area / area)
    M = np.float32([[s, 0, out_hw[1] / 2 - s * cx], [0, s, out_hw[0] / 2 - s * cy]])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m2 = cv2.warpAffine(mask, M, (out_hw[1], out_hw[0]), flags=cv2.INTER_NEAREST)
    g2 = cv2.warpAffine(gray, M, (out_hw[1], out_hw[0]), flags=cv2.INTER_LINEAR)
    return m2, g2


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def _ssim_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean local SSIM over the mask (standard 11x11 Gaussian formulation)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)   # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    va = blur(a * a) - mu_a ** 2
    vb = blur(b * b) - mu_b ** 2
    cab = blur(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cab + C2)) / (
        (mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2) + 1e-12)
    m = mask > 0
    return float(s[m].mean()) if m.any() else 0.0


def _landmarks(path: str):
    res = subprocess.run([MP_PYTHON, LM_SCRIPT, path],
                         capture_output=True, text=True, timeout=120)
    if res.returncode != 0 or not res.stdout.strip():
        return None
    out = json.loads(res.stdout.strip().splitlines()[-1])
    return np.asarray(out["landmarks"]) if out.get("ok") else None


def _umeyama2d(src: np.ndarray, dst: np.ndarray):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / (s ** 2).sum() * len(src)
    return lambda p: scale * (p - mu_s) @ R.T + mu_d


def _landmark_error(our_png: str, ref_jpg: str):
    """Mean 2D residual after similarity alignment, / inter-ocular distance."""
    lo, lr = _landmarks(our_png), _landmarks(ref_jpg)
    if lo is None or lr is None:
        return None
    T = _umeyama2d(lo, lr)
    res = np.linalg.norm(T(lo) - lr, axis=1)
    inter_ocular = np.linalg.norm(lr[33] - lr[263])   # outer eye corners
    return float(res.mean() / (inter_ocular + 1e-9))


def _mesh_stats(mesh_ply: str) -> dict:
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(mesh_ply)
    tris = np.asarray(m.triangles)
    edges = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]],
                               tris[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {"vertices": len(m.vertices), "triangles": len(tris),
            "boundary_edges": int((counts == 1).sum()),
            "watertight": bool((counts == 2).all())}


def evaluate(visit_dir: str, out_dir: str) -> dict:
    refs = sorted(glob.glob(os.path.join(visit_dir, "2025*.jpg")))
    ours = sorted(glob.glob(os.path.join(out_dir, "render_*_az*.png")))
    if not refs or not ours:
        raise RuntimeError(f"missing renders ({len(ours)}) or refs ({len(refs)})")

    ref_data = []
    for r in refs:
        img = cv2.imread(r)
        m, g = _norm_mask(_silhouette(img), img)
        if m is not None:
            ref_data.append((os.path.basename(r), r, m, g))

    pairs = []
    for o in ours:
        img = cv2.imread(o)
        m, g = _norm_mask(_silhouette(img), img)
        if m is None:
            continue
        best = max(ref_data, key=lambda rd: _iou(m, rd[2]))
        iou = _iou(m, best[2])
        entry = {"render": os.path.basename(o), "ref": best[0],
                 "silhouette_iou": round(iou, 4),
                 "ssim_lum": round(_ssim_masked(
                     g, best[3], np.logical_and(m, best[2])), 4)}
        if "az+0" in o:   # landmarks only meaningful near-frontal
            err = _landmark_error(o, best[1])
            entry["landmark_err_interocular"] = (
                round(err, 4) if err is not None else None)
        pairs.append(entry)
        print(f"[quality] {entry}", flush=True)

    ious = [p["silhouette_iou"] for p in pairs]
    report = {
        "pairs": pairs,
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "min_iou": round(float(np.min(ious)), 4) if ious else None,
        "mean_ssim": round(float(np.mean([p["ssim_lum"] for p in pairs])), 4)
        if pairs else None,
        "mesh": _mesh_stats(os.path.join(out_dir, "mesh.ply")),
    }
    with open(os.path.join(out_dir, "quality_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[quality] mean IoU={report['mean_iou']} mean SSIM={report['mean_ssim']} "
          f"mesh={report['mesh']}", flush=True)
    return report


if __name__ == "__main__":
    import sys
    evaluate(sys.argv[1], sys.argv[2])
