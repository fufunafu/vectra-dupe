"""Stage 3 - learned dense reconstruction.

For each registered view: run Depth Anything V2 to get a dense (affine-invariant)
disparity map, then fit a robust affine transform that maps it into COLMAP's metric
scale using the sparse 3D points visible in that view. Fuse all aligned depth maps
with their COLMAP poses into a single surface via Open3D TSDF integration.
"""
from __future__ import annotations
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d

_MODEL = None
_PROC = None
# MPS deadlocks on Depth Anything's first forward pass; CPU is reliable and the
# M5 Pro runs it in ~1-2 s/image. Override with VECTRA_DEPTH_DEVICE=mps to retry.
_DEVICE = os.environ.get("VECTRA_DEPTH_DEVICE", "cpu")
_MODEL_NAME = "depth-anything/Depth-Anything-V2-Large-hf"


def _load_model():
    global _MODEL, _PROC
    if _MODEL is None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        print(f"[dense] loading {_MODEL_NAME} on {_DEVICE} ...")
        _PROC = AutoImageProcessor.from_pretrained(_MODEL_NAME)
        _MODEL = AutoModelForDepthEstimation.from_pretrained(_MODEL_NAME).to(_DEVICE).eval()
    return _MODEL, _PROC


def predict_disparity(bgr: np.ndarray) -> np.ndarray:
    """Dense affine-invariant disparity (higher = closer), at the image resolution."""
    model, proc = _load_model()
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inputs = proc(images=rgb, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        pred = model(**inputs).predicted_depth  # (1, h, w), disparity-like
    pred = F.interpolate(pred[None], size=(H, W), mode="bilinear", align_corners=False)
    return pred[0, 0].float().cpu().numpy()


def _robust_affine(disp_vals: np.ndarray, inv_depth: np.ndarray, iters: int = 200,
                   thresh: float = 0.1):
    """RANSAC fit inv_depth ~= a*disp + b (in 1/Z space). Returns (a, b, inlier_frac)."""
    n = len(disp_vals)
    if n < 8:
        A = np.c_[disp_vals, np.ones(n)]
        a, b = np.linalg.lstsq(A, inv_depth, rcond=None)[0]
        return a, b, 1.0
    rng = np.random.default_rng(0)
    best_in, best = -1, (1.0, 0.0)
    scale = np.median(np.abs(inv_depth)) + 1e-9
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if abs(disp_vals[i] - disp_vals[j]) < 1e-6:
            continue
        a = (inv_depth[i] - inv_depth[j]) / (disp_vals[i] - disp_vals[j])
        b = inv_depth[i] - a * disp_vals[i]
        res = np.abs(a * disp_vals + b - inv_depth)
        nin = int(np.sum(res < thresh * scale))
        if nin > best_in:
            best_in, best = nin, (a, b)
    a, b = best
    res = np.abs(a * disp_vals + b - inv_depth)
    inl = res < thresh * scale
    if inl.sum() >= 8:  # refit on inliers
        A = np.c_[disp_vals[inl], np.ones(inl.sum())]
        a, b = np.linalg.lstsq(A, inv_depth[inl], rcond=None)[0]
    return a, b, best_in / n


def aligned_depth(disp: np.ndarray, uv: np.ndarray, sparse_depth: np.ndarray):
    """Map a disparity map into metric depth (COLMAP scale) using sparse correspondences."""
    H, W = disp.shape
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
    d_at = disp[v, u]
    inv_z = 1.0 / np.clip(sparse_depth, 1e-6, None)
    a, b, frac = _robust_affine(d_at, inv_z)
    inv_dense = a * disp + b
    Z = np.where(inv_dense > 1e-6, 1.0 / inv_dense, 0.0)
    # clip to plausible range around observed sparse depths
    lo, hi = np.percentile(sparse_depth, 2), np.percentile(sparse_depth, 98)
    Z[(Z < lo * 0.6) | (Z > hi * 1.6)] = 0.0
    return Z.astype(np.float32), frac


def fuse(views: list[dict], work_dir: str, mask_dir: str, out_dir: str,
         voxel_div: int = 700) -> str:
    # scene scale from union of sparse depths -> voxel size
    all_d = np.concatenate([v["depth"] for v in views if len(v["depth"])])
    scene = float(np.percentile(all_d, 95) - np.percentile(all_d, 5))
    voxel = max(scene / voxel_div, 1e-4)
    trunc = voxel * 5
    print(f"[dense] scene depth-span~{scene:.3f}, voxel={voxel:.5f}, sdf_trunc={trunc:.5f}")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for k, v in enumerate(views):
        bgr = cv2.imread(os.path.join(work_dir, v["name"]))
        mask = cv2.imread(os.path.join(mask_dir, v["name"] + ".png"), cv2.IMREAD_GRAYSCALE)
        if bgr is None or len(v["depth"]) < 8:
            continue
        H, W = bgr.shape[:2]
        disp = predict_disparity(bgr)
        Z, frac = aligned_depth(disp, v["uv"], v["depth"])
        if mask is not None:
            Z[mask < 128] = 0.0

        color = o3d.geometry.Image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy())
        depth = o3d.geometry.Image(Z)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1.0, depth_trunc=scene * 3,
            convert_rgb_to_intensity=False)
        K = v["K"]
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        volume.integrate(rgbd, intr, v["extrinsic"])
        print(f"[dense]  {k+1:2d}/{len(views)} {v['name']}  align-inliers={frac:.0%}")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out = os.path.join(out_dir, "fused_raw.ply")
    o3d.io.write_triangle_mesh(out, mesh)
    print(f"[dense] fused mesh: {len(mesh.vertices)} verts -> {out}")
    return out


if __name__ == "__main__":
    import sys, sfm
    rec = sfm.load(sys.argv[1])
    fuse(sfm.camera_views(rec), sys.argv[2], sys.argv[3], sys.argv[4])
