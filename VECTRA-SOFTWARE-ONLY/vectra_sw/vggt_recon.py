"""Learned multi-view reconstruction with VGGT (Visual Geometry Grounded Transformer).

One feed-forward pass over the photos yields per-view camera poses, depth maps, and
world-space point maps with confidence -- robust to the low-parallax / turntable
capture that defeats classical SfM. We turn the confident world points into a colored
point cloud, then a Poisson surface.
"""
from __future__ import annotations
import os
import numpy as np
# NOTE: torch is imported lazily inside the VGGT functions only. Importing this module
# for to_mesh() must NOT load torch, or Open3D's Poisson (OpenMP) deadlocks against
# torch's libomp in the same process.

_MODEL = None
_DEVICE = os.environ.get("VECTRA_VGGT_DEVICE", "cpu")   # MPS deadlocks; CPU is safe


def _load():
    global _MODEL
    if _MODEL is None:
        import torch
        from vggt.models.vggt import VGGT
        print(f"[vggt] loading facebook/VGGT-1B on {_DEVICE} ...", flush=True)
        _MODEL = VGGT.from_pretrained("facebook/VGGT-1B").to(_DEVICE).eval()
    return _MODEL


def _head_masks(imgs: np.ndarray) -> np.ndarray:
    """Per-view head masks computed directly on the VGGT-preprocessed images
    (same near-black backdrop as the originals), so they are pixel-aligned with
    the point maps by construction. Eroded a little to kill silhouette bleed."""
    import cv2
    masks = np.zeros(imgs.shape[:3], bool)
    for i, im in enumerate((imgs * 255).astype(np.uint8)):
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        fg = (gray > 18).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
        if n > 1:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = (lab == big).astype(np.uint8)
            ff = m * 255
            h, w = m.shape
            cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
            m = (m * 255) | cv2.bitwise_not(ff)
            fg = (m > 0).astype(np.uint8)
        fg = cv2.erode(fg, np.ones((3, 3), np.uint8), iterations=1)
        masks[i] = fg.astype(bool)
    return masks


def _cross_view_filter(wp: np.ndarray, valid: np.ndarray, extrinsic: np.ndarray,
                       intrinsic: np.ndarray, neighbors: int = 2,
                       rel_tol: float = 0.01) -> np.ndarray:
    """Keep a point only if at least one neighboring view agrees on its depth.

    Ghost/double surfaces come from per-view point maps that disagree; a point
    from view s is projected into views s±1..s±neighbors and compared against
    that view's own depth there. A point visible in a neighbor but off by more
    than rel_tol of the median scene depth (and never confirmed) is dropped.
    Points that land in no neighbor's mask keep their benefit of the doubt.
    """
    S, H, W = wp.shape[:3]
    # per-view depth maps in each view's own camera
    depths = np.full((S, H, W), np.nan, np.float32)
    for s in range(S):
        R, t = extrinsic[s, :, :3], extrinsic[s, :, 3]
        z = wp[s].reshape(-1, 3) @ R[2] + t[2]
        d = np.where(valid[s].reshape(-1), z, np.nan)
        depths[s] = d.reshape(H, W)
    tol = rel_tol * np.nanmedian(depths)
    keep = np.zeros((S, H, W), bool)
    seen_bad = np.zeros((S, H, W), bool)
    for s in range(S):
        pts = wp[s][valid[s]]
        if not len(pts):
            continue
        confirmed = np.zeros(len(pts), bool)
        rejected = np.zeros(len(pts), bool)
        for nb in range(max(0, s - neighbors), min(S, s + neighbors + 1)):
            if nb == s:
                continue
            R, t = extrinsic[nb, :, :3], extrinsic[nb, :, 3]
            cam = pts @ R.T + t
            z = cam[:, 2]
            ok = z > 1e-6
            u = np.full(len(pts), -1.0)
            v = np.full(len(pts), -1.0)
            u[ok] = cam[ok, 0] / z[ok] * intrinsic[nb, 0, 0] + intrinsic[nb, 0, 2]
            v[ok] = cam[ok, 1] / z[ok] * intrinsic[nb, 1, 1] + intrinsic[nb, 1, 2]
            ui, vi = np.round(u).astype(int), np.round(v).astype(int)
            inb = ok & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            dn = np.full(len(pts), np.nan, np.float32)
            dn[inb] = depths[nb, vi[inb], ui[inb]]
            vis = np.isfinite(dn)
            agree = vis & (np.abs(dn - z) < tol)
            confirmed |= agree
            rejected |= vis & ~agree
        good = confirmed | ~rejected
        keep[s][valid[s]] = good
        seen_bad[s][valid[s]] = rejected & ~confirmed
    n_valid, n_keep = int(valid.sum()), int(keep.sum())
    print(f"[vggt] cross-view filter kept {n_keep}/{n_valid} "
          f"(dropped {int(seen_bad.sum())} inconsistent)", flush=True)
    return keep


def reconstruct(image_paths: list[str], out_dir: str, conf_percentile: float = 25.0):
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    model = _load()
    print(f"[vggt] preprocessing {len(image_paths)} images ...", flush=True)
    images = load_and_preprocess_images(image_paths).to(_DEVICE)   # (S,3,H,W)

    print("[vggt] running forward pass (CPU, may take a few min) ...", flush=True)
    with torch.no_grad():
        preds = model(images)               # batch dim added internally
    print("[vggt] forward done; keys:", list(preds.keys()), flush=True)

    HW = images.shape[-2:]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], HW)
    extrinsic = extrinsic[0].cpu().numpy()  # (S,3,4) world->cam (OpenCV)
    intrinsic = intrinsic[0].cpu().numpy()

    # world point map + confidence
    if "world_points" in preds:
        wp = preds["world_points"][0].cpu().numpy()          # (S,H,W,3)
        conf = preds["world_points_conf"][0].cpu().numpy()   # (S,H,W)
    else:
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        depth = preds["depth"][0].cpu().numpy()              # (S,H,W,1)
        wp = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
        conf = preds["depth_conf"][0].cpu().numpy()

    imgs = images.cpu().numpy().transpose(0, 2, 3, 1)        # (S,H,W,3) in [0,1]

    masks = _head_masks(imgs)
    conf = np.where(masks, conf, 0.0)   # background contributes nothing downstream

    thr = np.percentile(conf[masks], conf_percentile) if masks.any() else 0.0
    valid = masks & (conf >= thr) & np.isfinite(wp).all(-1)
    print(f"[vggt] mask+conf kept {int(valid.sum())}/{valid.size} "
          f"(conf>={thr:.2f}, p{conf_percentile:g} within mask)", flush=True)
    valid &= _cross_view_filter(wp, valid, extrinsic, intrinsic)

    pts = wp[valid]
    cols = imgs[valid]

    np.savez(os.path.join(out_dir, "vggt_raw.npz"),
             points=pts, colors=cols, extrinsic=extrinsic, intrinsic=intrinsic)

    # per-view maps for metric measurements (iris pixel -> 3D world point) and
    # for TSDF meshing (valid = mask & conf & cross-view-consistent pixels)
    names = [os.path.basename(p) for p in image_paths]
    np.savez(os.path.join(out_dir, "vggt_perview.npz"),
             world_points=wp.astype(np.float32),
             conf=conf.astype(np.float32),
             valid=valid,
             images=(imgs * 255).astype(np.uint8),
             extrinsic=extrinsic, intrinsic=intrinsic,
             names=np.array(names))
    return {"points": pts, "colors": cols,
            "extrinsics": [np.vstack([e, [0, 0, 0, 1]]) for e in extrinsic]}


def to_mesh(points: np.ndarray, colors: np.ndarray, out_ply: str,
            depth: int = 9) -> str:
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    pc = pc.voxel_down_sample(voxel_size=_auto_voxel(points))
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pc.estimate_normals()
    pc.orient_normals_consistent_tangent_plane(30)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.05))  # trim balloon
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_ply, mesh)
    print(f"[vggt] poisson mesh -> {out_ply}: {len(mesh.vertices)} verts", flush=True)
    return out_ply


def _auto_voxel(points: np.ndarray) -> float:
    ext = np.percentile(points, 97, 0) - np.percentile(points, 3, 0)
    return float(np.linalg.norm(ext) / 400.0)


if __name__ == "__main__":
    import sys, glob
    paths = sorted(glob.glob(os.path.join(sys.argv[1], "*.JPG")))
    r = reconstruct(paths, sys.argv[2])
    to_mesh(r["points"], r["colors"], os.path.join(sys.argv[2], "mesh_vggt.ply"))
