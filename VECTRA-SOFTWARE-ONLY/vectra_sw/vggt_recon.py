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


def reconstruct(image_paths: list[str], out_dir: str, conf_percentile: float = 50.0):
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

    pts = wp.reshape(-1, 3)
    cols = imgs.reshape(-1, 3)
    c = conf.reshape(-1)
    thr = np.percentile(c, conf_percentile)
    keep = (c >= thr) & np.isfinite(pts).all(1)
    pts, cols = pts[keep], cols[keep]
    print(f"[vggt] kept {keep.sum()}/{len(keep)} points (conf>={thr:.2f})", flush=True)

    np.savez(os.path.join(out_dir, "vggt_raw.npz"),
             points=pts, colors=cols, extrinsic=extrinsic, intrinsic=intrinsic)

    # per-view maps for metric measurements (iris pixel -> 3D world point)
    names = [os.path.basename(p) for p in image_paths]
    np.savez(os.path.join(out_dir, "vggt_perview.npz"),
             world_points=wp.astype(np.float32),
             conf=conf.astype(np.float32),
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
