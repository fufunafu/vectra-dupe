"""Finish stage for CLOUD reconstructions: texture-bake the MVS mesh from the
original photos, render canonical views, build the comparison sheet.

Consumes outputs/<visit>/cloud/{mesh_raw.ply, cameras.npz} produced by
`modal run cloud/recon_gpu.py --visit <visit>`; writes the same artifacts the
local finish stage does (mesh.ply canonical, mesh_textured.glb, render_*.png,
comparison.png), so the app viewer and the quality gate work unchanged.

    .venv/bin/python -m vectra_sw.finish_cloud "data/V1 Pre TX" outputs/V1_Pre_TX
"""
from __future__ import annotations
import glob
import os
import sys

import numpy as np
import open3d as o3d

from . import compare, render_views, texture


def _mesh_from_depths(out_dir: str, cloud: str, names: list[str],
                      K: np.ndarray, w2c: np.ndarray):
    """TSDF-fuse the per-view MVS depth maps (hybrid path).

    Direct Poisson over the strictly-fused point cloud is detailed but sparse
    (pebbly balloon skin, holes). The raw geometric depth maps carry ~10x more
    surface; TSDF averaging turns them into a complete, smooth head while
    disagreements cancel inside the truncation band. Returns a mesh or None.
    """
    import cv2
    p = os.path.join(cloud, "depths.npz")
    if not os.path.isfile(p):
        return None
    d = np.load(p, allow_pickle=True)
    depth_by_name = {str(n): np.asarray(dm, np.float32)
                     for n, dm in zip(d["names"], d["depths"])}
    fused = o3d.io.read_point_cloud(os.path.join(cloud, "fused.ply"))
    pts = np.asarray(fused.points)
    if len(pts) < 1000:
        return None
    ext = np.percentile(pts, 98, 0) - np.percentile(pts, 2, 0)
    diag = float(np.linalg.norm(ext))
    voxel = diag / 300.0
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=voxel * 5.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    z_hi = float(np.percentile(depth_by_name[names[0]][depth_by_name[names[0]] > 0], 98)
                 if (depth_by_name.get(names[0]) is not None) else diag * 3) * 2.0
    n_used = 0
    for i, name in enumerate(names):
        dm = depth_by_name.get(name)
        if dm is None or (dm > 0).mean() < 0.01:
            continue
        hd, wd = dm.shape
        work = cv2.imread(os.path.join(out_dir, "work", name))
        if work is None:
            continue
        Hw, Ww = work.shape[:2]
        mask_p = os.path.join(out_dir, "masks", name + ".png")
        mk = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE) if os.path.isfile(mask_p) else None
        if mk is not None:
            mk = cv2.resize(mk, (wd, hd), interpolation=cv2.INTER_NEAREST)
            dm = np.where(mk > 128, dm, 0.0)
        # MVS depth carries ±2-3mm-scale speckle that vertex smoothing can't
        # remove after meshing; bilateral filtering kills it in the depth map
        # while preserving real edges (nose silhouette, lips).
        valid = dm > 0
        sm = cv2.bilateralFilter(dm, d=7, sigmaColor=float(dm[valid].std() * 0.5)
                                 if valid.any() else 1.0, sigmaSpace=5)
        dm = np.where(valid, sm, 0.0)
        color = cv2.cvtColor(cv2.resize(work, (wd, hd), cv2.INTER_AREA),
                             cv2.COLOR_BGR2RGB)
        Kd = K[i].copy()
        Kd[0] *= wd / Ww
        Kd[1] *= hd / Hw
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(np.ascontiguousarray(dm)),
            depth_scale=1.0, depth_trunc=z_hi, convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(
            wd, hd, Kd[0, 0], Kd[1, 1], Kd[0, 2], Kd[1, 2])
        vol.integrate(rgbd, intr, w2c[i])
        n_used += 1
    mesh = vol.extract_triangle_mesh()
    if len(mesh.vertices) == 0:
        return None
    tl, cnt, _ = mesh.cluster_connected_triangles()
    tl, cnt = np.asarray(tl), np.asarray(cnt)
    if len(cnt):
        mesh.remove_triangles_by_mask(cnt[tl] < cnt.max())
        mesh.remove_unreferenced_vertices()
    # MVS depth noise shows as orange-peel at this voxel size; Taubin here is
    # display-track smoothing (DSLR pipeline measures nothing yet), so smooth
    # harder than the phone pipeline's measurement mesh would allow.
    iters = int(os.environ.get("VECTRA_CLOUD_TAUBIN", "15"))
    mesh = mesh.filter_smooth_taubin(number_of_iterations=iters)
    mesh.compute_vertex_normals()
    print(f"[cloud-finish] tsdf from {n_used} depth maps: "
          f"{len(mesh.vertices)} verts", flush=True)
    return mesh


def run(visit_dir: str, out_dir: str):
    cloud = os.path.join(out_dir, "cloud")
    mesh_path = os.path.join(cloud, "mesh_raw.ply")
    cams_path = os.path.join(cloud, "cameras.npz")
    if not (os.path.isfile(mesh_path) and os.path.isfile(cams_path)):
        raise SystemExit(f"no cloud results under {cloud} — run the modal job first")

    cams = np.load(cams_path)
    names = [str(n) for n in cams["names"]]
    w2c = np.asarray(cams["world_to_cam"])            # (S,4,4)
    K = np.asarray(cams["K"])                         # (S,3,3) at work resolution
    extr34 = w2c[:, :3, :]
    extr44 = [w for w in w2c]

    # hybrid TSDF over the depth maps (complete + smooth); Poisson mesh fallback
    mesh = _mesh_from_depths(out_dir, cloud, names, K, w2c)
    if mesh is None:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
    print(f"[cloud-finish] mesh {len(mesh.vertices)} verts, {len(names)} cameras",
          flush=True)

    # work-image resolution these intrinsics are valid at (all photos uniform)
    import cv2
    work0 = cv2.imread(os.path.join(out_dir, "work", names[0]))
    work_hw = work0.shape[:2]

    originals = {os.path.basename(p): p
                 for p in glob.glob(os.path.join(visit_dir, "IMG_*.JPG"))
                 + glob.glob(os.path.join(visit_dir, "IMG_*.jpg"))}
    photo_paths = {n: originals[n] for n in names if n in originals}

    B = render_views.canonical_basis(extr44)
    _, samples = texture.bake(
        mesh, names, K, extr34, work_hw, photo_paths,
        os.path.join(out_dir, "masks"),
        os.path.join(out_dir, "mesh_textured.glb"), canonical=B)

    V, N, C = samples
    renders = render_views.render_arrays(V, N, C, extr44, out_dir, gain=1.0)

    # canonical mesh.ply for the viewer + quality gate (same as finish_vggt)
    Vm = np.asarray(mesh.vertices)
    mesh.vertices = o3d.utility.Vector3dVector((Vm - Vm.mean(0)) @ B)
    mesh.vertex_normals = o3d.utility.Vector3dVector(
        np.asarray(mesh.vertex_normals) @ B)
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "mesh.ply"), mesh)

    compare.build(visit_dir, renders, os.path.join(out_dir, "comparison.png"))
    print(f"DONE -> {out_dir}/mesh.ply (cloud), comparison.png", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
