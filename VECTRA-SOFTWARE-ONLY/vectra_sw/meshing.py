"""Surface meshing from the VGGT reconstruction (torch-free process).

Primary path: TSDF-fuse the per-view depth maps. Views that disagree average
out inside the truncated signed-distance volume instead of stacking as ghost
double surfaces, and hair/dark features are kept (artifact control is geometric
— mask + cross-view filtering upstream in vggt_recon — not a luminance cull).

NOTE: Open3D's Poisson reconstruction is unusable on macOS — it terminates the
whole process (exit 0!) mid-solve on real data. TSDF + marching cubes and
alpha-shape (fallback) are the safe reconstructors here.
"""
from __future__ import annotations
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

TAUBIN_ITERS = 5
TSDF_VOXEL_DIV = 350     # voxel = scene diag / this
TSDF_TRUNC_VOXELS = 4.0


def clean_cloud(points: np.ndarray, colors: np.ndarray):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    ext = np.percentile(points, 98, 0) - np.percentile(points, 2, 0)
    diag = float(np.linalg.norm(ext))
    voxel = diag / 300
    pc = pc.voxel_down_sample(voxel)
    pc, _ = pc.remove_statistical_outlier(20, 2.0)
    lab = np.asarray(pc.cluster_dbscan(eps=voxel * 4, min_points=10))
    if lab.max() >= 0:
        big = np.bincount(lab[lab >= 0]).argmax()
        pc = pc.select_by_index(np.where(lab == big)[0])
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
    n = np.asarray(pc.normals)
    out = np.asarray(pc.points) - np.asarray(pc.points).mean(0)
    n[np.sum(n * out, 1) < 0] *= -1
    pc.normals = o3d.utility.Vector3dVector(n)
    return pc, voxel


def build_mesh_tsdf(world_points: np.ndarray, valid: np.ndarray,
                    images: np.ndarray, extrinsic: np.ndarray,
                    intrinsic: np.ndarray, out_ply: str):
    """Fuse per-view depth maps into a TSDF and extract the surface.

    world_points (S,H,W,3) float, valid (S,H,W) bool, images (S,H,W,3) uint8,
    extrinsic (S,3,4) world->cam, intrinsic (S,3,3), all at the same resolution.
    """
    S, H, W = valid.shape
    pts_all = world_points[valid]
    ext = np.percentile(pts_all, 98, 0) - np.percentile(pts_all, 2, 0)
    diag = float(np.linalg.norm(ext))
    voxel = diag / TSDF_VOXEL_DIV
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=voxel * TSDF_TRUNC_VOXELS,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for s in range(S):
        R, t = extrinsic[s, :, :3], extrinsic[s, :, 3]
        z = world_points[s].reshape(-1, 3) @ R[2] + t[2]
        depth = np.where(valid[s].reshape(-1), z, 0.0).reshape(H, W)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(images[s])),
            o3d.geometry.Image(depth.astype(np.float32)),
            depth_scale=1.0, depth_trunc=float(z.max() + 1.0),
            convert_rgb_to_intensity=False)
        K = intrinsic[s]
        intr = o3d.camera.PinholeCameraIntrinsic(
            W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        T = np.vstack([extrinsic[s], [0, 0, 0, 1]])
        vol.integrate(rgbd, intr, T)
    mesh = vol.extract_triangle_mesh()
    if len(mesh.vertices) == 0:
        raise RuntimeError("TSDF fusion produced an empty mesh")
    return _finish_mesh(mesh, out_ply, tag="tsdf")


def build_mesh(points: np.ndarray, colors: np.ndarray, out_ply: str):
    """Fallback: alpha-shape over the filtered cloud (used if TSDF data is
    missing). Open3D Poisson is NOT an option on macOS — see module docstring."""
    pc, voxel = clean_cloud(points, colors)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pc, alpha=voxel * 6)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh = _finish_mesh(mesh, out_ply, tag="alpha-shape", color_from=pc)
    return mesh


def _finish_mesh(mesh, out_ply: str, tag: str, color_from=None):
    """Shared cleanup: keep the head component, light smoothing, color, save."""
    tl, cnt, _ = mesh.cluster_connected_triangles()
    tl, cnt = np.asarray(tl), np.asarray(cnt)
    if len(cnt):
        mesh.remove_triangles_by_mask(cnt[tl] < cnt.max())
        mesh.remove_unreferenced_vertices()
    colors = (np.asarray(mesh.vertex_colors).copy()
              if mesh.has_vertex_colors() else None)
    smoothed = mesh.filter_smooth_taubin(number_of_iterations=TAUBIN_ITERS)
    if len(smoothed.vertices) == len(mesh.vertices):
        mesh = smoothed
        if colors is not None:      # Taubin drops colours; count/order preserved
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    if color_from is not None:
        tree = cKDTree(np.asarray(color_from.points))
        _, idx = tree.query(np.asarray(mesh.vertices), k=1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(color_from.colors)[idx])
    o3d.io.write_triangle_mesh(out_ply, mesh)
    print(f"[mesh] {tag} -> {out_ply}: {len(mesh.vertices)} verts, "
          f"{len(mesh.triangles)} tris", flush=True)
    return mesh


def dense_surface_points(mesh, n: int = 600000):
    """Sample a dense colored point set from the (hole-filled) mesh surface for
    splat rendering -- gives solid coverage with no gaps."""
    pcd = mesh.sample_points_uniformly(number_of_points=n, use_triangle_normal=True)
    return (np.asarray(pcd.points), np.asarray(pcd.normals),
            np.asarray(pcd.colors))
