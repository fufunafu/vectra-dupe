"""Surface meshing from the VGGT point cloud (torch-free; Open3D's Poisson build
fatally aborts on macOS, so we use alpha-shape reconstruction which fills the
hair-streak holes). Colors are transferred from the dense cloud by nearest neighbor.
"""
from __future__ import annotations
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import scipy.sparse as sp


def _vertex_adjacency(mesh):
    tris = np.asarray(mesh.triangles)
    n = len(mesh.vertices)
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    return A, deg


def inpaint_dark_streaks(mesh, iters: int = 12, ratio: float = 0.82):
    """Replace vertices markedly darker than their neighbours (hair-bridge streaks)
    with the neighbour-average colour; preserves broad dark features (eyes, brows)."""
    C = np.asarray(mesh.vertex_colors).copy()
    A, deg = _vertex_adjacency(mesh)
    for _ in range(iters):
        neigh = A.dot(C) / deg[:, None]
        lum = C @ np.array([0.299, 0.587, 0.114])
        nlum = neigh @ np.array([0.299, 0.587, 0.114])
        dark = lum < ratio * nlum
        C[dark] = neigh[dark]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    return mesh


def clean_cloud(points: np.ndarray, colors: np.ndarray, skin_only=True,
                lum_thresh=0.17):
    if skin_only:
        lum = colors @ np.array([0.299, 0.587, 0.114])
        keep = lum > lum_thresh
        points, colors = points[keep], colors[keep]
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


def build_mesh(points: np.ndarray, colors: np.ndarray, out_ply: str):
    pc, voxel = clean_cloud(points, colors)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pc, alpha=voxel * 6)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    # keep largest connected component (the face)
    tl, cnt, _ = mesh.cluster_connected_triangles()
    tl, cnt = np.asarray(tl), np.asarray(cnt)
    if len(cnt):
        mesh.remove_triangles_by_mask(cnt[tl] < cnt.max())
        mesh.remove_unreferenced_vertices()
    mesh = mesh.filter_smooth_taubin(number_of_iterations=25)
    mesh.compute_vertex_normals()
    # transfer color from dense cloud by nearest neighbor
    tree = cKDTree(np.asarray(pc.points))
    _, idx = tree.query(np.asarray(mesh.vertices), k=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pc.colors)[idx])
    inpaint_dark_streaks(mesh)
    o3d.io.write_triangle_mesh(out_ply, mesh)
    print(f"[mesh] alpha-shape -> {out_ply}: {len(mesh.vertices)} verts, "
          f"{len(mesh.triangles)} tris", flush=True)
    return mesh


def dense_surface_points(mesh, n: int = 600000):
    """Sample a dense colored point set from the (hole-filled) mesh surface for
    splat rendering -- gives solid coverage with no gaps."""
    pcd = mesh.sample_points_uniformly(number_of_points=n, use_triangle_normal=True)
    return (np.asarray(pcd.points), np.asarray(pcd.normals),
            np.asarray(pcd.colors))
