"""Stage 4 - clean the fused mesh: keep the largest connected component (the head),
drop small floaters, light smoothing."""
from __future__ import annotations
import numpy as np
import open3d as o3d


def clean(in_ply: str, out_ply: str, min_cluster_frac: float = 0.1) -> str:
    mesh = o3d.io.read_triangle_mesh(in_ply)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if len(counts):
        keep = counts >= max(min_cluster_frac * counts.max(), 50)
        remove = ~keep[labels]
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()

    mesh = mesh.filter_smooth_taubin(number_of_iterations=8)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_ply, mesh)
    print(f"[mesh] cleaned -> {out_ply}: {len(mesh.vertices)} verts, "
          f"{len(mesh.triangles)} tris")
    return out_ply
