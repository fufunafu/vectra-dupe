"""Mesh + render + compare from a saved VGGT point cloud (vggt_raw.npz).

Runs in a torch-free process so Open3D's OpenMP-based Poisson reconstruction does not
deadlock against PyTorch's libomp. See pipeline_vggt for the full two-stage flow.
"""
from __future__ import annotations
import os, sys
import numpy as np

from . import meshing, render_views, compare


def run(visit_dir: str, out_dir: str, mesh_fill: bool = True):
    d = np.load(os.path.join(out_dir, "vggt_raw.npz"))
    pts, cols = d["points"], d["colors"]
    extr = [np.vstack([e, [0, 0, 0, 1]]) for e in d["extrinsic"]]

    mesh_path = os.path.join(out_dir, "mesh.ply")
    if mesh_fill:
        # alpha-shape mesh fills the hair-streak holes; sample its surface densely
        # for a solid, gap-free splat render.
        mesh = meshing.build_mesh(pts, cols, mesh_path)
        V, N, C = meshing.dense_surface_points(mesh)
        renders = render_views.render_arrays(V, N, C, extr, out_dir)
        _save_canonical(mesh, extr, mesh_path)   # upright mesh for the web viewer
    else:
        renders = render_views.render_point_cloud(pts, cols, extr, out_dir)
    compare.build(visit_dir, renders, os.path.join(out_dir, "comparison.png"))
    print(f"DONE -> {out_dir}/mesh.ply, comparison.png", flush=True)


def _save_canonical(mesh, extr, path):
    """Re-save the mesh oriented to the canonical face frame (x=right, y=up,
    z=front) so the desktop viewer shows it upright and facing forward."""
    import numpy as np
    import open3d as o3d
    from .render_views import canonical_basis
    B = canonical_basis(extr)
    V = np.asarray(mesh.vertices)
    mesh.vertices = o3d.utility.Vector3dVector((V - V.mean(0)) @ B)
    mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals) @ B)
    o3d.io.write_triangle_mesh(path, mesh)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
