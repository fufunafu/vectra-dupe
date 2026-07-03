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
        # TSDF-fuse the per-view depth maps (ghost-free), then sample the surface
        # densely for a solid, gap-free splat render. Falls back to alpha-shape
        # over the point soup if the per-view data is missing (old runs).
        mesh = _mesh_from_perview(out_dir, mesh_path)
        if mesh is None:
            mesh = meshing.build_mesh(pts, cols, mesh_path)
        samples = _bake_texture(visit_dir, out_dir, mesh, extr)
        if samples is not None:
            Vt, Nt, Ct = samples          # photo-projected texture colors
            renders = render_views.render_arrays(Vt, Nt, Ct, extr, out_dir,
                                                 gain=1.0)
        else:
            V, N, C = meshing.dense_surface_points(mesh)
            renders = render_views.render_arrays(V, N, C, extr, out_dir)
        _save_canonical(mesh, extr, mesh_path)   # upright mesh for the web viewer
    else:
        renders = render_views.render_point_cloud(pts, cols, extr, out_dir)
    compare.build(visit_dir, renders, os.path.join(out_dir, "comparison.png"))
    print(f"DONE -> {out_dir}/mesh.ply, comparison.png", flush=True)


def _bake_texture(visit_dir: str, out_dir: str, mesh, extr):
    """Project the original photos onto a UV atlas (texture.bake); returns dense
    textured surface samples for rendering, or None if baking isn't possible."""
    import glob
    p = os.path.join(out_dir, "vggt_perview.npz")
    if not os.path.exists(p):
        return None
    try:
        from . import texture
        pv = np.load(p)
        names = [str(n) for n in pv["names"]]
        originals = {os.path.basename(x): x
                     for x in glob.glob(os.path.join(visit_dir, "IMG_*.JPG"))
                     + glob.glob(os.path.join(visit_dir, "IMG_*.jpg"))}
        photo_paths = {n: originals[n] for n in names if n in originals}
        if len(photo_paths) < 3:
            print("[texture] bake skipped: originals not found", flush=True)
            return None
        B = render_views.canonical_basis(extr)
        _, samples = texture.bake(
            mesh, names, pv["intrinsic"], pv["extrinsic"],
            tuple(pv["world_points"].shape[1:3]), photo_paths,
            os.path.join(out_dir, "masks"),
            os.path.join(out_dir, "mesh_textured.glb"), canonical=B)
        return samples
    except Exception as e:  # noqa: BLE001
        print(f"[texture] bake skipped: {e}", flush=True)
        return None


def _mesh_from_perview(out_dir: str, mesh_path: str):
    """TSDF mesh from vggt_perview.npz, or None if it lacks the valid mask
    (produced by older runs) so the caller can fall back."""
    p = os.path.join(out_dir, "vggt_perview.npz")
    if not os.path.exists(p):
        return None
    pv = np.load(p)
    if "valid" not in pv:
        return None
    try:
        return meshing.build_mesh_tsdf(pv["world_points"], pv["valid"],
                                       pv["images"], pv["extrinsic"],
                                       pv["intrinsic"], mesh_path)
    except Exception as e:  # noqa: BLE001
        print(f"[mesh] tsdf failed ({e}); falling back to alpha-shape", flush=True)
        return None


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
