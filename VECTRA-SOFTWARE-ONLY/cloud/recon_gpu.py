"""Cloud GPU reconstruction: VGGT poses + COLMAP CUDA multi-view stereo.

Runs on Modal (https://modal.com). The Mac uploads the preprocessed work images
(+ head masks); an L4 GPU box runs VGGT for camera poses (exported in COLMAP
format via VGGT's own demo_colmap.py), then COLMAP patch-match stereo with
geometric consistency and stereo fusion at full working resolution — the dense,
cross-view-consistent geometry the local CPU VGGT path can't reach. Poisson
meshing runs here too (Open3D's Poisson is broken on macOS but fine on Linux).

One-time setup on the Mac:
    python -m pip install modal
    python -m modal setup            # browser auth

Run for one visit (from VECTRA-SOFTWARE-ONLY/):
    modal run cloud/recon_gpu.py --visit V1_Pre_TX

Inputs:  outputs/<visit>/work/*.JPG (2000px), outputs/<visit>/masks/*.JPG.png
Outputs: outputs/<visit>/cloud/{fused.ply, mesh_raw.ply, cameras.npz,
         depths.npz, sparse/}  (~$0.30-1.00 per visit on an L4)
"""
from __future__ import annotations

import io
import os
import tarfile

import modal

app = modal.App("vectra-recon-gpu")

# COLMAP's official image ships CUDA-enabled colmap; add python + torch + VGGT.
image = (
    modal.Image.from_registry("colmap/colmap:latest", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "numpy", "pillow", "opencv-python-headless",
        "open3d", "pycolmap", "trimesh", "huggingface_hub", "einops",
        "safetensors", "scipy",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/facebookresearch/vggt /opt/vggt",
        "python -m pip install -e /opt/vggt",
    )
    # demo_colmap.py's track-refinement deps
    .pip_install("git+https://github.com/cvg/LightGlue.git")
    .pip_install("hydra-core", "omegaconf")
    .run_commands("bash -c 'for f in /opt/vggt/requirements*.txt; do "
                  "python -m pip install -r $f || true; done'")
)


def _tar_bytes(paths: dict[str, str]) -> bytes:
    """{arcname: local_path} -> tar bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for arc, p in paths.items():
            tf.add(p, arcname=arc)
    return buf.getvalue()


def _untar(data: bytes, dest: str) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(dest)  # noqa: S202 (trusted archive, our own round-trip)


@app.function(image=image, gpu="L40S", timeout=3600)
def reconstruct(scene_tar: bytes, poisson_depth: int = 10,
                fusion_min_pixels: int = 3) -> bytes:
    import json
    import shutil
    import subprocess

    import numpy as np

    scene = "/scene"
    shutil.rmtree(scene, ignore_errors=True)
    os.makedirs(scene)
    _untar(scene_tar, scene)
    img_dir = os.path.join(scene, "images")
    mask_dir = os.path.join(scene, "masks")
    n_img = len(os.listdir(img_dir))
    print(f"[cloud] {n_img} images", flush=True)

    # --- 1) VGGT -> COLMAP sparse model. BA (LightGlue tracks) gives the
    # covisibility patch-match needs; fall back to feed-forward poses if BA
    # can't build a model.
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    base = ["python", "/opt/vggt/demo_colmap.py", "--scene_dir", scene]
    for cmd in (base + ["--use_ba", "--shared_camera"], base):
        print(f"[cloud] vggt poses: {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd, env=env, cwd="/opt/vggt").returncode == 0:
            break
    else:
        raise RuntimeError("demo_colmap.py failed with and without BA")
    sparse = os.path.join(scene, "sparse")
    if not os.path.isdir(sparse):
        raise RuntimeError("demo_colmap.py produced no sparse/ model")

    # --- 2) COLMAP dense: undistort, patch-match stereo (geometric), fuse.
    dense = os.path.join(scene, "dense")
    subprocess.run(["colmap", "image_undistorter", "--image_path", img_dir,
                    "--input_path", sparse, "--output_path", dense,
                    "--output_type", "COLMAP"], check=True)
    # Explicit stereo sources by CAMERA GEOMETRY: each reference gets the 8
    # views whose optical axes are angularly closest (filename order breaks at
    # the sequence's angle jumps — profiles/up/down shots got near-empty depth).
    # Sidesteps covisibility-based selection (empty for a track-less model).
    import pycolmap
    rec0 = pycolmap.Reconstruction(os.path.join(dense, "sparse"))
    cams_axis, names_ordered = [], []
    for _, im in sorted(rec0.images.items(), key=lambda kv: kv[1].name):
        R = im.cam_from_world.matrix()[:3, :3]
        cams_axis.append(R[2])          # viewing direction in world coords
        names_ordered.append(im.name)
    import numpy as np
    A = np.asarray(cams_axis)
    dots = A @ A.T
    with open(os.path.join(dense, "stereo", "patch-match.cfg"), "w") as f:
        for i, name in enumerate(names_ordered):
            order = np.argsort(-dots[i])
            nbrs = [names_ordered[j] for j in order if j != i][:8]
            f.write(f"{name}\n{', '.join(nbrs)}\n")
    subprocess.run(["colmap", "patch_match_stereo", "--workspace_path", dense,
                    "--workspace_format", "COLMAP",
                    "--PatchMatchStereo.geom_consistency", "true"], check=True)
    fused = os.path.join(dense, "fused.ply")

    def run_fusion(input_type: str, extra: list[str]) -> int:
        cmd = ["colmap", "stereo_fusion", "--workspace_path", dense,
               "--workspace_format", "COLMAP", "--input_type", input_type,
               "--output_path", fused,
               "--StereoFusion.min_num_pixels", str(fusion_min_pixels), *extra]
        if os.path.isdir(mask_dir):
            cmd += ["--StereoFusion.mask_path", mask_dir]
        subprocess.run(cmd, check=True)
        n = 0
        if os.path.isfile(fused):
            import open3d as o3d_
            n = len(o3d_.io.read_point_cloud(fused).points)
        print(f"[cloud] fusion ({input_type}): {n} points", flush=True)
        return n

    # geometric first; if the pose network wasn't consistent enough for the
    # strict defaults, retry photometric with relaxed agreement thresholds
    # (noisier, but Poisson + the density trim absorb it).
    if run_fusion("geometric", ["--StereoFusion.max_reproj_error", "3",
                                "--StereoFusion.max_depth_error", "0.02"]) < 10_000:
        run_fusion("photometric", ["--StereoFusion.max_reproj_error", "4",
                                   "--StereoFusion.max_depth_error", "0.05",
                                   "--StereoFusion.max_normal_error", "25"])

    # --- 3) Poisson mesh (works on Linux) + density trim + largest component.
    import open3d as o3d
    pc = o3d.io.read_point_cloud(fused)
    print(f"[cloud] fused cloud: {len(pc.points)} points", flush=True)
    if len(pc.points) < 10_000:
        raise RuntimeError(
            f"stereo fusion produced only {len(pc.points)} points — "
            "check patch-match sources/masks in the log above")
    if not pc.has_normals():
        pc.estimate_normals()
        pc.orient_normals_consistent_tangent_plane(30)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=poisson_depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.05))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    tl, cnt, _ = mesh.cluster_connected_triangles()
    tl, cnt = np.asarray(tl), np.asarray(cnt)
    if len(cnt):
        mesh.remove_triangles_by_mask(np.asarray(cnt)[tl] < cnt.max())
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    # colour from the fused cloud (Poisson drops colours)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(pc.points))
    _, idx = tree.query(np.asarray(mesh.vertices), k=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pc.colors)[idx])
    mesh_path = os.path.join(scene, "mesh_raw.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"[cloud] poisson mesh: {len(mesh.vertices)} verts", flush=True)

    # --- 4) cameras.npz (undistorted model) + per-view geometric depth maps.
    import pycolmap
    rec = pycolmap.Reconstruction(os.path.join(dense, "sparse"))
    names, K_all, w2c_all = [], [], []
    for _, im in sorted(rec.images.items(), key=lambda kv: kv[1].name):
        cam = rec.cameras[im.camera_id]
        K = cam.calibration_matrix()
        w2c = np.eye(4)
        w2c[:3, :] = im.cam_from_world.matrix()
        names.append(im.name)
        K_all.append(K)
        w2c_all.append(w2c)
    np.savez(os.path.join(scene, "cameras.npz"),
             names=np.array(names), K=np.array(K_all),
             world_to_cam=np.array(w2c_all))

    depth_dir = os.path.join(dense, "stereo", "depth_maps")
    depths, dnames = [], []
    for name in names:
        p = os.path.join(depth_dir, f"{name}.geometric.bin")
        if not os.path.isfile(p):
            continue
        with open(p, "rb") as f:                      # COLMAP .bin depth format
            hdr = b""
            for _ in range(3):
                while not hdr.endswith(b"&"):
                    hdr += f.read(1)
                hdr += b"|"
            w, h, c = [int(x) for x in hdr.replace(b"|", b"").split(b"&")[:3]]
            d = np.fromfile(f, np.float32, w * h * c).reshape(h, w, c)[:, :, 0]
        depths.append(d[::2, ::2].astype(np.float16))  # half res is plenty
        dnames.append(name)
    np.savez_compressed(os.path.join(scene, "depths.npz"),
                        names=np.array(dnames),
                        depths=np.array(depths, dtype=object) if depths else [],
                        allow_pickle=True)

    out = {
        "fused.ply": fused,
        "mesh_raw.ply": mesh_path,
        "cameras.npz": os.path.join(scene, "cameras.npz"),
        "depths.npz": os.path.join(scene, "depths.npz"),
        "sparse": sparse,
    }
    stats = {"images": n_img, "fused_points": len(pc.points),
             "mesh_verts": len(mesh.vertices)}
    with open(os.path.join(scene, "cloud_stats.json"), "w") as f:
        json.dump(stats, f)
    out["cloud_stats.json"] = os.path.join(scene, "cloud_stats.json")
    return _tar_bytes(out)


@app.local_entrypoint()
def main(visit: str, poisson_depth: int = 10):
    """modal run cloud/recon_gpu.py --visit V1_Pre_TX"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "outputs", visit)
    work, masks = os.path.join(out_dir, "work"), os.path.join(out_dir, "masks")
    if not os.path.isdir(work):
        raise SystemExit(f"no work images at {work} — run the preprocess stage first")

    paths = {f"images/{n}": os.path.join(work, n)
             for n in sorted(os.listdir(work)) if " " not in n}
    if os.path.isdir(masks):
        paths.update({f"masks/{n}": os.path.join(masks, n)
                      for n in sorted(os.listdir(masks)) if " " not in n})
    print(f"[local] uploading {len(paths)} files from {out_dir}")
    result = reconstruct.remote(_tar_bytes(paths), poisson_depth=poisson_depth)

    cloud_dir = os.path.join(out_dir, "cloud")
    os.makedirs(cloud_dir, exist_ok=True)
    _untar(result, cloud_dir)
    print(f"[local] results -> {cloud_dir}")
