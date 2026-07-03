"""Plan 1 orchestrator: photos -> textured 3D head, compared against VECTRA's render.

Usage:
    python -m vectra_sw.pipeline "data/V1 Pre TX" outputs/V1_Pre_TX
"""
from __future__ import annotations
import os, sys, time

# NOTE: do NOT import dense/mesh/render_views at module load. They pull in PyTorch /
# Open3D, whose OpenMP runtime deadlocks COLMAP's multi-threaded matcher if loaded in
# the same process before SfM runs. Import them lazily, after SfM completes.
from . import preprocess, sfm, compare


def run(visit_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    pre = preprocess.run(visit_dir, out_dir)

    import cv2
    work_w = cv2.imread(os.path.join(pre["work_dir"], pre["images"][0])).shape[1]
    focal = sfm.focal_px_from_exif(os.path.join(visit_dir, pre["images"][0]), work_w)

    sfm.run(pre["work_dir"], out_dir, mask_dir=pre["mask_dir"], focal_px=focal)
    rec = sfm.load(os.path.join(out_dir, "sparse"))
    views = sfm.camera_views(rec)

    from . import dense, mesh, render_views   # lazy: loads torch/open3d only now
    raw = dense.fuse(views, pre["work_dir"], pre["mask_dir"], out_dir)
    clean_ply = mesh.clean(raw, os.path.join(out_dir, "mesh.ply"))

    extr = [v["extrinsic"] for v in views]
    renders = render_views.render(clean_ply, extr, out_dir)
    compare.build(visit_dir, renders, os.path.join(out_dir, "comparison.png"))

    print(f"\nDONE in {time.time()-t0:.0f}s -> {out_dir}/mesh.ply, comparison.png")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
