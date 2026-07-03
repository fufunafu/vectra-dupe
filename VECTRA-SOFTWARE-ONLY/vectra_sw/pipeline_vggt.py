"""Plan-1 (revised) orchestrator: learned multi-view reconstruction with VGGT.

    python -m vectra_sw.pipeline_vggt "data/V1 Pre TX" outputs/V1_Pre_TX
"""
from __future__ import annotations
import os, sys, time, glob

from . import preprocess, compare


def run(visit_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    pre = preprocess.run(visit_dir, out_dir)
    paths = [os.path.join(pre["work_dir"], n) for n in pre["images"]]

    from . import vggt_recon, render_views   # lazy (torch/open3d)
    r = vggt_recon.reconstruct(paths, out_dir)
    ply = vggt_recon.to_mesh(r["points"], r["colors"],
                             os.path.join(out_dir, "mesh.ply"))

    renders = render_views.render(ply, r["extrinsics"], out_dir)
    compare.build(visit_dir, renders, os.path.join(out_dir, "comparison.png"))
    print(f"\nDONE in {time.time()-t0:.0f}s -> {out_dir}/mesh.ply, comparison.png",
          flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
