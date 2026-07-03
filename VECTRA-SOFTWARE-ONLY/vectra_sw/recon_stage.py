"""Stage 1 of the app job: preprocess + VGGT reconstruction (the PyTorch stage).
Run as its own process so torch's OpenMP never shares a process with Open3D/COLMAP.

    python -m vectra_sw.recon_stage "data/V1 Pre TX" outputs/V1_Pre_TX
"""
from __future__ import annotations
import os, sys

from . import preprocess, vggt_recon


def run(visit_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    pre = preprocess.run(visit_dir, out_dir)
    # use the clean basename list (avoids iCloud "name 3.JPG" duplicate copies)
    paths = [os.path.join(pre["work_dir"], n) for n in pre["images"]
             if " " not in n and os.path.exists(os.path.join(pre["work_dir"], n))]
    print(f"[recon] reconstructing {len(paths)} views", flush=True)
    vggt_recon.reconstruct(paths, out_dir)
    print("[recon] done", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
