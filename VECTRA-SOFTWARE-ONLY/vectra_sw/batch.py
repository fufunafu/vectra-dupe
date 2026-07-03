"""Process every visit and collect the quality trend.

Runs the three pipeline stages as subprocesses (same OpenMP isolation as the
app) plus the quality gate, for each data/<visit>/ directory:

    .venv/bin/python -m vectra_sw.batch            # all visits
    .venv/bin/python -m vectra_sw.batch "V1 Pre TX" "V1 Post TX"   # subset

Writes outputs/quality_trend.csv (one row per visit).
"""
from __future__ import annotations
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUTPUTS = os.path.join(ROOT, "outputs")
PYEXE = sys.executable


def _stage(mod: str, *args: str) -> None:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    subprocess.run([PYEXE, "-m", f"vectra_sw.{mod}", *args],
                   cwd=ROOT, env=env, check=True)


def run_visit(visit: str, skip_recon_if_done: bool = True) -> dict:
    visit_dir = os.path.join(DATA, visit)
    out_dir = os.path.join(OUTPUTS, visit.replace(" ", "_"))
    t0 = time.time()
    perview = os.path.join(out_dir, "vggt_perview.npz")
    have_recon = os.path.exists(perview)
    if have_recon and skip_recon_if_done:
        # only trust a perview file new enough to carry the `valid` mask
        import numpy as np
        have_recon = "valid" in np.load(perview)
    if not have_recon:
        _stage("recon_stage", visit_dir, out_dir)
    _stage("finish_vggt", visit_dir, out_dir)
    _stage("metric", out_dir)
    _stage("quality", visit_dir, out_dir)
    with open(os.path.join(out_dir, "quality_report.json")) as f:
        q = json.load(f)
    with open(os.path.join(out_dir, "measurements.json")) as f:
        m = json.load(f)
    return {"visit": visit, "seconds": round(time.time() - t0),
            "mean_iou": q["mean_iou"], "min_iou": q["min_iou"],
            "mean_ssim": q["mean_ssim"],
            "verts": q["mesh"]["vertices"],
            "boundary_edges": q["mesh"]["boundary_edges"],
            "face_width_mm": m["measures_mm"]["bizygomatic_face_width"],
            "face_height_mm": m["measures_mm"]["morphological_face_height"]}


def main() -> int:
    visits = sys.argv[1:] or sorted(
        d for d in os.listdir(DATA)
        if os.path.isdir(os.path.join(DATA, d))
        and any(f.upper().endswith(".JPG") for f in os.listdir(os.path.join(DATA, d))))
    rows = []
    for v in visits:
        print(f"=== {v} ===", flush=True)
        try:
            rows.append(run_visit(v))
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {v}: {e}", flush=True)
            rows.append({"visit": v, "error": str(e)})
    os.makedirs(OUTPUTS, exist_ok=True)
    keys = ["visit", "seconds", "mean_iou", "min_iou", "mean_ssim", "verts",
            "boundary_edges", "face_width_mm", "face_height_mm", "error"]
    out_csv = os.path.join(OUTPUTS, "quality_trend.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")
    for r in rows:
        print(r)
    return 0 if all("error" not in r for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
