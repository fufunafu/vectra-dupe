"""Compare two captures from the command line (Phase 4 validation tool).

Each input is either a capture session directory (vectra-dupe-session/1,
fused on the fly) or an already-fused mesh (.ply/.obj). Writes heatmap.ply,
heatmap.png and result.json to the output directory and prints the regions.

Usage:
  .venv/bin/python tools/compare_cli.py BEFORE AFTER [-o out_dir]

Examples:
  # two sessions captured with the iOS app, copied off the phone
  .venv/bin/python tools/compare_cli.py captures/pre-tx captures/post-tx -o results/

  # meshes exported from elsewhere (e.g. converted VECTRA exports)
  .venv/bin/python tools/compare_cli.py pre.ply post.ply -o results/
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import open3d as o3d  # noqa: E402

from server import processing  # noqa: E402
from vectra3d import fuse, io_session  # noqa: E402


def load_input(path: str) -> str:
    """Return a mesh file path for the input, fusing sessions if needed."""
    if os.path.isdir(path):
        poses, _, _ = io_session.load_session(path)
        mesh = fuse.fuse_session(poses)
        mesh = processing.normalize_to_front_frame(mesh, poses[0].world_to_camera)
        out = os.path.join(path, "mesh.ply")
        o3d.io.write_triangle_mesh(out, mesh)
        print(f"fused {path} -> {out} ({len(mesh.vertices)} vertices)")
        return out
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("-o", "--out", default="compare-out")
    args = ap.parse_args()

    summary = processing.compare_sessions_on_disk(
        load_input(args.before), load_input(args.after), args.out)

    print(f"\nartifacts in {args.out}/  (heatmap.ply, heatmap.png, result.json)")
    print(f"surface RMS: {summary['surface_rms_mm']} mm")
    if not summary["regions"]:
        print("no volume change detected")
    for r in summary["regions"]:
        flag = "" if r["significant"] else "  (below noise floor)"
        print(f"  {r['volume_ml']:+.3f} mL  peak {r['peak_mm']:+.2f} mm  "
              f"at {r['center']}{flag}")
    print(json.dumps({"net_significant_volume_ml":
                      summary["net_significant_volume_ml"]}))


if __name__ == "__main__":
    main()
