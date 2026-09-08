#!/usr/bin/env python3
"""Overlay the filler regions on a frontal image using MediaPipe landmarks.

    python tools/render_face_regions.py photo.jpg out.png [--landmarks lm.json] [--labels]

Runs the pipeline's landmark CLI (VECTRA-SOFTWARE-ONLY/.venv-mp) unless a JSON with
"landmarks" is given. Writes the overlay PNG and prints per-region pixel areas.
"""
import argparse, json, os, subprocess, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vectra3d.face_regions import REGIONS, region_polygons  # noqa: E402
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MP_PYTHON = os.environ.get("VECTRA_MP_PYTHON", os.path.join(_ROOT, "VECTRA-SOFTWARE-ONLY", ".venv-mp", "bin", "python"))
LANDMARK_SCRIPT = os.environ.get("VECTRA_LANDMARK_SCRIPT", os.path.join(_ROOT, "VECTRA-SOFTWARE-ONLY", "tools", "landmark_detect.py"))

CLASS_RGB = {"P": (76, 106, 156), "D": (46, 138, 122), "T": (201, 138, 27),
             "M": (184, 80, 106), "N": (124, 129, 144)}


def detect(image_path):
    proc = subprocess.run([MP_PYTHON, LANDMARK_SCRIPT, image_path, "--whole-image-only"],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        sys.exit("landmark detector failed:\n" + proc.stderr[-800:])
    d = json.loads(proc.stdout.strip().splitlines()[-1])
    if not d.get("ok"):
        sys.exit("no face detected")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("out")
    ap.add_argument("--landmarks"); ap.add_argument("--labels", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.38)
    a = ap.parse_args()
    d = json.load(open(a.landmarks)) if a.landmarks else detect(a.image)
    L = np.asarray(d["landmarks"], float)
    im = Image.open(a.image).convert("RGB")
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(over)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    polys = region_polygons(L)
    for rid, ps in polys.items():
        rgb = CLASS_RGB[REGIONS[rid]["cls"]]
        for p in ps:
            pts = [tuple(x) for x in p[:, :2]]
            dr.polygon(pts, fill=rgb + (int(255 * a.alpha),), outline=(30, 34, 48, 200))
            if a.labels:
                c = p[:, :2].mean(0)
                dr.text((c[0], c[1]), rid, fill=(20, 20, 20, 255), font=font, anchor="mm")
    out = Image.alpha_composite(im.convert("RGBA"), over).convert("RGB")
    # TrueDepth frames are stored in sensor orientation; stand the face upright
    # for display using the forehead->chin axis (landmarks 10 -> 152).
    dx, dy = L[152, :2] - L[10, :2]
    if abs(dx) > abs(dy):
        out = out.rotate(-90 if dx > 0 else 90, expand=True)
    out.save(a.out)
    from vectra3d.face_regions import region_masks
    masks = region_masks(L, (im.height, im.width))
    for rid, m in masks.items():
        print(f"{rid:15s} {int(m.sum()):8d} px")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
