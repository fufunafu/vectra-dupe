"""Stage 6 - build a side-by-side sheet: VECTRA's render screenshots (top) vs our
reconstruction's canonical renders (bottom)."""
from __future__ import annotations
import glob, os
import cv2
import numpy as np


def _tile(paths, h):
    tiles = []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        s = h / im.shape[0]
        tiles.append(cv2.resize(im, (round(im.shape[1] * s), h)))
    if not tiles:
        return None
    return np.hstack(tiles)


def build(visit_dir: str, render_paths: list[str], out_path: str, row_h: int = 360):
    vectra = sorted(glob.glob(os.path.join(visit_dir, "2025*.jpg")))
    top = _tile(vectra, row_h)
    bot = _tile(render_paths, row_h)
    if top is None or bot is None:
        raise RuntimeError("missing renders for comparison")
    w = max(top.shape[1], bot.shape[1])
    def pad(x):
        c = np.zeros((x.shape[0], w, 3), np.uint8)
        c[:, :x.shape[1]] = x
        return c
    label_h = 28
    def band(text, width):
        b = np.full((label_h, width, 3), 40, np.uint8)
        cv2.putText(b, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return b
    sheet = np.vstack([band("VECTRA (reference)", w), pad(top),
                       band("OURS (classical SfM + learned dense)", w), pad(bot)])
    cv2.imwrite(out_path, sheet)
    print(f"[compare] wrote {out_path}")
    return out_path
