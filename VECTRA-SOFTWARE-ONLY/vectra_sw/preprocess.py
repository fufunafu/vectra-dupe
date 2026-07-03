"""Stage 1 - ingest clinical photos: drop the slate/ID board photo, downscale to a
working resolution, and build a head mask off the black studio backdrop."""
from __future__ import annotations
import glob, os, shutil
import cv2
import numpy as np

MAX_DIM = 2000          # working resolution (long edge) for SfM + dense
MASK_THRESH = 18        # luminance above this = foreground (bg is near-black)


def _whiteboard_score(img: np.ndarray) -> float:
    """Area fraction of the largest bright, desaturated, solid landscape rectangle ->
    spikes for the slate/whiteboard photo, ~0 for faces."""
    small = cv2.resize(img, (512, 341))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 175) & (hsv[..., 1] < 40)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    best = 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if a > 700 and a / (bw * bh + 1e-6) > 0.75 and 0.9 < bw / (bh + 1e-6) < 2.8:
            best = max(best, a)
    return best / (512 * 341)


def find_photos(visit_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(visit_dir, "IMG_*.JPG")) +
                  glob.glob(os.path.join(visit_dir, "IMG_*.jpg")))


def head_mask(img: np.ndarray) -> np.ndarray:
    """Largest bright connected component (the head), holes filled."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fg = (gray > MASK_THRESH).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return fg * 255
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lab == biggest).astype(np.uint8) * 255
    # fill internal holes
    ff = mask.copy()
    h, w = mask.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    mask = mask | cv2.bitwise_not(ff)
    # pull the boundary in: the outermost pixels mix backdrop and subject, which
    # poisons silhouette features and texture projection
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return mask


def run(visit_dir: str, out_dir: str, max_dim: int = MAX_DIM) -> dict:
    work_dir = os.path.join(out_dir, "work")
    mask_dir = os.path.join(out_dir, "masks")
    for d in (work_dir, mask_dir):
        shutil.rmtree(d, ignore_errors=True)   # avoid stale files across runs
        os.makedirs(d, exist_ok=True)

    photos = find_photos(visit_dir)
    if not photos:
        raise RuntimeError(f"no IMG_*.JPG in {visit_dir}")

    # detect + drop the slate photo (highest whiteboard score, if clearly a slate)
    scores = [(_whiteboard_score(cv2.imread(p)), p) for p in photos]
    scores.sort(reverse=True)
    dropped = []
    if scores and scores[0][0] > 0.01:
        dropped.append(os.path.basename(scores[0][1]))
        photos = [p for p in photos if os.path.basename(p) != dropped[0]]

    kept = []
    for p in photos:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        s = max_dim / max(h, w)
        if s < 1.0:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        name = os.path.basename(p)
        cv2.imwrite(os.path.join(work_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(mask_dir, name + ".png"), head_mask(img))
        kept.append(name)

    print(f"[preprocess] kept {len(kept)} photos, dropped slate {dropped or 'none'}, "
          f"work res long-edge={max_dim}")
    return {"work_dir": work_dir, "mask_dir": mask_dir, "images": kept, "dropped": dropped}


if __name__ == "__main__":
    import sys
    run(sys.argv[1], sys.argv[2])
