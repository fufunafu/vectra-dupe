# VECTRA Software Reproduction — Plan

Goal: reproduce Canfield/VECTRA's photo→3D reconstruction *software* (not the
camera hardware), then make it better. Input is the clinical photo set; the
benchmark is VECTRA's own textured 3D render screenshots.

## The data
Per visit (`data/<visit>/`):
- **~26 clinical DSLR photos** `IMG_*.JPG` (5184×3456, black backdrop): 1 slate/ID
  board + a head rotation series (front → both obliques → both profiles → up/down).
  These are the **inputs**.
- **7 render screenshots** `2025*.jpg` (1892×1108): VECTRA's textured 3D model
  rendered from canonical angles. These are the **reference output** we compare to.

7 visits of the same subject (V1 Pre/Post-TX, V2 Pre/Post, V3 wk4, V4 wk8, V5 wk20)
= a treatment time-series → enables longitudinal volume/change analysis (Plan 2).

## Hardware reality
Apple M5 Pro, 20-core GPU (Metal), 64 GB, **no CUDA**. COLMAP *sparse* SfM runs on
CPU; COLMAP *dense* is CUDA-only → use a Metal-accelerated learned dense step instead.

---

## Plan 1 — Reproduce VECTRA "the same" (classical SfM + learned dense)
Decision: scope to **V1 Pre TX** first; **classical SfM (COLMAP) + learned dense**.

1. **Preprocess** (`preprocess.py`) — drop slate photo, build head masks off the
   black background.
2. **SfM** (`sfm.py`) — COLMAP/pycolmap: camera intrinsics + poses + sparse cloud.
3. **Dense** (`dense.py`) — Depth Anything V2 per view → align each learned depth to
   COLMAP sparse points (robust affine fit in disparity space) → Open3D TSDF fusion.
4. **Mesh** (`mesh.py`) — extract mesh from TSDF, keep largest component, crop to head.
5. **Render** (`render_views.py`) — render the same canonical views as VECTRA.
6. **Compare** (`compare.py`) — side-by-side sheet: ours vs VECTRA's renders.

Deliverable: textured mesh (`outputs/<visit>/mesh.ply`) + `comparison.png`.

## Plan 2 — Make it better (after baseline)
1. Learned pose-free reconstruction (VGGT / MASt3R-SfM) for more complete geometry.
2. FLAME/3DMM face prior → hole-fill + consistent topology across all 7 visits.
3. Metric scale via interpupillary distance → real mm / mL.
4. Photoreal texture (Gaussian Splatting / delighting).
5. Longitudinal per-region volume-change maps (reuse validated ±0.2 mL noise floor).
6. Package into the existing three.js viewer.
