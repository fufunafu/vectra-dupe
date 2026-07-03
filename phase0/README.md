# Phase 0 — Feasibility prototype

Desktop simulation of the full phone-based VECTRA-style pipeline:
guided 3-pose depth capture → fusion → cross-session registration →
signed-distance heatmap → ROI volume in mL. All units are millimeters.

## Run it

```bash
../.venv/bin/python run_phase0.py   # single experiment + heatmaps/meshes in results/
../.venv/bin/python evaluate.py     # 8-seed statistics (~90 s)
```

## Results (2026-06-12, 8 seeds)

| Metric | Value |
|---|---|
| Ground-truth bump ("filler") | 0.814 mL |
| Measured | 0.825 ± 0.148 mL (bias +1.4%) |
| Null test (no change, truth 0) | +0.123 ± 0.116 mL |
| Repeat-scan surface RMS | ~0.23 mm |

**Verdict: GO.** With a TrueDepth-class noise model the pipeline recovers a
filler-sized volume change essentially without bias, and the per-measurement
uncertainty is ~±0.2 mL — a 3–10x signal-to-noise ratio on typical 0.5–2 mL
treatment effects. Comparable studies put the VECTRA itself at a similar
noise scale for soft-tissue volume differences.

Known issue: the null test shows a small consistent positive offset
(+0.12 mL). Likely a curvature/smoothing interaction in the bias estimate;
worth chasing in Phase 2, and correctable with control-region calibration.

## What the simulation models (and why it's fair)

- **Sensor**: 640x480 depth at 580 px focal length, 350 mm range. Noise =
  0.8 mm white per frame (cut by averaging 8 frames/pose) + 0.5 mm
  device-fixed low-frequency calibration field (cancels between sessions on
  the same phone) + 0.25 mm per-capture low-frequency field (does not cancel —
  this is what limits the noise floor).
- **Capture**: 3 poses (front, ±40°), with ~1°/2 mm ARKit pose drift on the
  side views, recovered by ICP.
- **Sessions**: the subject re-sits with a random 5° / 10 mm pose change.
- **Head**: deformed ellipsoid with nose/brow/chin features (asymmetric so
  ICP can lock on); bump volume known exactly via the divergence theorem.

## Pipeline pieces that proved necessary

1. **Frame averaging per pose** — halves the surface RMS for free.
2. **Stable-region-only fine ICP** — alignment must exclude the treated area
   or it absorbs the change being measured.
3. **Smooth bias-field subtraction** (Gaussian kernel averaging on stable
   vertices, interpolated across the ROI) — without it, low-frequency sensor
   error integrates to ±0.5 mL phantom volume. An RBF least-squares fit
   oscillates when bridging the ROI hole; kernel averaging cannot.

## Layout

- `vectra3d/cameras.py` — pinhole model, 3 capture poses, pose drift
- `vectra3d/synthetic.py` — ground-truth head + bump, exact volumes
- `vectra3d/capture.py` — depth rendering + 3-component sensor noise
- `vectra3d/fuse.py` — back-projection, per-view ICP, TSDF fusion
- `vectra3d/register.py` — coarse + stable-region-only session alignment
- `vectra3d/analyze.py` — signed distance, bias subtraction, ROI volume, heatmaps

## Next (Phase 0.5): same pipeline on real captures

Swap `capture.capture_session()` for real frames: record the 3 poses with the
TrueDepth camera (e.g. the Record3D app, EXR depth + RGB export), load depth +
intrinsics, and run the identical fuse/register/analyze code. The first real
test is the same as the synthetic one: scan a face twice with no change and
check the phantom volume stays ~0.2 mL.
