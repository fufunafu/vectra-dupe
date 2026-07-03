# Vectra-dupe

Phone-based reproduction of the VECTRA M3 workflow: guided 3-pose TrueDepth
face capture → 3D reconstruction → before/after registration → volumetric
difference with a color heatmap, in milliliters.

## Components

| Path | What it is | Status |
|---|---|---|
| `vectra3d/` | Core pipeline: session format, fusion, registration, bias correction, auto change-region detection | tested (synthetic ground truth) |
| `phase0/` | Feasibility study + accuracy/noise-floor evaluation | **GO**: 0.8 mL bump measured at +1.4% bias, σ ≈ 0.2 mL |
| `server/` | FastAPI processing server + three.js web viewer | e2e-tested over HTTP |
| `ios-app/` | SwiftUI + ARKit guided capture app (XcodeGen project) | source complete; needs Xcode + iPhone to build |
| `tools/compare_cli.py` | Compare two captures/meshes from the CLI | tested |
| `tests/e2e_test.py` | Full-loop integration test | passing |

## Quick start

```bash
# one-time setup
python3.12 -m venv .venv
.venv/bin/pip install open3d numpy scipy matplotlib fastapi "uvicorn[standard]" python-multipart requests

# prove the pipeline end to end (synthetic captures, ~2 min)
.venv/bin/python tests/e2e_test.py

# run the server + web viewer
cd server && ../.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8008
# open http://localhost:8008  (patients, sessions, 3D heatmaps)
```

Capture real faces with the iOS app (`ios-app/README.md`), pointed at the
server's LAN address. Sessions upload in the shared
`vectra-dupe-session/1` format (`vectra3d/io_session.py` is the contract).

## How it works (and what made it accurate)

1. **Metric scale for free** — the iPhone TrueDepth sensor returns depth in
   real millimeters, replacing the VECTRA's calibrated stereo rig.
2. **Guided capture** — eye-line/midline overlay (the VECTRA's two
   alignment lines), auto-shutter on pose+distance+expression gates, 8-frame
   depth burst averaging per pose.
3. **Fusion** — per-view ICP refinement of ARKit pose drift, then TSDF
   integration of the three views.
4. **Stable-region registration** — before/after alignment excludes detected
   change regions, otherwise ICP absorbs the very change being measured.
5. **Bias-field subtraction** — smooth systematic error (sensor low-frequency
   + residual registration) is kernel-averaged from stable areas and
   interpolated across the treated region. Without this, phantom volumes of
   ±0.5 mL appear; with it the null test reads ~0.1–0.2 mL.
6. **Auto region detection** — smoothed signed-distance threshold + connected
   components, geometry frozen between detection and measurement passes,
   scan boundary masked. Regions under 0.2 mL are flagged "below noise
   floor" instead of hidden.

Validated numbers (synthetic TrueDepth-class noise, 8 seeds): a known
0.814 mL cheek bump measured 0.825 ± 0.148 mL; repeat-scan phantom volume
+0.12 ± 0.12 mL; surface RMS ~0.23 mm. See `phase0/README.md` for the
noise model and `PLAN.md` for the roadmap and validation protocol against
the real VECTRA exports in `Vectra-files/`.

## Honest limitations

- iOS-only capture (Android lacks a universal depth sensor).
- Expression/jaw drift between sessions is the dominant real-world error;
  the capture gates help but real-face validation (Phase 4) is pending.
- Visualization/tracking aid only — not a medical device.
