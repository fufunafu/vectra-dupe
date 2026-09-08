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

The local server runs one reconstruction at a time. Use one Uvicorn worker:
its FIFO queue is in memory, and duplicate requests for an active session
share the same job. Status moves from `queued` to `processing` to `done` or
`failed`; `?wait=true` waits for that same queued job. Graceful shutdown drains
the queue. After an interrupted shutdown, unfinished sessions must be submitted
again. Separate CLI processes are not coordinated by this server queue.

Object Capture keeps full detail and accepts the first result that passes the
existing quality checks. `VECTRA_OC_ATTEMPTS` (default 5) and
`VECTRA_OC_PHOTO_ATTEMPTS` (default 3) are maximum attempts for retrying failures.
Successful jobs record their actual attempt count and processing duration.

Object Capture display crops follow the detected face landmarks with a 25 mm
margin, preserving the chin even when the capture origin is offset. An uncropped
textured surface is saved as `display_uncropped.npz` for later display adjustments.
This does not change the depth mesh's measurement crop. When no facial landmarks
are available, the display uses a broader 160 mm head crop.

To rebuild an existing capture as a **new session**, keeping the original
processing result intact, run this while the server queue is idle:

```bash
.venv/bin/python tools/reprocess_copy.py PATIENT_ID SESSION_ID --label "Scan v2"
```

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

### Measurement safety screens

Processing now reports preview quality separately from measurement eligibility.
The web viewer shows `Processing`, `Unchecked`, `Quality warning`, or `Checks passed`
instead of treating every exported model as a successful measurement. The last
label means engineering screens passed, **not clinical validation**.

The central frontal depth surface must have at least 85% ray coverage and no more
than 10% rays intersecting distinct overlapping layers. Comparisons additionally
require bidirectional alignment fitness of 85% at 2 mm, inlier RMSE at most 1 mm,
and 85% area-weighted valid distance coverage globally and in each detected region.
These are conservative failure screens, not a calibrated cc detection limit.
Scans outside the expected canonical millimetre frame can also fail them.

The API blocks active jobs, failed processing and changed mesh files. Existing
legacy meshes are checked on read without changing their files or reconstructing
them again. Geometry is checked again during comparison, which returns HTTP 422
on a quality failure without writing a new volume result. Rear LiDAR and
real-patient repeatability remain unvalidated.

The optional **Experimental surface comparison** checkbox allows finished depth
scans with quality warnings to be selected. It produces an uncorrected surface
discrepancy map with the geometry/alignment warnings attached, never a cc estimate
or a zero-volume result. It does not apply intercanthal scaling. Results use a
separate `__experimental` suffix, leaving normal comparison results intact.
Active jobs, missing/changed meshes, invalid geometry, and copies of the same
capture remain blocked in both modes. Normal volume thresholds are unchanged.

Experimental mode also offers **Photo models: inner-eye scale + upper-face
alignment** for sessions with retained `display_uncropped.npz` and indexed
landmarks. This is a separate research path, not a repair or promotion of the
depth measurement mesh. It redetects landmarks on a close-up render, normalizes
both models by their inner-canthus distance, and rigidly aligns upper-face
landmarks plus a bounded forehead-only refinement. Its surface check uses
point-to-triangle distances so different tessellations do not masquerade as
alignment error. Mouth and cheek changes are excluded from alignment; no bias
field is subtracted.

Without a directly measured intercanthal distance, this path provides a
relative-scale heatmap only. With a user-entered distance in mm, it can report
exploratory signed volumes over fixed, projected left/right lower-face patches.
Any missing or layered ray in a patch withholds that patch's volume. These are
not validated clinical measurements, gauze volumes, or treatment amounts.
Results and landmark diagnostics are stored separately with a `__photo` suffix;
the original captures, depth meshes, and previous results remain unchanged.

The 3D heatmap has a **Heatmap sensitivity** slider at the bottom left. Moving
right reveals smaller differences with stronger colors; moving left reduces
contrast. Reset returns to the default +/-2.5 scale. The live legend follows
the slider, uncalibrated photo results retain approximate-reference units,
and missing samples remain gray. This is display only: no change to alignment,
volumes, region detection, or quality checks. Saved PNG maps keep their default
scale. New comparisons include a hash-bound `heatmap.display.json` containing
unclipped scalar samples. Older results without this file show a disabled slider;
rerun Compare, or use `tools/backfill_heatmap_display.py RESULT_DIRECTORY` for
a saved photo result whose sources and geometry are unchanged. Sensitivity
works on localhost, HTTPS, and plain HTTP LAN addresses. On LAN HTTP, an in-app
SHA-256 fallback verifies the same asset hashes without requiring Web Crypto.
These checks detect mismatched assets; they do not encrypt the HTTP connection.

Photo comparisons also keep the original before-scan texture under the heatmap.
**Show face texture** toggles the photo layer, and **Heatmap opacity** blends
between the untouched face (0%) and stronger change colors. Small differences
fade toward the photo; missing samples receive no tint, so an untinted area is
not proof of zero change. Turn off the photo layer to see invalid samples in
gray. Texture coordinates are transferred per triangle corner to preserve atlas
seams, with no geometry deformation or landmark movement. The original atlas is
kept at full resolution in `heatmap.texture.png`; `heatmap.uv.bin` maps it onto
the displayed comparison mesh. These patient assets stay local and out of Git.

The interactive viewer uses reference-style colors: positive distances are
blue/cyan, negative distances orange/yellow. This reverses the old color
convention, not the underlying signed data. **Contour lines** adds white
isolines at seven evenly spaced levels on each side of zero. The displayed
line interval and legend update with sensitivity. Missing samples and clipped
extremes have no contour lines; photo opacity also controls the lines. These
are surface-distance contours, not cc contours. Previously saved PNG maps keep
their original red/blue palette and default scale.

On the textured overlay, near-zero color and contour opacity smoothly fade to
transparent instead of whitening the face. The legend shows this transparency
over a checkerboard. Full tint is reached at 60% of the chosen display range;
this is a visual fade, not a measurement threshold. Plain color-map mode keeps
its white zero point, since there is no underlying photo to reveal.

For a recoverable local diagnostic run, set `VECTRA_OC_ATTEMPTS=1` and
`VECTRA_OC_DEBUG=1` when running `tools/reprocess_copy.py`. The new session retains
the OBJ, diffuse texture and landmark render in `oc_debug/`; the source is untouched.
Diagnostics contain patient imagery and must remain local and out of Git.
`tools/recover_preview.py PATIENT_ID COPY_SESSION_ID` can reuse a copy's retained
OBJ after a face-search fix. All existing alignment guards still run, and the
original session, raw data and depth measurement mesh are checked for changes.

- iOS-only capture (Android lacks a universal depth sensor).
- Expression/jaw drift between sessions is the dominant real-world error;
  the capture gates help but real-face validation (Phase 4) is pending.
- Visualization/tracking aid only — not a medical device.
