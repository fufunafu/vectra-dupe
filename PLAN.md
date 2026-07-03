# Vectra-Dupe: Phone-Based 3D Facial Volume Analysis

Goal: reproduce the core VECTRA M3 workflow — guided multi-view face capture →
3D reconstruction → before/after registration → volumetric difference with a
color contour map — as a phone app.

---

## 1. What the VECTRA actually does (decomposed)

| Stage | VECTRA M3 | What we need on a phone |
|---|---|---|
| Capture | 6 calibrated cameras, 3 viewpoints, single 3.5 ms flash | 3 guided poses (front, left ¾, right ¾) captured sequentially with live alignment guides |
| Scale | Factory-calibrated stereo rig → true millimeters | iPhone TrueDepth / ARKit gives **metric** depth — this solves the hardest problem for free |
| Reconstruction | Stereophotogrammetry → textured mesh (~1.2 mm resolution) | Depth-map fusion or photogrammetry → textured mesh |
| Registration | Align before/after scans on stable regions | ICP alignment on forehead / nose bridge / temples |
| Analysis | Signed surface distance → color contour map + volume (mL) | Same math: signed distance field + ROI volume integration |

Key realization: the VECTRA's magic is not the cameras — it's **metric scale +
repeatable alignment**. A single photo has no absolute scale; the iPhone's
TrueDepth sensor (the Face ID camera) returns depth in real millimeters, which
is exactly what the calibrated stereo rig provides. Published studies put
TrueDepth facial scan accuracy at ~0.5–1.5 mm RMS — in the same class as the
M3's 1.2 mm geometry resolution.

## 2. Recommended architecture

**iOS-first, capture on device, heavy processing on a server.**

- **Capture (on phone, Swift/ARKit):**
  - `ARFaceTrackingConfiguration` gives live face landmarks → drive the
    alignment overlay: horizontal line through the pupils + vertical midline
    (the same two lines the VECTRA uses), plus yaw targets for the two ¾ views.
  - Auto-shutter when pose, roll, and distance are within tolerance — removes
    operator skill from the equation, which is what makes captures repeatable.
  - For each of the 3 poses, save: RGB frame + `AVDepthData` depth map +
    camera intrinsics + ARKit pose. Also capture a short burst so we can pick
    the sharpest frame.
  - Expression gating: reject capture if the face mesh shows non-neutral
    expression (smile/brow raise), since expression change ruins volume diffs.

- **Reconstruction (server, Python — Open3D / PyTorch):**
  - Back-project the 3 depth maps to point clouds using intrinsics; rough-align
    with ARKit poses + facial landmarks; refine with colored ICP; fuse via TSDF
    or Poisson reconstruction → single watertight textured mesh. Export PLY/OBJ.
  - Optional quality upgrade later: fit a FLAME 3D face model to the fused data
    for hole-free, topology-consistent meshes (makes cross-session
    correspondence much easier).

- **Analysis (server):**
  - Register session B onto session A using ICP restricted to **stable
    regions** (forehead, nasal bridge, temples — masked via landmarks). Never
    align on the treated area itself.
  - Compute signed per-vertex distance A→B, render as a color contour overlay
    (the orange/blue map in `Example-Output`).
  - User draws an ROI on the front view → integrate the signed distance over
    that patch → volume change in mL.

- **Viewing (on phone):** SceneKit/RealityKit 3D viewer with the heatmap
  texture; side-by-side and overlay modes; timeline slider across sessions
  (Pre TX → Post TX → week 4 → 8 → 20, matching the real VECTRA workflow).

**Why not pure 3-photo photogrammetry?** It works (that's literally the M3),
but from one moving camera the reconstruction has unknown scale; you'd need a
reference object (credit card / iris-diameter prior, ~±2–4% error) and faces
move between shots. Use it only as the Android fallback later (MediaPipe
landmarks + multi-view stereo + iris-based scaling).

## 3. Phased roadmap

### Phase 0 — Desktop feasibility prototype — ✅ DONE (see `phase0/`)

Built and validated on simulated TrueDepth captures (2026-06-12):
0.8 mL synthetic "filler" recovered at 0.825 ± 0.148 mL (bias +1.4%);
noise floor ±0.12 mL; repeat-scan RMS ~0.23 mm. Verdict: **GO**.
Next sub-step (Phase 0.5): run the identical pipeline on real TrueDepth
captures (Record3D export) and repeat the null test on a real face.

Original plan:
1. Python pipeline: take an iPhone TrueDepth capture (use the free "Record3D"
   app or a 50-line capture script) of a face from 3 poses → fuse → mesh.
2. Validate metric accuracy against an object of known volume (e.g. clay blob
   on a flat surface, weighed/water-displaced).
3. Implement registration + signed-distance heatmap + ROI volume on two scans
   taken minutes apart (expected diff ≈ 0 — measures the noise floor).
   **The noise floor determines whether the whole product is viable**; we need
   it well under the ~1–5 mL changes fillers produce.

### Phase 1 — Capture app — ✅ SOURCE COMPLETE (see `ios-app/`)
SwiftUI + ARKit app: guided 3-pose capture with the eye-line/midline overlay,
auto-shutter (yaw/pitch/roll/distance gates), expression gating via blend
shapes, 8-frame depth burst averaging, session storage, upload to the server.
Extrinsic/coordinate math verified numerically against the pipeline; needs
full Xcode + an iPhone with Face ID to build and field-test (none on this
machine — `cd ios-app && xcodegen generate && open VectraCapture.xcodeproj`).

### Phase 2 — Reconstruction + analysis service — ✅ DONE (see `server/`)
FastAPI server: patients/sessions/upload/process/compare endpoints, filesystem
storage, fusion ~5 s/session and comparison ~10 s on this Mac. Auto change-
region detection added to the pipeline (`vectra3d/compare.py`): smoothed
threshold + connected components, frozen-geometry re-measurement, scan-
boundary masking, sub-noise-floor flagging. End-to-end test over live HTTP
passes (`tests/e2e_test.py`: 0.850 mL measured vs 0.814 truth, clean null).

### Phase 3 — Compare & longitudinal UI — ✅ DONE (v1, see `server/static/`)
Web viewer at the server root: session list, pairwise compare, 3D heatmap
mesh (three.js), per-region mL readout with significance flags, 2D map link.
Deferred to later: PDF export, side-by-side sync view, ROI hand-editing.

### Phase 4 — Validation against the real VECTRA — ⏳ NEEDS REAL CAPTURES
Tooling is ready: `tools/compare_cli.py` compares two session folders or two
meshes and emits heatmap + volumes. Protocol: same subject, same day,
VECTRA M3 capture + phone capture, before/after a known intervention
(also: two phone captures minutes apart for the real-face null test —
acceptance gate: phantom volume < 0.3 mL). The files in `Vectra-files/`
define the report format to match.

## 4. Risks & honest caveats

- **Noise floor vs effect size** — biggest risk. Filler changes are 0.5–2 mm
  of surface displacement; we must prove repeat-scan noise is below that.
  Phase 0 exists to kill or confirm the project cheaply.
- **Expression & posture drift** between sessions dominates error in practice
  (true on the real VECTRA too). Mitigate with expression gating and stable-
  region-only registration.
- **TrueDepth field of view** — one frontal depth frame misses the jawline
  sides; that's exactly why we take the two ¾ views.
- **Android** has no universal depth sensor → defer; iOS-only v1.
- **Regulatory**: as a documentation/visualization aid this is fine; marketing
  it for clinical *diagnosis* would raise FDA/medical-device questions. Keep
  claims to "tracking and visualization."
- **Hair/headband**: VECTRA clinics use a hair net (visible in your files) —
  recommend the same; hair destroys reconstructions.

## 5. Suggested stack summary

| Layer | Choice |
|---|---|
| Phone app | Swift / SwiftUI, ARKit (face tracking + TrueDepth), RealityKit viewer |
| Reconstruction | Python, Open3D (ICP, TSDF/Poisson), later FLAME fitting |
| Landmarks (server-side) | MediaPipe Face Landmarker |
| API | FastAPI + job queue; any GPU-less box works for v1 |
| Mesh formats | PLY internally, OBJ export (VECTRA's .tom is proprietary — don't chase it) |
