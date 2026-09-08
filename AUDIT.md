# Vectra-dupe audit and improvement plan

Date: 2026-09-08. Last commit before this audit: `3c30857` (2026-07-03).
Scope: phone pipeline (`vectra3d/`, `server/`, `ios-app/`), DSLR pipeline
(`VECTRA-SOFTWARE-ONLY/`), tests, and repo hygiene. Findings were produced by
three parallel code reviews and spot-checked against source; line numbers are
as of `3c30857`.

> **Working-tree note (2026-09-08 09:40):** after this audit was drafted, an
> uncommitted change removed Selfie mode entirely (front `FaceTrackingBackend`,
> `CaptureMode`, the front-camera Info.plist copy) and left Operator mode as the
> only capture path. Findings 2 and 10 refer to deleted code; finding 1 now
> reduces to "no expression gate exists in Operator mode". Line numbers in
> `ios-app/` are as of `3c30857` and will not match that tree.

## 1. Where the project stands

| Area | State | Evidence |
|---|---|---|
| Phone measurement path (TSDF → register → volume) | Works on synthetic data | `tests/e2e_test.py`: 0.814 mL bump measured 0.781 mL, null test 0 regions (run 2026-09-08) |
| Phone real-face null test | **Not met** | +2.6 mL phantom, RMS 3.7 mm (2026-07-03, expression drift on slow capture) |
| Phone display mesh (Object Capture) | Works, slow, non-deterministic | 2–5 min/session, 5 attempts always run |
| iOS Operator (rear-camera) mode | Built to an iPhone 16 Pro on 2026-09-08 (uncommitted). Yaw sign was inverted in both the old face-anchor and rear paths (head frame +x is the subject's own left); fixed once in `CaptureGeometry.viewAngles`. Vision `.right` orientation confirmed upright; Vision pitch = head tilt, so rear backend keeps geometric pitch pre-freeze. LiDAR keyframes still untested. | July face-anchor data, peer session |
| DSLR reconstruction | Plateaued | mean IoU 0.82–0.86, SSIM 0.50–0.59 vs VECTRA on 7 visits |
| DSLR metric scale | **Unreliable** | IPD vs iris scale disagree 16%; face width varies ±1.8% across visits |
| DSLR longitudinal mL (the stated goal) | **Does not exist** | no `longitudinal.py`, FLAME, cross-visit registration, or ROI code |
| Tests | One synthetic e2e (phone). None for DSLR, iOS, registration, bias field, store | |
| CI / packaging | None. No root requirements file, nothing pinned, no `pyproject` | |
| Server data | The 3 real phone sessions were deleted from `server/data` on 2026-09-08 via the viewer's trash button. **Recovered** the same day from the iPhone app container, with 10 more sessions, to `server/data/recovered-from-phone/sessions/` (13 sessions, 204 MB). | server log; peer session |

The e2e test also exits non-zero after passing because the test server does not
stop within its 10 s `wait` (`tests/e2e_test.py:149`). Anyone reading the exit
code in CI would see a failure.

## 2. Findings, ranked by impact on the mL number

### Measurement correctness (phone)

1. **Expression drift is ungated between poses.** Blendshapes are checked
   against a fixed 0.25 threshold per pose (`CaptureBackend.swift:320-328`),
   not against the front pose's vector, and the check is off past 45° yaw and
   entirely in Operator mode (`RearCaptureBackend.swift:157`). No blink,
   `mouthClose`, or lips-parted shape is checked. This is the mechanism behind
   the +2.6 mL phantom.
2. **Front-camera world tracking is never enabled.** `isWorldTrackingEnabled`
   does not appear in `ios-app/`. The locked-face-frame design needs a
   translated camera transform for the ±40°+ poses; with a face-only config
   those extrinsics are wrong or unreachable. One-line fix, must be verified on
   device.
3. **Depth bursts average 8 frames across camera motion using one extrinsic**
   (`CaptureController.swift:684-731`); the colour JPEG also gets the first
   frame's pose. The 55 mm/s stillness gate allows 10–30 mm of drift inside a
   burst. No dedupe of repeated depth timestamps.
4. **Silent ICP rejection in fusion.** When a view's correction exceeds
   15 mm / 8° the raw ARKit pose is used with no log and nothing in
   `stats.json` (`vectra3d/fuse.py:284-285`). Misregistered views are then
   Taubin-smoothed into the measurement mesh.
5. **No registration quality gate in compare.** `register.py` discards ICP
   fitness and inlier RMSE, so a non-converged before/after alignment still
   yields a confident mL figure. `surface_rms_mm` is reported after bias
   subtraction, which hides misalignment.
6. **NaN holes shrink volumes silently.** Vertices with |d| > 10 mm, no ray
   hit, or near a boundary are dropped (`analyze.py:82-85`) and
   `roi_volume_mm3` sums what remains with no coverage figure.
7. **Overlapping ROIs double-count.** Spherical ROIs (extent + 8 mm, capped at
   30 mm, `compare.py:99-102`) are summed in `processing.py:392`.
8. **Session-dependent face crop.** The crop box is anchored on the centroid of
   a 135 mm sphere (`processing.py:120-123, 358-370`), so collar/hair content
   moves the box between sessions and changes both the stable set and the
   measured area.
9. **Bias field kernel (25 mm Gaussian, `analyze.py:110`) removes any change
   broader than ~25 mm.** Diffuse oedema or weight change is subtracted as
   bias. Acceptable as a design choice, but undocumented in the UI.
10. **Fabricated intrinsics fallback** on the phone: `fx = fy = width` when
    calibration data is nil (`CaptureBackend.swift:256`). Should abort.

### DSLR correctness

11. **Metric scale is a resolution artifact.** Landmarks are detected on the
    518 px VGGT frame upscaled 3× (`metric.py:52-55`), then looked up with
    integer-pixel rounding and a 9×9 median window larger than the iris
    (`metric.py:40-48`). The 5184 px originals are never used. This alone
    explains the 16% IPD/iris disagreement and the ±1.8% face-width spread,
    which is ~5% volume error before any real change is measured.
12. **Intrinsics unconstrained.** EXIF gives 85 mm on a 22.3 mm sensor
    (fx ≈ 1974 px at 518); VGGT predicts ~1210 and cloud BA lands at ~1396.
    `sfm.focal_px_from_exif` computes the truth but neither path uses it.
13. **Calibration guard leaves rejected views uncorrected.** 4–6 views per
    visit exceed the 0.06×diag guard (`calibrate.py:203-205`) and keep raw
    poses, producing exactly the ghost ears calibration targets. Cross-view
    filters use array-index neighbours rather than yaw neighbours
    (`vggt_recon.py:82`, `calibrate.py:232`), and the strict filter drops
    points seen by no neighbour, contrary to its docstring (`calibrate.py:250`).
14. **Quality gate is not fair by construction.** Reference view chosen by
    max-IoU (selection bias, `quality.py:146`), silhouettes area-normalised so
    scale errors are invisible (`quality.py:50-52`), FOV/elevation guessed
    (`render_views.py:10,36`), SSIM compares a headlight splat render to studio
    lighting. The cloud MVS path has never been scored at all.

### Server and security

15. **Path traversal deletes data.** `store.session_dir` only checks `isdir`
    (`store.py:72-76`); `DELETE .../sessions/%2E%2E` would `rmtree` a whole
    patient. Compare body `before`/`after` are also unvalidated.
16. **No auth, CORS `*`, `/files` mounts all of `DATA_DIR`** (`app.py:19,179`).
    Any LAN client can enumerate patients and download raw face photos.
17. **Job handling:** `?wait=true` blocks a worker for minutes; background
    tasks are in-process and not persisted (a restart leaves sessions stuck in
    `processing`); concurrent `/process` on one session race on `mesh.ply` and
    `oc_*` scratch (`processing.py:157-159`); JSON read-modify-write without
    locks or atomic rename.

### iOS robustness

18. No ARSession interruption or scene-phase handling; a call mid-capture
    silently corrupts every later extrinsic.
19. JPEG encode and session save run on the main thread; the uploader loads
    every file into one `Data` body (`Uploader.swift:71-84`), ~2× peak memory.
20. AE/AWB/focus lock is applied only for the orbit, so the 9 keyframes are
    shot under continuous AE; lock is not restored on cancel.
21. `huntStart` is not reset on cancel, so a restarted pose inherits the
    relaxed +6° gate (`CaptureController.swift:1067-1074`).

### Hygiene

22. `tools/compare_cli.py:36` assigns the `(mesh, transform)` tuple to `mesh`
    and fails on any session-dir input.
23. Dead code: `fuse.depth_to_cloud`, `fuse.build_textured_mesh`,
    `cameras.intrinsic_o3d`; DSLR `pipeline.py/sfm.py/dense.py/mesh.py/
    pipeline_vggt.py` (~350 lines, Poisson path that crashes on macOS).
24. Duplication: two `_cross_view_filter`s, three TSDF blocks, four
    largest-component cleanups (DSLR); `_rigid_fit` vs `_umeyama`, two
    keep-main-components (phone). Registration, signed distance, bias field
    and volume code exist only in `vectra3d` and are not reused by the DSLR
    side.
25. Config: 17 `VECTRA_*` env vars read at import time across 4 files plus
    ~30 magic constants; DSLR hard-codes IPD 63 mm, Canon sensor width,
    `2025*.jpg` reference glob, `.venv-mp` path, an iCloud filename hack.
26. Two DSLR venvs (`.venv`, `.venv-mp`) with stale `pyvenv.cfg` paths;
    mediapipe is now in both, so the subprocess hop may be obsolete.
    `requirements.txt` lists dead deps and omits vggt, xatlas, fastapi,
    pywebview, modal, mediapipe.
27. README numbers are stale on the iOS side (3 poses, 6°, 25–45 cm, "verified
    in tests/"); `SessionsListView` labels every session "3-view TrueDepth
    scan"; DSLR `compare.py:39` labels cloud output "VGGT + TSDF".

## 3. Improvement plan

Ordered so each phase de-risks the next. Effort is engineering time, not
calendar time; hardware items need you plus an iPhone.

### Phase A. Make the numbers honest (≈3 days, no hardware)

Goal: every mL figure comes with the evidence to trust or reject it.

- A1. Diagnostics block in `result.json` and `stats.json`: per-view ICP
  accept/reject with the rejected delta (fix 4), compare ICP fitness and
  inlier RMSE before and after exclusion (fix 5), NaN fraction inside each
  ROI (fix 6). Fail the compare, not just warn, when fitness is below a
  threshold. Show these in the viewer. 0.5 d.
- A2. ROI accounting: assign each vertex to its nearest region or union the
  masks before summing; flag truncated regions. 2 h.
- A3. Landmark-anchored face crop (nose tip / eye corners from the front pose
  landmarks already computed for OC alignment) and persist `world_to_norm`
  in `stats.json`. 1 d.
- A4. Tighten the colored-refine guard to ~1 mm / 0.5° (`register.py:28`).
  0.5 h.
- A5. Fix `tools/compare_cli.py`; write output next to, not inside, the input.
  0.5 h.
- A6. Unit tests on tiny synthetic meshes for `register`, `subtract_bias_field`,
  `detect_change_regions`, `store`, and an `io_session` round-trip. Tighten
  the e2e tolerance from 0.45 to ~0.2 mL and fix the shutdown timeout so the
  exit code is meaningful. 1 d.

### Phase B. Capture repeatability on the phone (≈2.5 days + one field session)

This is the single largest lever on the real-face null test.

- B1. ~~Enable `isWorldTrackingEnabled`~~ Moot if the Selfie-mode removal is kept;
  Operator mode already uses world tracking. Yaw-sign fix done 2026-09-08.
- B2. Gate every pose on distance from the front pose's blendshape vector
  (L∞ < 0.1), add blink and `mouthClose`; keep the gate on in Operator mode
  via Vision where possible. Persist blendshapes, timestamps, tracking state
  and motion metric per pose in `session.json` so bad captures are
  diagnosable after the fact. 0.5 d.
- B3. Fix the burst: either reproject each depth frame into the first
  frame's pose before averaging, or write all frames with their own poses and
  fuse server-side. Give the JPEG its own pose. Dedupe on depth timestamp.
  Abort on missing calibration instead of fabricating intrinsics. 1 d.
- B4. Fast-capture UX: session clock, auto-finish orbit at ~45 s / 120
  frames, cut keyframes to the 5 that cover the mid-face ROI, pin the front
  distance to ~300–350 mm ±30, lock AE before the front pose and restore on
  cancel, reset `huntStart` on cancel. 0.5 d.
- B5. ARSession interruption and scene-phase handling that aborts back to
  `.aligning(.front)`. 2 h.
- B6. Field test: two back-to-back fast captures, lips closed, hair back.
  Acceptance: phantom < 0.3 mL. Then the Operator-mode checks from the
  existing next-steps list (yaw sign, `.right` orientation, LiDAR keyframes).
  This is the go/no-go for the phone pipeline as a product.

### Phase C. Server hardening (≈1.5 days)

- C1. Slug-regex validation for `pid`/`sid`/compare ids in `store.py`
  (fix 15). 1 h.
- C2. Bearer token, drop CORS `*`, replace the `/files` mount with an endpoint
  that serves only session artefacts (never `patients.json` or `raw/`)
  (fix 16). 0.5 d.
- C3. Atomic JSON writes (tmp + rename), per-session processing lock, persist
  job state and mark stale jobs failed on startup. 0.5 d.
- C4. Early-exit the Object Capture loop when alignment rms < ~5 mm; stream
  uploads to disk instead of RAM. 2 h.
- C5. Move JPEG encode and save off the main thread in the app; upload from
  file with retry. 0.5 d.

### Phase D. Repo hygiene and CI (≈1 day)

- D1. Root `pyproject.toml` with pinned deps, `vectra3d` installable so the
  `sys.path` hacks go away; a pinned DSLR requirements file that actually
  lists vggt, xatlas, fastapi, pywebview, modal, mediapipe, plus a fetch step
  for `face_landmarker.task`. Collapse `.venv-mp` if mediapipe now imports in
  `.venv`.
- D2. GitHub Actions: unit tests plus the e2e with `VECTRA_DISABLE_OC=1`.
- D3. Delete the dead DSLR COLMAP/Poisson path and the unused `vectra3d`
  functions; fold the duplicated TSDF/cleanup/filter blocks into one helper
  each.
- D4. One `config.py` per pipeline for the `VECTRA_*` vars and magic
  constants. Replace `print` parsing with `logging` and JSON progress events
  (also fixes the DSLR app reporting "Built surface mesh" on failure,
  `app/server.py:33`).
- D5. Refresh both READMEs and the session-type labels.

### Phase E. DSLR: fix scale, then decide (≈4 days)

The DSLR track only matters if it can produce a metric mesh stable to well
under 1% across visits. Do E1–E2 first and re-evaluate before spending more.

- E1. Metric scale: run landmarks on the full-resolution originals, map to
  518 px coordinates, bilinear world-point lookup, pool IPD across all 35
  views; drop the iris cross-check. Target: face width CV < 0.5% across the 7
  visits. 1 d.
- E2. Fix intrinsics from EXIF (shared camera) in both local and cloud paths,
  then score the cloud MVS path in `batch.py` for the first time. 1–2 d.
- E3. Calibration consistency: all-or-nothing solve, yaw-neighbour filters,
  fix the strict-filter logic at `calibrate.py:250`. 0.5 d.
- E4. Fair quality gate: rasterise the textured mesh at a matched FOV, assign
  references by yaw, use lighting-free metrics. 1 d.
- E5. Only if E1–E2 hit target: `vectra_sw/longitudinal.py` that imports
  `vectra3d.register/analyze/compare` rather than re-implementing them, with a
  stable region derived from the 3D landmarks and a noise floor measured on
  V1 Pre vs V2 Pre. 2–3 d. FLAME fitting after that, for correspondence only.

### Suggested order

A → C1 → B (with the field test as the milestone) → D → E. Phases A, C, and D
need no hardware and can be done before the next iPhone session, so the field
test in B6 lands on an instrumented pipeline that can explain its own result.
