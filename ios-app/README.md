# VectraCapture — iOS guided 3-pose face capture

SwiftUI + ARKit app that captures the three VECTRA-style poses (front,
left ¾, right ¾) with the TrueDepth camera and uploads them to the
processing server in the shared `vectra-dupe-session/1` format.

## Build

Requires a Mac with full Xcode (15+) and an iPhone with Face ID (TrueDepth).
ARKit face tracking does not run in the simulator — use a real device.

```bash
brew install xcodegen        # already installed if `which xcodegen` works
cd ios-app
xcodegen generate            # produces VectraCapture.xcodeproj
open VectraCapture.xcodeproj # set your signing team, then run on device
```

## How capture works

- `ARFaceTrackingConfiguration` tracks the face; the overlay draws the fixed
  target cross (horizontal eye line + vertical midline — the same two
  alignment lines the VECTRA M3 uses) and the live line between the
  subject's eyes, green when aligned.
- A pose auto-captures after 0.6 s of holding: yaw within 6° of the target
  (0°, −40°, +40°), pitch/roll level, 25–45 cm away, neutral expression
  (blend-shape gated: no smile/jaw-open/brow-raise).
- Each capture averages an 8-frame TrueDepth burst (pixels valid in <60% of
  frames are dropped) — this halves the surface noise, as measured in
  phase 0.
- Geometry is saved in the face-anchor frame, so it does not matter whether
  the user turns their head or moves the phone; depth maps are millimeters,
  extrinsics are OpenCV-convention world-to-camera. The conversion from
  ARKit's camera axes was verified numerically against the pipeline
  (`tests/` in the repo root).

## Using it

1. Settings tab: server URL (the Mac running `server/`, e.g.
   `http://192.168.1.20:8008`) and patient name.
2. Capture tab → Start capture → follow the prompts through the 3 poses.
3. Sessions tab → Upload. The server fuses the mesh immediately; comparisons
   run from the web viewer at the server URL.
