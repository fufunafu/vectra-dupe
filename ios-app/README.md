# VectraCapture: iOS operator face capture

SwiftUI + ARKit app for scanning another person's face with the rear camera.
The subject holds their head still while the operator moves the phone through
nine guided positions, then captures overlapping photos during a free orbit.
Operator capture is the only camera workflow.

## Build

Requires a Mac with Xcode and an iPhone running iOS 17 or later with ARKit
world tracking support. Real capture requires a physical device; the simulator
can run the synthetic demo.

```bash
cd ios-app
xcodegen generate            # optional: regenerate the checked-in project
open VectraCapture.xcodeproj # select your signing team, then run on device
```

## How capture works

- ARKit world tracking records camera movement. Vision detects the subject's
  face to establish a head frame, which freezes after the front capture.
- The nine positions cover the front, obliques, profiles, brow, jaw, and ears.
  A position captures after a stable hold with angle, distance, and level gates.
  Ask the subject to maintain a neutral expression; the rear camera does not
  provide ARKit face blend shapes for automatic expression checks.
- On devices with LiDAR, the app averages up to eight depth frames per position.
  Without LiDAR, it saves station photographs for display-only photogrammetry.
- Free orbit automatically gathers sharp, overlapping color photographs.
- Sessions contain photos, available depth in millimeters, camera calibration,
  head-relative camera transforms, a patient ID, and a timestamp in the shared
  `vectra-dupe-session/1` format.

## Using it

1. In Settings, enter the processing server's LAN URL (for example,
   `http://192.168.1.20:8008`) and configure the server patient.
2. In Capture, frame the subject with the rear camera, tap Start capture, and
   enter the capture's patient ID.
3. Follow the nine positions, then sweep around the face and finish the scan.
4. In Sessions, export the scan or upload it for server reconstruction.
   Depth captures also support an immediate on-device 3D preview. Photo-only
   captures require server reconstruction before they can be viewed in 3D.
5. Open the server URL in a browser to view the reconstructed model and compare
   eligible depth sessions. Photo-only sessions cannot be used for volume analysis.

The server prefers Apple Object Capture for textured display models and uses
depth fusion for measurement geometry. Real-face measurement accuracy still
requires validation, including for rear LiDAR captures.

## Field-test checklist (2026-09-08 build)

Convention check, verified offline on the 2026-07-03 TrueDepth sessions and
baked into `CaptureGeometry.viewAngles`: the head frame's +x is the subject's
own left (right-handed, y up, z toward the camera), and the state machine's
yaw is flipped once so "+yaw = camera toward the subject's right". Vision's
`.right` orientation is confirmed upright on real ARKit frames, and Vision's
yaw needs no flip. What still needs a person holding the phone:

1. Trust the developer profile (Settings > General > VPN & Device Management)
   the first time the build refuses to launch.
2. Start a capture, then, with the subject facing the lens, move the phone
   toward the subject's LEFT before the front pose captures. The hint must
   say "their right". If it says "their left", the yaw sign is wrong; flip
   `atan2(-x, z)` back in `viewAngles` and the Vision sign together.
3. Pupil markers should sit on the pupils in the live view. If they sit on
   the wrong eyes or off the face, the orientation mapping is wrong.
4. Front pose captured: "head anchor frozen" prints; the left_half hint must
   now point toward the subject's anatomical left as the phone moves.
5. LiDAR: after upload + process, `stats.json` should show a head-shaped
   `mesh.ply` at mm scale; tune `VECTRA_LIDAR_SDF_TRUNC_MM` (default 10) if
   the surface is thick or holed.

Fast null test (< 0.3 mL acceptance): same subject, lips closed, hair back,
two back-to-back captures with the subject not moving between them. Nine
poses in under a minute, then tap "Finish & save scan" after a short sweep
rather than filling the coverage grid. Upload both to the same server
patient, process, and compare in the viewer; the phantom volume after the
face crop is the number to beat.
