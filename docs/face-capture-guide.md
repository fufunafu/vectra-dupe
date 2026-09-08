# Face capture, step by step

Based on the current iOS and server source in this workspace, reviewed September 8, 2026. This describes the implementation; the version installed on a phone may differ.

The capture has two parts: **seven guided viewpoints**, followed by **a free sweep of overlapping photographs**. The subject keeps their head still while the operator moves the phone. On a LiDAR-equipped phone, the seven stops supply measured depth plus color. The sweep supplies additional photographs for the detailed, textured 3D display.

## The exact order

**Front → L ½ → Left → R ½ → Right → Brow → Jaw → photo sweep.**

The labels that sound like “1.5 left/right” are **L ½ and R ½**. They mean intermediate oblique views. They do not specify a 1.5-second hold or a distance. All left/right directions refer to the subject's own left and right. After Jaw, the app moves straight into the photo sweep.

| Step | Phone position | What this viewpoint adds |
|---|---|---|
| 1. Front | Directly in front, around eye level: yaw 0°, elevation 0° | Central face, including the forehead, nose, mouth, chin, and visible cheeks. Establishes the shared reference for later views. |
| 2. L ½ | Around 35° toward the subject's left | An oblique view of the left cheek, side of the nose, and jaw contour. Supplies overlap between the front and left profile. |
| 3. Left | Around 72° toward the subject's left | Near-profile view of the nose projection, lips, chin, and lateral cheek/jaw. Adds surfaces less visible from the front. |
| 4. R ½ | Around 35° toward the subject's right | The corresponding right oblique view, with overlap connecting the front and right profile. |
| 5. Right | Around 72° toward the subject's right | Near-profile view of the right side, including the nose, lips, chin, and cheek/jaw contour. |
| 6. Brow | Back toward the front, phone elevated around 30° and aimed down | A higher view of the forehead, brow, and upper nose. Changes which surfaces are visible compared with eye level. The subject keeps their head still. |
| 7. Jaw | In front, phone lowered around 30° and aimed up | A lower view of the underside of the chin, lower jaw, and nose. This is the “chin” step; its app label is Jaw. |
| 8. Photo sweep | Slowly move through overlapping views, including above, level, and below | Additional color images from nearby angles for photogrammetry and texture. These saved sweep frames contain no depth. |

These anatomical descriptions explain the intended visibility from each angle. Each stop captures the camera's image and available depth across its view. There is no separate brow sensor, chin measurement, or automatic region-only capture at these steps. Hair, occlusion, framing, and sensor confidence affect what is actually recorded.

Source: [pose order, target angles, and instructions](../ios-app/Sources/CaptureController.swift), `Pose`; [visible labels](../ios-app/Sources/CaptureView.swift), `label(for:)`.

## Before the front capture: finding the face and tracking the phone

The current workflow uses the **rear camera**. It is implemented in Swift/SwiftUI with Apple's ARKit, Vision, and AVFoundation frameworks.

1. **The rear RGB camera provides the live color image.** These camera frames also supply the saved JPEGs.
2. **ARKit world tracking estimates the phone's position and orientation.** It combines camera observations with motion sensors, a technique called visual-inertial odometry. This provides a changing camera viewpoint as the operator moves. [Apple: Understanding World Tracking](https://developer.apple.com/documentation/arkit/understanding-world-tracking)
3. **Apple Vision locates the face in the image.** The app requests a face rectangle, head angles, and facial landmarks. Pupil positions place the eye guides on screen. This is face localization for capture guidance, not identity recognition.
4. **The app estimates where the head is in 3D.** With LiDAR, it takes the median of valid depth samples near the face-box center. Without LiDAR, it estimates distance from the apparent face width using an assumed 14 cm width. That photo-only estimate is for guidance, not a measured facial width.
5. **The front capture locks the head reference.** The app places an approximate head center 9 cm behind the detected face surface. When Front is captured, this coordinate frame freezes. Later camera positions are expressed relative to it. This is an estimated reference, not a scan of the inside of the head.

The fixed reference allows wider views even when Vision can no longer see the face clearly in profile. It also explains why the subject must remain still: the app continues tracking the phone relative to the original head location. The rear workflow does not use an ARKit face mesh or Face ID/TrueDepth capture.

Source: [RearCaptureBackend.swift](../ios-app/Sources/RearCaptureBackend.swift), `makeConfiguration`, `subjectSample`, `scheduleVision`, `faceDistanceMeters`, and `didCaptureFrontPose`.

## What happens at every guided stop

**Align → hold → short capture burst → keep one result → advance.** All seven stops use this same mechanism.

The app checks the target angle, elevation, phone roll, distance, and camera stillness. The stable hold lasts **0.6 seconds**, with a brief 0.35-second tolerance for alignment jitter. The distance gate is **25 to 62 cm from the estimated head reference**, rather than a precise skin-to-lens distance.

Base tolerances are approximately ±12° in yaw, ±10° in elevation, and ±10° in phone roll. Profiles allow ±15° elevation. After a prolonged search for alignment, the yaw window can widen by up to another 6°. These are target windows, so saved views need not land at exactly the listed angles.

The operator must ask for a neutral expression. Although the controller has an expression flag, the rear backend always sets it to true. This workflow does not automatically verify a neutral expression.

### On a phone with LiDAR

ARKit exposes a LiDAR-backed `sceneDepth` map alongside the color image. It is a grid of depth samples that can be converted into 3D points using camera calibration. [Apple: Displaying a point cloud using scene depth](https://developer.apple.com/documentation/arkit/displaying-a-point-cloud-using-scene-depth)

For each guided stop, the app:

1. Collects **eight depth frames** under normal conditions.
2. Discards invalid samples and, when confidence data is available, low-confidence samples.
3. Converts depth from meters into **millimeters**.
4. Averages valid depth at each pixel. A pixel must be valid in at least 60% of the collected frames: five of eight in a full burst. Otherwise it becomes zero, meaning invalid.
5. Keeps **one color JPEG**, selecting the frame with the lowest estimated camera motion during the burst.
6. Stores the averaged depth map, image dimensions, camera calibration, and camera transform relative to the head reference.

The eight frames are repeated observations from essentially the same viewpoint to reduce depth noise. They do not become eight separate saved depth maps. If a burst times out after four seconds, at least three depth frames are needed to finish; otherwise the app returns to alignment.

The saved depth transform comes from the first depth frame of the burst. The selected JPEG may come from a later frame, which is another reason the hold matters. Depth maps retain the resolution supplied by ARKit; a higher-resolution photo does not increase the depth map's sampling resolution.

### On a phone without LiDAR

The same seven stops still occur. The app selects one color frame from a short burst and stores it as a station photograph such as `key_front`. There is **no measured depth map**. It also requests a high-resolution still when the configured format and rate limits allow.

The resulting session supports a photo-based 3D display. The current server does not use photo-only sessions for volume analysis.

Source: [CaptureController.swift](../ios-app/Sources/CaptureController.swift), capture constants, `collectBurstFrame`, `finishBurst`, and `finishStationPhoto`; [depth filtering](../ios-app/Sources/RearCaptureBackend.swift), `depthSample`.

## Why the final stage takes many pictures

The guided views provide broad coverage with a small number of saved stations. The final sweep adds **closely spaced, overlapping color views**. Their shared visual features let photogrammetry reconstruct a textured surface, while the photos also supply its visible skin color and detail. Apple's Object Capture builds 3D objects from photographs through `PhotogrammetrySession`. [Apple: PhotogrammetrySession](https://developer.apple.com/documentation/realitykit/photogrammetrysession)

During this sweep, the app attempts to lock exposure, white balance, and focus so the appearance stays consistent. It automatically accepts a frame when distance is in range, motion is below its sweep ceiling, the image passes a sharpness check, and the viewpoint is sufficiently different from previous accepted images.

| Sweep setting | Current behavior |
|---|---|
| Viewpoint spacing | At least 4° from every previously accepted sweep viewpoint in combined yaw/elevation space. |
| Standard photo limit | Up to 180 accepted sweep JPEGs. Pausing at one angle does not continually add duplicates. |
| Coverage display | 39 coarse cells: 13 horizontal directions, spaced 15° apart, across three elevation bands near −25°, 0°, and +25°. Multiple photos can occupy a cell. |
| Additional stills | Up to 30 in the sweep, at least 1.5 seconds apart, if high-resolution capture is enabled. A newly covered cell can trigger a request. |
| What a photo stores | JPEG, resolution, camera intrinsics, and its head-relative camera transform. No depth map. |
| How it ends | Tap **Finish & save scan**. At the photo cap, the app asks the operator to finish; it does not automatically save. |

In this implementation, the **LiDAR configuration disables the extra high-resolution still path** to preserve a compatible depth/video format. The photo-only configuration enables it when a recommended format is available. Comments describe these stills as approximately 12 MP, but the actual dimensions come from the device's returned frame. The 1.5-second value belongs to this optional still rate limit, not the L ½/R ½ stops.

Implementation detail: entering the sweep clears the separate still buffer. Earlier high-resolution stills from station requests are therefore not reliably retained in the final session; the selected standard station JPEGs are retained. The guide describes the saved output rather than promising every requested still survives.

Source: [CaptureController.swift](../ios-app/Sources/CaptureController.swift), `beginOrbit`, `harvestOrbitFrame`, `captureStill`, `orbitBucket`, and `finishOrbit`; [camera configuration](../ios-app/Sources/RearCaptureBackend.swift), `makeConfiguration`.

## What is saved, and what the server builds

Finishing writes a session folder on the phone. Uploading it is a later action from Sessions.

| Saved item | Meaning |
|---|---|
| `depth_front.bin`, `depth_left_half.bin`, etc. | One averaged float32 depth map per guided LiDAR stop, in millimeters; zero means invalid. |
| `color_front.jpg`, etc. | The selected color image accompanying a LiDAR stop. |
| `color_key_front.jpg`, etc. | Selected guided station photos in the photo-only workflow. |
| `color_orbit_000.jpg`, etc. | The dense sweep photographs. Optional extra stills use `color_still_000.jpg`, etc. |
| `session.json` | Device tag, timestamp, entered patient ID, image/depth dimensions, file references, camera calibration, and head-relative transforms. Photo-only sessions have `capture_kind: photo_only`. |

**Camera intrinsics** describe how image pixels relate to camera rays: focal lengths and image center. **Camera extrinsics** describe where the camera was and how it was oriented relative to the head reference. Together with depth, they let the server place samples in a common 3D space.

The server uses two reconstruction paths:

1. **Depth geometry for measurement:** the guided depth maps become point clouds. ICP, or iterative closest point registration, refines alignment between overlapping views. TSDF, or truncated signed distance function fusion, combines them into one surface. This produces `mesh.ply`, the geometry used by the measurement pipeline.
2. **Photographic geometry for display:** Apple Object Capture uses the photo set to reconstruct a detailed textured model. For depth sessions, the server aligns this display reconstruction to the depth reference using facial landmarks. The viewer receives `mesh.glb` and/or `mesh_textured.glb`. If photo reconstruction is unavailable or rejected, depth sessions can fall back to displaying the depth surface with photo-derived color.

The extra photos can improve the display surface and texture. They do not add depth samples to the measurement fusion. A realistic-looking model is not evidence of measurement accuracy: this project's real-face accuracy, including rear LiDAR, still requires validation. Photo-only sessions have no depth measurement mesh.

Sources: [SessionWriter.swift](../ios-app/Sources/SessionWriter.swift), [capture save flow](../ios-app/Sources/CaptureController.swift), [depth fusion](../vectra3d/fuse.py), [server reconstruction paths](../server/processing.py), and [Object Capture alignment](../vectra3d/photogrammetry.py).
