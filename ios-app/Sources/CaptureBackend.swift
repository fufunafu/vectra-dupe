import ARKit
import AVFoundation
import CoreGraphics
import simd

/// Which camera pipeline drives the capture. Selfie is the original front
/// TrueDepth face-tracking flow; Operator films someone else with the BACK
/// camera (world tracking + LiDAR depth where the device has it).
enum CaptureMode: String {
    case selfieFront
    case operatorRear

    var isSupported: Bool {
        switch self {
        case .selfieFront: return ARFaceTrackingConfiguration.isSupported
        case .operatorRear: return ARWorldTrackingConfiguration.isSupported
        }
    }

    func makeBackend() -> CaptureBackend? {
        guard isSupported else { return nil }
        switch self {
        case .selfieFront: return FaceTrackingBackend()
        case .operatorRear: return RearWorldTrackingBackend()
        }
    }
}

/// Everything the (camera-agnostic) capture state machine needs from one
/// ARFrame: where the camera sits relative to the subject's head, the guide
/// overlay's eye points, and the extrinsic to store if this frame is captured.
struct SubjectSample {
    var yawDeg: Float
    var pitchDeg: Float
    var rollDeg: Float
    var distanceMM: Float
    var eyeLeft: CGPoint
    var eyeRight: CGPoint
    var expressionNeutral: Bool
    /// Camera position in the head frame, millimeters (drives the stillness
    /// motion metric).
    var cameraPositionMM: SIMD3<Double>
    /// head/face frame -> OpenCV camera frame, translation in mm — the
    /// `world_to_camera` written to session.json if this frame is captured.
    var worldToCameraCV: simd_double4x4
    /// How far the subject appears to have moved since the head frame was
    /// locked (rear capture only; 0 when unknown/not applicable).
    var subjectDriftMM: Float = 0
}

/// One frame's metric depth, already in the session format's conventions
/// (row-major float32 mm, 0 = invalid) with intrinsics at the map's resolution.
struct DepthSample {
    var depthMM: [Float]
    var width: Int
    var height: Int
    var fx: Float, fy: Float, cx: Float, cy: Float
}

/// Camera-specific half of the capture: owns the AR configuration, tracks the
/// subject (face anchor or Vision + head anchor), extracts metric depth, and
/// converts camera poses into the head-centered frame the server expects.
/// The state machine in CaptureController is shared across backends.
protocol CaptureBackend: AnyObject {
    /// `device` tag written to session.json.
    var deviceTag: String { get }
    /// False on rear captures without LiDAR: the session is photo-only.
    var providesDepth: Bool { get }
    /// Whether the configuration chosen supports captureHighResolutionFrame.
    var stillCaptureSupported: Bool { get }
    func makeConfiguration() -> ARConfiguration
    /// Clear per-session state (locked anchors, cached eye points, …).
    func reset()
    /// Measure the subject in this frame; nil until a face has been seen.
    func subjectSample(for frame: ARFrame, viewportSize: CGSize) -> SubjectSample?
    /// Metric depth for this frame; nil when unavailable (no sensor / dropout).
    func depthSample(for frame: ARFrame) -> DepthSample?
    /// Extrinsic for an out-of-band frame (a high-res still's completion
    /// handler), using the current subject anchor.
    func worldToCameraCV(for frame: ARFrame) -> simd_double4x4?
    /// The front keyframe was just captured — the rear backend freezes its
    /// head anchor here so every later pose shares one world frame.
    func didCaptureFrontPose()
    /// Lock (or restore) AE/AWB/focus for the free-orbit phase.
    func setCameraLocked(_ locked: Bool)
}

// MARK: - shared geometry

enum CaptureGeometry {
    /// Camera pose expressed in the subject/head frame, converted from ARKit's
    /// OpenGL-style camera axes (y up, z backward) to OpenCV (y down,
    /// z forward) so the saved extrinsics match the processing pipeline.
    /// `subjectTransform` is the head frame -> ARKit world transform (the face
    /// anchor on the front camera, the synthesized head anchor on the rear).
    static func cameraInSubjectFrameCV(subjectTransform: simd_float4x4,
                                       cameraTransform: simd_float4x4) -> simd_double4x4 {
        var glToCV = matrix_identity_float4x4
        glToCV.columns.1.y = -1
        glToCV.columns.2.z = -1
        let camInSubject = subjectTransform.inverse * cameraTransform * glToCV
        return simd_double4x4(camInSubject)
    }

    /// (yaw°, pitch°, distance m) of a camera-to-subject transform: which
    /// "view" of the head this is. Pitch is elevation measured against the
    /// full horizontal distance — atan2(y, z) alone is degenerate near a
    /// profile (z -> 0 as yaw -> 90°).
    static func viewAngles(camToSubjectCV: simd_double4x4)
        -> (yawDeg: Float, pitchDeg: Float, distanceM: Double) {
        let camPos = camToSubjectCV.columns.3
        let yaw = atan2(Float(camPos.x), Float(camPos.z)) * 180 / .pi
        let horiz = (camPos.x * camPos.x + camPos.z * camPos.z).squareRoot()
        let pitch = atan2(Float(camPos.y), Float(horiz)) * 180 / .pi
        let dist = simd_length(SIMD3<Double>(camPos.x, camPos.y, camPos.z))
        return (yaw, pitch, dist)
    }
}

// MARK: - Front TrueDepth (face tracking) backend

/// The original capture pipeline: ARFaceTrackingConfiguration provides both
/// the metric depth (TrueDepth) and the head-centered coordinate frame (the
/// face anchor). Past ~±40° of camera yaw the anchor drops, so the last good
/// transform is kept locked (the head is held still) and the capture keeps
/// running off ARKit's world-tracked camera pose.
final class FaceTrackingBackend: CaptureBackend {
    let deviceTag = "iphone-truedepth"
    let providesDepth = true
    private(set) var stillCaptureSupported = false

    /// The most recent face-anchor transform seen while the face WAS tracked.
    private var lockedFaceTransform: simd_float4x4?
    /// Last projected eye points while the face was tracked, reused to keep
    /// the on-screen guide line drawn through the profile poses.
    private var lastEyes: (CGPoint, CGPoint)?

    /// Object Capture's reconstructed mesh density is capped by INPUT image
    /// resolution, so pick the highest-resolution supported streaming format;
    /// prefer the one that supports captureHighResolutionFrame (iOS 16+) —
    /// full-res stills are the single biggest OC sharpness lever.
    func makeConfiguration() -> ARConfiguration {
        let config = ARFaceTrackingConfiguration()
        config.isLightEstimationEnabled = true
        var format = ARFaceTrackingConfiguration.supportedVideoFormats.max {
            let a = $0.imageResolution, b = $1.imageResolution
            return a.width * a.height < b.width * b.height
        }
        stillCaptureSupported = false
        if #available(iOS 16.0, *),
           let rec = ARFaceTrackingConfiguration
               .recommendedVideoFormatForHighResolutionFrameCapturing {
            format = rec
            stillCaptureSupported = true
        }
        if let fmt = format {
            config.videoFormat = fmt
            let r = fmt.imageResolution
            print("[capture] video format \(Int(r.width))×\(Int(r.height)) "
                  + "(\(String(format: "%.1f", r.width * r.height / 1e6)) MP)"
                  + (stillCaptureSupported ? " + high-res stills" : ""))
        }
        return config
    }

    func reset() {
        lockedFaceTransform = nil
        lastEyes = nil
    }

    func subjectSample(for frame: ARFrame, viewportSize: CGSize) -> SubjectSample? {
        // Keep the locked face frame fresh whenever the face IS tracked. Once
        // the phone orbits past the tracking limit the anchor disappears, so we
        // fall back to the last locked transform (the head is held still, so it
        // remains an accurate face frame) and keep driving the capture from
        // ARKit's world-tracked camera pose — which never depends on the face.
        let trackedFace = frame.anchors
            .compactMap { $0 as? ARFaceAnchor }
            .first { $0.isTracked }
        if let f = trackedFace { lockedFaceTransform = f.transform }
        guard let faceTransform = trackedFace?.transform ?? lockedFaceTransform else {
            return nil
        }
        let faceTracked = trackedFace != nil

        let camToFaceCV = CaptureGeometry.cameraInSubjectFrameCV(
            subjectTransform: faceTransform, cameraTransform: frame.camera.transform)
        let (yaw, pitch, dist) = CaptureGeometry.viewAngles(camToSubjectCV: camToFaceCV)

        // Expression + eye line need a live anchor; on the locked frame we can't
        // measure them, so we assume the held-still neutral face and reuse the
        // last eye line for the guide overlay. Expression detection is ALSO
        // unreliable once the camera swings past ~45° yaw: ARKit may still report
        // a (poorly-fit) anchor that misreads a profile as a non-neutral
        // expression, falsely blocking capture with "Relax your face".
        let neutral = abs(yaw) < 45
            ? (trackedFace.map(Self.isExpressionNeutral) ?? true)
            : true
        let eyes: (CGPoint, CGPoint)
        if let face = trackedFace {
            eyes = projectedEyes(faceAnchor: face, frame: frame,
                                 viewportSize: viewportSize)
            lastEyes = eyes
        } else {
            eyes = lastEyes ?? (.zero, .zero)
        }

        // Roll is the tilt of the on-screen line between the eyes, so the gate
        // matches exactly the line the user sees and levels against. Folded
        // into [-90, 90] so the eye point order (camera mirroring) can't flip
        // a level line to ~180°. Only meaningful with a live anchor NEAR THE
        // FRONT: past ~45° yaw ARKit may keep reporting a poorly-fit anchor
        // whose projected "eyes" land on the nose/cheek, and the line between
        // them reads as a big roll that blocks profile poses forever.
        var roll: Float = 0
        if faceTracked && abs(yaw) < 45 {
            roll = atan2(Float(eyes.1.y - eyes.0.y),
                         Float(eyes.1.x - eyes.0.x)) * 180 / .pi
            if roll > 90 { roll -= 180 } else if roll < -90 { roll += 180 }
        }

        let camPos = camToFaceCV.columns.3
        return SubjectSample(
            yawDeg: yaw, pitchDeg: pitch, rollDeg: roll,
            distanceMM: Float(dist) * 1000,
            eyeLeft: eyes.0, eyeRight: eyes.1,
            expressionNeutral: neutral,
            cameraPositionMM: SIMD3<Double>(camPos.x, camPos.y, camPos.z) * 1000,
            worldToCameraCV: camToFaceCV.inverse.scaledTranslationMM())
    }

    func depthSample(for frame: ARFrame) -> DepthSample? {
        guard let depthData = frame.capturedDepthData else { return nil }

        let converted = depthData.converting(
            toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let buffer = converted.depthDataMap
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer)
        let h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer) / MemoryLayout<Float32>.size
        guard let base = CVPixelBufferGetBaseAddress(buffer)?
            .assumingMemoryBound(to: Float32.self) else { return nil }

        var depthMM = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                let v = base[y * stride + x]
                depthMM[y * w + x] = v.isFinite && v > 0 ? v * 1000 : 0
            }
        }

        // Intrinsics for the depth map come from its calibration data,
        // scaled from the reference (full sensor) resolution.
        var fx = Float(w), fy = Float(w), cx = Float(w) / 2, cy = Float(h) / 2
        if let calib = converted.cameraCalibrationData {
            let k = calib.intrinsicMatrix
            let refW = Float(calib.intrinsicMatrixReferenceDimensions.width)
            let scale = Float(w) / refW
            fx = k.columns.0.x * scale
            fy = k.columns.1.y * scale
            cx = k.columns.2.x * scale
            cy = k.columns.2.y * scale
        }
        return DepthSample(depthMM: depthMM, width: w, height: h,
                           fx: fx, fy: fy, cx: cx, cy: cy)
    }

    /// Extrinsic for a high-res still's own frame. The face transform is
    /// world-anchored and the head is held still, so the CURRENT locked
    /// transform is valid even though the still lands ~100 ms after its
    /// trigger.
    func worldToCameraCV(for frame: ARFrame) -> simd_double4x4? {
        guard let faceTransform = lockedFaceTransform else { return nil }
        return CaptureGeometry.cameraInSubjectFrameCV(
            subjectTransform: faceTransform,
            cameraTransform: frame.camera.transform)
            .inverse.scaledTranslationMM()
    }

    func didCaptureFrontPose() {}   // the face anchor IS the head frame

    /// Lock (or restore) AE/AWB/focus on the front camera while ARKit runs —
    /// exposure drift across the orbit sweep was the main source of texture
    /// seams in the projected atlas.
    func setCameraLocked(_ locked: Bool) {
        guard #available(iOS 16.0, *),
              let device = ARFaceTrackingConfiguration
                  .configurableCaptureDeviceForPrimaryCamera else { return }
        Self.applyCameraLock(device: device, locked: locked)
    }

    /// Shared AVCaptureDevice AE/AWB/focus lock, used by both backends.
    static func applyCameraLock(device: AVCaptureDevice, locked: Bool) {
        do {
            try device.lockForConfiguration()
            if locked {
                if device.isExposureModeSupported(.locked) { device.exposureMode = .locked }
                if device.isWhiteBalanceModeSupported(.locked) { device.whiteBalanceMode = .locked }
                if device.isFocusModeSupported(.locked) { device.focusMode = .locked }
            } else {
                if device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposureMode = .continuousAutoExposure
                }
                if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                    device.whiteBalanceMode = .continuousAutoWhiteBalance
                }
                if device.isFocusModeSupported(.continuousAutoFocus) {
                    device.focusMode = .continuousAutoFocus
                }
            }
            device.unlockForConfiguration()
            print("[capture] camera \(locked ? "locked" : "auto") for orbit")
        } catch {
            print("[capture] camera lock failed: \(error)")
        }
    }

    static func isExpressionNeutral(_ anchor: ARFaceAnchor) -> Bool {
        let keys: [ARFaceAnchor.BlendShapeLocation] = [
            .jawOpen, .mouthSmileLeft, .mouthSmileRight, .mouthPucker,
            .browInnerUp, .browDownLeft, .browDownRight, .cheekPuff]
        for key in keys {
            if let v = anchor.blendShapes[key]?.floatValue, v > 0.25 { return false }
        }
        return true
    }

    private func projectedEyes(faceAnchor: ARFaceAnchor, frame: ARFrame,
                               viewportSize: CGSize) -> (CGPoint, CGPoint) {
        // The eye transforms are centered on the eyeball, not the pupil. Push
        // each point forward by one eyeball radius along the gaze direction
        // (toward `lookAtPoint`) so the marker lands on the cornea/pupil rather
        // than the upper iris (the user looks slightly down at the screen).
        let eyeballRadius: Float = 0.0125          // metres, centre -> cornea
        let lookAt = faceAnchor.lookAtPoint        // face-space focus point

        func project(_ eye: simd_float4x4) -> CGPoint {
            let centerFace = simd_make_float3(eye.columns.3)
            var gaze = lookAt - centerFace
            let len = simd_length(gaze)
            gaze = len > 1e-5 ? gaze / len : SIMD3<Float>(0, 0, 1)
            let pupilFace = centerFace + gaze * eyeballRadius
            let world = faceAnchor.transform * SIMD4<Float>(pupilFace, 1)
            return frame.camera.projectPoint(
                simd_make_float3(world), orientation: .portrait,
                viewportSize: viewportSize)
        }
        return (project(faceAnchor.leftEyeTransform),
                project(faceAnchor.rightEyeTransform))
    }
}
