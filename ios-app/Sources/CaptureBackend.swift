import ARKit
import AVFoundation
import CoreGraphics
import simd

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
    /// head/face frame -> OpenCV camera frame, translation in mm; the
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

/// Rear capture interface: configures ARKit, tracks the subject with Vision,
/// extracts available depth, and converts camera poses into the head frame.
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
    /// The front keyframe was just captured; the rear backend freezes its
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
    /// `subjectTransform` maps the synthesized head frame into ARKit world space.
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
    /// full horizontal distance; atan2(y, z) alone is degenerate near a
    /// profile (z -> 0 as yaw -> 90°).
    ///
    /// Yaw sign: the head frame is right-handed with +y up and +z out of the
    /// face toward the camera, which FORCES +x to be the subject's own LEFT
    /// (viewer's right). That is true of ARKit's face anchor (verified on the
    /// 2026-07-03 TrueDepth sessions: at the front pose the camera's image
    /// +y axis, i.e. image-left, lies along head -x) and of the synthesized
    /// rear anchor (x = up × z). The capture state machine speaks in
    /// "+yaw = camera toward the subject's RIGHT" (pose targets, hints, the
    /// coverage grid), so the sign is flipped here, once, for every backend.
    /// Saved extrinsics are untouched: they come from the matrices, not yaw.
    static func viewAngles(camToSubjectCV: simd_double4x4)
        -> (yawDeg: Float, pitchDeg: Float, distanceM: Double) {
        let camPos = camToSubjectCV.columns.3
        let yaw = atan2(-Float(camPos.x), Float(camPos.z)) * 180 / .pi
        let horiz = (camPos.x * camPos.x + camPos.z * camPos.z).squareRoot()
        let pitch = atan2(Float(camPos.y), Float(horiz)) * 180 / .pi
        let dist = simd_length(SIMD3<Double>(camPos.x, camPos.y, camPos.z))
        return (yaw, pitch, dist)
    }
}

// MARK: - Camera settings

/// Locks exposure, white balance, and focus during the orbit capture.
enum CaptureCameraSettings {
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
}
