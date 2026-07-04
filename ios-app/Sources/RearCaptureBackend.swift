import ARKit
import AVFoundation
import CoreGraphics
import Vision
import simd

/// Rear-camera (Operator) capture: ARWorldTrackingConfiguration + LiDAR
/// sceneDepth on Pro devices, photo-only elsewhere.
///
/// There is no ARFaceAnchor on the back camera, so the head-centered frame the
/// server expects (origin ≈ head centre, x subject-right, y up, z out of the
/// face) is SYNTHESIZED: Vision finds the face in the image, the LiDAR depth
/// (or a face-size pinhole estimate) places it in ARKit's gravity-aligned
/// world, and a head anchor is built facing the camera. The anchor keeps
/// re-estimating while the operator lines up the FRONT pose, then freezes when
/// that pose is captured — the subject sits still, so from then on ARKit's
/// world tracking alone carries every wider pose (Vision loses faces at
/// profile angles by design; nothing here depends on it after the freeze).
final class RearWorldTrackingBackend: CaptureBackend {
    static var isSupported: Bool { ARWorldTrackingConfiguration.isSupported }
    /// LiDAR = metric depth keyframes = measurement-grade session.
    static var hasLiDAR: Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    var deviceTag: String { Self.hasLiDAR ? "iphone-rear-lidar" : "iphone-rear-photo" }
    var providesDepth: Bool { Self.hasLiDAR }
    private(set) var stillCaptureSupported = false

    /// Head frame -> ARKit world. Continuously re-estimated from Vision until
    /// the front keyframe is captured, then frozen for the rest of the session.
    private var headAnchor: simd_float4x4?
    private var anchorFrozen = false
    /// Face-surface point (world, meters) the anchor was built from — the
    /// reference for the subject-movement check after the freeze.
    private var anchorFacePoint: SIMD3<Float>?
    private var subjectDriftMM: Float = 0
    private var lastDriftCheckAt: TimeInterval = 0

    /// The face-anchor origin sits ~at the head centre, not on the skin; the
    /// server crops a head-sized sphere around it. Push the synthesized origin
    /// back from the face surface by the same amount.
    private let faceToHeadCentreM: Float = 0.09
    /// Nominal temple-to-temple face width for the no-LiDAR distance estimate.
    private let nominalFaceWidthM: Float = 0.14

    // MARK: Vision (throttled, off-main)

    /// Latest Vision detection. All image coordinates are NATIVE-buffer
    /// normalized (top-left origin) so depth sampling and unprojection don't
    /// have to re-derive the orientation mapping.
    private struct VisionFace {
        var centerNative: CGPoint          // face box centre
        var widthFracUpright: CGFloat      // box width, fraction of upright width
        var yawDeg: Float                  // subject head yaw (0 = facing camera)
        var pitchDeg: Float
        var eyesNative: (CGPoint, CGPoint)?
        var timestamp: TimeInterval
    }
    private let visionQueue = DispatchQueue(label: "vectra.rear.vision", qos: .userInitiated)
    private var visionBusy = false
    private var lastVisionAt: TimeInterval = 0
    private var latestFace: VisionFace?          // main-thread only
    private var lastAnchorEstimateAt: TimeInterval = 0
    private var lastEyes: (CGPoint, CGPoint)?

    func makeConfiguration() -> ARConfiguration {
        let config = ARWorldTrackingConfiguration()
        config.isLightEstimationEnabled = true
        config.worldAlignment = .gravity   // the head frame's +y IS world up
        stillCaptureSupported = false
        if Self.hasLiDAR {
            // Keep ARKit's default 4:3 video format: sceneDepth is guaranteed
            // compatible with it and shares its aspect, so depth intrinsics
            // scale cleanly from the RGB intrinsics. The high-res-still format
            // is 4K 16:9 on many devices, which can drop sceneDepth — depth
            // keyframes matter more than 12 MP texture stills here.
            config.frameSemantics.insert(.sceneDepth)
        } else if #available(iOS 16.0, *),
                  let rec = ARWorldTrackingConfiguration
                      .recommendedVideoFormatForHighResolutionFrameCapturing {
            // Photo-only capture lives or dies on image resolution — take the
            // high-res still format so photogrammetry gets 12 MP inputs.
            config.videoFormat = rec
            stillCaptureSupported = true
        }
        let r = config.videoFormat.imageResolution
        print("[capture] rear video format \(Int(r.width))×\(Int(r.height))"
              + (Self.hasLiDAR ? " + LiDAR depth" : " (photo-only)")
              + (stillCaptureSupported ? " + high-res stills" : ""))
        return config
    }

    func reset() {
        headAnchor = nil
        anchorFrozen = false
        anchorFacePoint = nil
        subjectDriftMM = 0
        lastDriftCheckAt = 0
        latestFace = nil
        lastAnchorEstimateAt = 0
        lastEyes = nil
        lastVisionAt = 0
    }

    func didCaptureFrontPose() {
        guard headAnchor != nil else { return }
        anchorFrozen = true
        print("[capture] head anchor frozen")
    }

    // MARK: per-frame sample

    func subjectSample(for frame: ARFrame, viewportSize: CGSize) -> SubjectSample? {
        scheduleVision(frame: frame)

        if !anchorFrozen {
            // (Re-)estimate the head anchor whenever a fresh Vision face is in.
            if let face = latestFace, face.timestamp != lastAnchorEstimateAt,
               let distM = faceDistanceMeters(face: face, frame: frame) {
                lastAnchorEstimateAt = face.timestamp
                let facePoint = unprojectNative(face.centerNative, depthM: distM,
                                                frame: frame)
                if let anchor = headAnchorFacingCamera(facePoint: facePoint,
                                                       frame: frame) {
                    headAnchor = anchor
                    anchorFacePoint = facePoint
                }
            }
        } else {
            updateSubjectDrift(frame: frame)
        }
        guard let anchor = headAnchor else { return nil }   // "Looking for a face…"

        let camToHeadCV = CaptureGeometry.cameraInSubjectFrameCV(
            subjectTransform: anchor, cameraTransform: frame.camera.transform)
        var (yaw, pitch, dist) = CaptureGeometry.viewAngles(camToSubjectCV: camToHeadCV)

        // Until the anchor freezes, the head frame is BUILT facing the camera,
        // so the geometric yaw is ~0 no matter where the subject looks. The
        // front-pose angle gate must instead come from Vision's head pose: the
        // subject has to actually face the lens. (Moving the phone around the
        // subject zeroes this too — the anchor re-estimates every detection —
        // so the usual "move the phone" hints stay actionable.)
        if !anchorFrozen, let face = latestFace {
            yaw = face.yawDeg
            pitch = face.pitchDeg
        }

        let eyes = projectedEyes(frame: frame, viewportSize: viewportSize)
        let camPos = camToHeadCV.columns.3
        return SubjectSample(
            yawDeg: yaw, pitchDeg: pitch,
            rollDeg: cameraRollDeg(frame: frame),
            distanceMM: Float(dist) * 1000,
            eyeLeft: eyes.0, eyeRight: eyes.1,
            expressionNeutral: true,   // no blendshapes on the rear camera
            cameraPositionMM: SIMD3<Double>(camPos.x, camPos.y, camPos.z) * 1000,
            worldToCameraCV: camToHeadCV.inverse.scaledTranslationMM(),
            subjectDriftMM: subjectDriftMM)
    }

    func worldToCameraCV(for frame: ARFrame) -> simd_double4x4? {
        guard let anchor = headAnchor else { return nil }
        return CaptureGeometry.cameraInSubjectFrameCV(
            subjectTransform: anchor, cameraTransform: frame.camera.transform)
            .inverse.scaledTranslationMM()
    }

    // MARK: LiDAR depth

    func depthSample(for frame: ARFrame) -> DepthSample? {
        guard let sceneDepth = frame.sceneDepth else { return nil }
        let depthBuf = sceneDepth.depthMap
        CVPixelBufferLockBaseAddress(depthBuf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuf, .readOnly) }
        let w = CVPixelBufferGetWidth(depthBuf)
        let h = CVPixelBufferGetHeight(depthBuf)
        let stride = CVPixelBufferGetBytesPerRow(depthBuf) / MemoryLayout<Float32>.size
        guard let base = CVPixelBufferGetBaseAddress(depthBuf)?
            .assumingMemoryBound(to: Float32.self) else { return nil }

        // Confidence-gate the LiDAR depth: low-confidence pixels (depth edges,
        // dark/IR-absorbing surfaces) are exactly the ones that smear the TSDF.
        var confBase: UnsafeMutablePointer<UInt8>?
        var confStride = 0
        let confBuf = sceneDepth.confidenceMap
        if let confBuf {
            CVPixelBufferLockBaseAddress(confBuf, .readOnly)
            confBase = CVPixelBufferGetBaseAddress(confBuf)?
                .assumingMemoryBound(to: UInt8.self)
            confStride = CVPixelBufferGetBytesPerRow(confBuf)
        }
        defer { if let confBuf { CVPixelBufferUnlockBaseAddress(confBuf, .readOnly) } }

        let minConf = UInt8(ARConfidenceLevel.medium.rawValue)
        var depthMM = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                let v = base[y * stride + x]
                guard v.isFinite, v > 0 else { continue }
                if let confBase, confBase[y * confStride + x] < minConf { continue }
                depthMM[y * w + x] = v * 1000
            }
        }

        // sceneDepth shares the camera image's FOV and aspect, so its
        // intrinsics are the RGB intrinsics scaled to the depth resolution.
        let k = frame.camera.intrinsics
        let scale = Float(w) / Float(frame.camera.imageResolution.width)
        return DepthSample(depthMM: depthMM, width: w, height: h,
                           fx: k.columns.0.x * scale, fy: k.columns.1.y * scale,
                           cx: k.columns.2.x * scale, cy: k.columns.2.y * scale)
    }

    func setCameraLocked(_ locked: Bool) {
        guard #available(iOS 16.0, *),
              let device = ARWorldTrackingConfiguration
                  .configurableCaptureDeviceForPrimaryCamera else { return }
        FaceTrackingBackend.applyCameraLock(device: device, locked: locked)
    }

    // MARK: head anchor construction

    /// Build the head frame at `facePoint` (world, meters): +y = world up
    /// (gravity — the subject sits upright), +z = horizontal direction from
    /// the head toward the camera, +x = y × z = the subject's right (camera
    /// moving to the subject's LEFT reads a negative yaw, matching the pose
    /// targets). Origin is pushed behind the face surface to ~the head centre.
    private func headAnchorFacingCamera(facePoint: SIMD3<Float>,
                                        frame: ARFrame) -> simd_float4x4? {
        let c = frame.camera.transform.columns.3
        let camPos = SIMD3<Float>(c.x, c.y, c.z)
        var toCam = camPos - facePoint
        toCam.y = 0
        guard simd_length(toCam) > 0.05 else { return nil }
        let z = simd_normalize(toCam)
        let up = SIMD3<Float>(0, 1, 0)
        let x = simd_normalize(simd_cross(up, z))
        let origin = facePoint - z * faceToHeadCentreM
        return simd_float4x4(columns: (SIMD4<Float>(x, 0),
                                       SIMD4<Float>(up, 0),
                                       SIMD4<Float>(z, 0),
                                       SIMD4<Float>(origin, 1)))
    }

    /// After the freeze, whenever Vision re-sees the face from a near-frontal
    /// vantage, compare where the face is NOW against where the anchor says it
    /// should be. The anchor is world-locked, so any gap is the subject having
    /// shifted — which silently corrupts every later extrinsic. Surfaced as a
    /// hint via SubjectSample.subjectDriftMM.
    private func updateSubjectDrift(frame: ARFrame) {
        guard let anchor = headAnchor, let reference = anchorFacePoint,
              let face = latestFace, face.timestamp > lastDriftCheckAt,
              let distM = faceDistanceMeters(face: face, frame: frame) else { return }
        let camToHeadCV = CaptureGeometry.cameraInSubjectFrameCV(
            subjectTransform: anchor, cameraTransform: frame.camera.transform)
        let (yaw, _, _) = CaptureGeometry.viewAngles(camToSubjectCV: camToHeadCV)
        // Only trust the check near the front: on a 3/4 view Vision's box
        // slides toward the visible cheek and reads as fake drift.
        guard abs(yaw) < 25 else { return }
        lastDriftCheckAt = face.timestamp
        let facePoint = unprojectNative(face.centerNative, depthM: distM, frame: frame)
        subjectDriftMM = simd_length(facePoint - reference) * 1000
    }

    // MARK: Vision plumbing

    private func scheduleVision(frame: ARFrame) {
        // Pre-freeze the anchor needs a fresh estimate (~6 Hz); post-freeze
        // Vision only feeds the drift check, so poll gently.
        let interval: TimeInterval = anchorFrozen ? 0.5 : 0.15
        guard !visionBusy, frame.timestamp - lastVisionAt >= interval else { return }
        visionBusy = true
        lastVisionAt = frame.timestamp
        let buffer = frame.capturedImage
        let timestamp = frame.timestamp
        let bufW = CGFloat(CVPixelBufferGetWidth(buffer))
        let bufH = CGFloat(CVPixelBufferGetHeight(buffer))

        visionQueue.async { [weak self] in
            let rects = VNDetectFaceRectanglesRequest()      // box + head yaw/pitch
            let landmarks = VNDetectFaceLandmarksRequest()   // pupils for the overlay
            // .right = portrait UI over a landscape native buffer.
            let handler = VNImageRequestHandler(cvPixelBuffer: buffer,
                                                orientation: .right, options: [:])
            var found: VisionFace?
            do {
                try handler.perform([rects, landmarks])
                if let obs = rects.results?.max(by: {
                    $0.boundingBox.width < $1.boundingBox.width }) {
                    let bb = obs.boundingBox   // upright space, lower-left origin
                    let center = Self.nativeNormPoint(
                        uprightX: bb.midX, uprightYFromBottom: bb.midY)
                    var eyes: (CGPoint, CGPoint)?
                    // Match the landmarks observation to the same face by box
                    // overlap (both requests usually agree on ordering, but
                    // don't rely on it).
                    let uprightSize = CGSize(width: bufH, height: bufW)
                    if let lmkObs = landmarks.results?.max(by: {
                            $0.boundingBox.intersection(bb).width
                                < $1.boundingBox.intersection(bb).width }),
                       let lp = lmkObs.landmarks?.leftPupil?
                           .pointsInImage(imageSize: uprightSize).first,
                       let rp = lmkObs.landmarks?.rightPupil?
                           .pointsInImage(imageSize: uprightSize).first {
                        eyes = (Self.nativeNormPoint(
                                    uprightX: lp.x / uprightSize.width,
                                    uprightYFromBottom: lp.y / uprightSize.height),
                                Self.nativeNormPoint(
                                    uprightX: rp.x / uprightSize.width,
                                    uprightYFromBottom: rp.y / uprightSize.height))
                    }
                    found = VisionFace(
                        centerNative: center,
                        widthFracUpright: bb.width,
                        // Sign convention only steers the left/right hint text;
                        // the front gate is symmetric. Flip here if the hints
                        // point the wrong way on device.
                        yawDeg: -Float(truncating: obs.yaw ?? 0) * 180 / .pi,
                        pitchDeg: Float(truncating: obs.pitch ?? 0) * 180 / .pi,
                        eyesNative: eyes,
                        timestamp: timestamp)
                }
            } catch {
                print("[capture] vision failed: \(error)")
            }
            DispatchQueue.main.async {
                self?.visionBusy = false
                if let found { self?.latestFace = found }
            }
        }
    }

    /// Map a point from Vision's upright space (x right, y measured from the
    /// BOTTOM, both normalized) into native-buffer normalized coordinates
    /// (top-left origin). The native buffer is landscape and the UI portrait
    /// (CGImagePropertyOrientation.right): rotating the buffer 90° CW makes it
    /// upright, so upright(x, y-from-top) ↔ native(x = y, y = 1 - x).
    private static func nativeNormPoint(uprightX: CGFloat,
                                        uprightYFromBottom: CGFloat) -> CGPoint {
        let uprightYFromTop = 1 - uprightYFromBottom
        return CGPoint(x: uprightYFromTop, y: 1 - uprightX)
    }

    // MARK: geometry helpers

    /// Distance to the face surface, meters: median LiDAR depth in a small
    /// window at the face-box centre, or (no LiDAR) a pinhole estimate from
    /// the box's angular size against a nominal face width.
    private func faceDistanceMeters(face: VisionFace, frame: ARFrame) -> Float? {
        if Self.hasLiDAR, let sceneDepth = frame.sceneDepth {
            let depthBuf = sceneDepth.depthMap
            CVPixelBufferLockBaseAddress(depthBuf, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(depthBuf, .readOnly) }
            let w = CVPixelBufferGetWidth(depthBuf)
            let h = CVPixelBufferGetHeight(depthBuf)
            let stride = CVPixelBufferGetBytesPerRow(depthBuf) / MemoryLayout<Float32>.size
            guard let base = CVPixelBufferGetBaseAddress(depthBuf)?
                .assumingMemoryBound(to: Float32.self) else { return nil }
            var confBase: UnsafeMutablePointer<UInt8>?
            var confStride = 0
            let confBuf = sceneDepth.confidenceMap
            if let confBuf {
                CVPixelBufferLockBaseAddress(confBuf, .readOnly)
                confBase = CVPixelBufferGetBaseAddress(confBuf)?
                    .assumingMemoryBound(to: UInt8.self)
                confStride = CVPixelBufferGetBytesPerRow(confBuf)
            }
            defer { if let confBuf { CVPixelBufferUnlockBaseAddress(confBuf, .readOnly) } }

            let cxp = Int(face.centerNative.x * CGFloat(w))
            let cyp = Int(face.centerNative.y * CGFloat(h))
            let minConf = UInt8(ARConfidenceLevel.medium.rawValue)
            var values: [Float] = []
            for dy in -2...2 {
                for dx in -2...2 {
                    let x = cxp + dx, y = cyp + dy
                    guard x >= 0, x < w, y >= 0, y < h else { continue }
                    let v = base[y * stride + x]
                    guard v.isFinite, v > 0 else { continue }
                    if let confBase, confBase[y * confStride + x] < minConf { continue }
                    values.append(v)
                }
            }
            guard values.count >= 3 else { return nil }
            return values.sorted()[values.count / 2]
        }
        // Photo-only: guidance-grade distance from the face box's angular size.
        // The box width lies along the native buffer's VERTICAL axis (portrait
        // UI over a landscape buffer), so fy is the matching focal length.
        guard face.widthFracUpright > 0.01 else { return nil }
        let fy = frame.camera.intrinsics.columns.1.y
        let facePx = Float(face.widthFracUpright) * Float(frame.camera.imageResolution.height)
        return fy * nominalFaceWidthM / facePx
    }

    /// Native-buffer normalized point + planar depth -> world point (meters).
    private func unprojectNative(_ point: CGPoint, depthM: Float,
                                 frame: ARFrame) -> SIMD3<Float> {
        let k = frame.camera.intrinsics
        let res = frame.camera.imageResolution
        let u = Float(point.x) * Float(res.width)
        let v = Float(point.y) * Float(res.height)
        // Pinhole ray in OpenCV axes (x right, y down, z forward), scaled so
        // z = the planar depth LiDAR reports; flip to ARKit's GL camera axes.
        let xc = (u - k.columns.2.x) / k.columns.0.x * depthM
        let yc = (v - k.columns.2.y) / k.columns.1.y * depthM
        let pCamGL = SIMD4<Float>(xc, -yc, -depthM, 1)
        let world = frame.camera.transform * pCamGL
        return SIMD3<Float>(world.x, world.y, world.z)
    }

    /// Roll of the phone about the optical axis, from gravity: world-up
    /// projected into the image plane sits along -x (camera-native landscape
    /// axes) when a portrait phone is level. Folded into [-90, 90] like the
    /// front camera's eye-line roll.
    private func cameraRollDeg(frame: ARFrame) -> Float {
        let t = frame.camera.transform
        let xAxis = SIMD3<Float>(t.columns.0.x, t.columns.0.y, t.columns.0.z)
        let yAxis = SIMD3<Float>(t.columns.1.x, t.columns.1.y, t.columns.1.z)
        let ux = xAxis.y   // dot(worldUp, axis) = the axis's world-y component
        let uy = yAxis.y
        var roll = atan2(uy, -ux) * 180 / .pi
        if roll > 90 { roll -= 180 } else if roll < -90 { roll += 180 }
        return roll
    }

    /// Eye markers for the guide overlay: Vision pupils mapped through the
    /// same display transform ARKit uses to draw the camera image, so they
    /// stay registered to the video. Falls back to the last known points when
    /// Vision loses the face (profile poses), like the front camera does.
    private func projectedEyes(frame: ARFrame,
                               viewportSize: CGSize) -> (CGPoint, CGPoint) {
        guard let native = latestFace?.eyesNative else {
            return lastEyes ?? (.zero, .zero)
        }
        let transform = frame.displayTransform(for: .portrait,
                                               viewportSize: viewportSize)
        func toView(_ p: CGPoint) -> CGPoint {
            let n = p.applying(transform)
            return CGPoint(x: n.x * viewportSize.width,
                           y: n.y * viewportSize.height)
        }
        let eyes = (toView(native.0), toView(native.1))
        lastEyes = eyes
        return eyes
    }
}
