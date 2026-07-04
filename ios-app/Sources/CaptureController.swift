import ARKit
import AVFoundation
import Combine
import CoreImage
import simd
import UIKit

/// One captured pose: burst-averaged depth in millimeters plus geometry, and
/// the matching color photo (for texturing) when available.
struct CapturedPose {
    let name: String           // "front" | "left" | "right"
    let depthMM: [Float]       // row-major, h*w, 0 = invalid
    let width: Int
    let height: Int
    let fx: Float, fy: Float, cx: Float, cy: Float
    let worldToCamera: simd_double4x4  // face-frame -> CV camera frame, mm

    /// JPEG of the RGB frame for this pose (camera-native landscape), plus the
    /// RGB camera intrinsics at that image's resolution. nil for demo captures.
    var colorJPEG: Data? = nil
    var rgbWidth: Int = 0
    var rgbHeight: Int = 0
    var rgbFx: Float = 0, rgbFy: Float = 0, rgbCx: Float = 0, rgbCy: Float = 0
}

/// One auto-harvested colour-only frame from the free-orbit phase: a JPEG +
/// its RGB intrinsics + pose, with NO depth. The server projects these onto the
/// (depth-fused) mesh for texture but never runs them through ICP/TSDF, so the
/// dense RGB set can be large without slowing the geometry path.
struct ColorFrameCapture {
    let name: String
    let jpeg: Data
    let width: Int, height: Int
    let fx: Float, fy: Float, cx: Float, cy: Float
    let worldToCamera: simd_double4x4   // face-frame -> CV camera frame, mm
}

/// Guidance + capture state machine around an ARSession. The camera-specific
/// half (front TrueDepth face tracking vs rear world tracking + LiDAR) lives
/// behind `CaptureBackend`; this class owns the shared pose sequence, gates,
/// burst averaging, orbit harvest, and session writing. "World" for the saved
/// session is the head-centered frame at each capture (x subject-right, y up,
/// z out of the face) — the face anchor on the front camera, a Vision-placed
/// head anchor on the rear.
final class CaptureController: NSObject, ObservableObject, ARSessionDelegate {
    enum Pose: Int, CaseIterable {
        // Nine depth keyframes covering wider arcs than the original five:
        // front + a 3/4 view and a near-profile each side, then chin-up/down
        // (camera above/below) and a near-ear view each side. This is the
        // BOUNDED set that drives geometry — it goes through ICP + TSDF, whose
        // cost grows ~O(n²), so it stays small. Dense colour is harvested
        // separately in the free-orbit phase (see Phase.orbiting). (`front`
        // MUST stay index 0 — the server uses poses[0] as the ICP reference and
        // the frame everything is normalized into.)
        //
        // PHONE-ORBIT capture: the SUBJECT holds the head still and the phone
        // swings around to each side. This is the only way to reach a true side
        // profile — ARKit's face anchor drops tracking past ~±40° of yaw, so a
        // head-turn capture can never see the nose in profile. Past that limit
        // we keep going on the LOCKED face frame (last good face transform) +
        // ARKit world tracking of the camera, which stays valid with no face.
        case front = 0, leftHalf, left, rightHalf, right,
             brow, jaw, earLeft, earRight
        var name: String {
            ["front", "left_half", "left", "right_half", "right",
             "brow", "jaw", "ear_left", "ear_right"][rawValue]
        }
        /// Camera yaw around the face, degrees (camera position seen from the
        /// face frame). The near-profiles sit at ±72° rather than a full 90° so
        /// they still share enough surface with the 3/4 view for ICP to chain;
        /// the ear views reach ±80°. Order matches vectra3d POSE_NAMES.
        var targetYawDeg: Float {
            [0, -35, -72, 35, 72, 0, 0, -80, 80][rawValue]
        }
        /// Camera ELEVATION around the face, degrees (above = +). brow looks
        /// down from above the eye line; jaw looks up from below. The rest are
        /// captured roughly level.
        var targetPitchDeg: Float {
            [0, 0, 0, 0, 0, 30, -30, 0, 0][rawValue]
        }
        /// True once the camera is past where ARKit can still track the face, so
        /// the capture relies on the locked face frame instead of a live anchor.
        var needsLockedFrame: Bool { abs(targetYawDeg) > 40 }
        /// Extra yaw tolerance for the ear views: they're captured blind on the
        /// dead-reckoned frame at ±80°, where hunting a ±12° window one-handed
        /// is maddening — and their depth adds the least (the orbit photos
        /// cover ears photogrammetrically). Field-tested pain point.
        var yawTolBonusDeg: Float {
            self == .earLeft || self == .earRight ? 6 : 0
        }
        var instruction: String {
            switch self {
            case .front: return "Look straight ahead and hold still"
            case .leftHalf: return "Hold still — move the phone to their LEFT"
            case .left: return "Keep going LEFT for a side profile"
            case .rightHalf: return "Hold still — move the phone to their RIGHT"
            case .right: return "Keep going RIGHT for a side profile"
            case .brow: return "Raise the phone above their eye line"
            case .jaw: return "Lower the phone below their chin"
            case .earLeft: return "Move the phone past their LEFT ear"
            case .earRight: return "Move the phone past their RIGHT ear"
            }
        }
    }

    enum Phase: Equatable {
        case idle
        case preview            // camera live, but no guides / auto-shutter yet
        case aligning(pose: Pose)
        case holding(pose: Pose, progress: Double)
        case capturing(pose: Pose)
        case orbiting           // free orbit: auto-harvest dense colour frames
        case done
    }

    @Published var phase: Phase = .idle
    @Published var statusText = "Tap Start"
    @Published var guidance = GuidanceState()
    @Published var finishedSession: URL?
    @Published var isDemo = false
    /// Operator-entered identifier for the current capture session (no spaces).
    @Published var patientId = ""

    /// Free-orbit coverage: the set of yaw×pitch grid cells that have been
    /// filled with a harvested colour frame, and how many frames in total. Both
    /// published so the coverage-map overlay updates live. Keys are built by
    /// `Self.orbitCellKey`.
    @Published var coverageCells: Set<String> = []
    @Published var orbitCapturedCount = 0

    /// Eyes-free tone + haptic feedback (no speech). Enabled flag is driven from
    /// the Settings toggle by the view.
    let cues = CaptureCues()

    /// Which camera pipeline runs: Selfie = front TrueDepth (the operator IS
    /// the subject, cues read "your left/right"); Operator = rear camera
    /// filming someone else ("their …"). Set by the view from the toggle.
    @Published private(set) var mode: CaptureMode = .selfieFront
    /// Camera-specific half of the capture (nil until a preview starts, or on
    /// devices that can't run the selected mode).
    private var backend: CaptureBackend?

    /// Switch camera pipelines. Ignored mid-capture (the view disables the
    /// toggle then); restarts the live preview so the new camera shows.
    func setMode(_ newMode: CaptureMode) {
        guard newMode != mode else { return }
        switch phase {
        case .idle, .preview, .done: break
        default: return
        }
        mode = newMode
        guard !isDemo else { return }
        if captureSupported {
            startPreview()
        } else {
            session.pause()
            phase = .idle
            statusText = unsupportedMessage
        }
    }

    /// Whether the current mode can run on this device (Selfie needs TrueDepth,
    /// Operator needs rear world tracking). Demo mode covers the rest.
    var captureSupported: Bool { mode.isSupported }
    /// False only for rear capture on a device without LiDAR: the session will
    /// be photo-only (display mesh, no measurement geometry).
    var capturesDepth: Bool {
        mode == .selfieFront || ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    private var unsupportedMessage: String {
        mode == .selfieFront
            ? "No TrueDepth camera — tap “Run demo” instead"
            : "This device can't run rear-camera AR — tap “Run demo” instead"
    }

    /// Adapt operator-voiced copy ("their …") to the subject when in Selfie mode.
    private func phrasing(_ text: String) -> String {
        mode == .selfieFront
            ? text.replacingOccurrences(of: "their", with: "your",
                                        options: .caseInsensitive)
            : text
    }

    struct GuidanceState {
        var yawDeg: Float = 0
        var pitchDeg: Float = 0
        var rollDeg: Float = 0
        var distanceMM: Float = 0
        var eyeLeft: CGPoint = .zero
        var eyeRight: CGPoint = .zero
        var hasFace = false
        var angleOK = false        // head yaw matches the target pose angle
        var aligned = false
        var expressionNeutral = true
    }

    let session = ARSession()
    private var captured: [CapturedPose] = []
    private var burst: [[Float]] = []
    private var burstGeometry: (w: Int, h: Int, fx: Float, fy: Float,
                                cx: Float, cy: Float,
                                worldToCamera: simd_double4x4)?
    /// RGB photo + intrinsics grabbed once per burst, for texturing the mesh.
    /// `worldToCamera` is only consumed by the photo-only station path (a depth
    /// pose reuses its own extrinsic for the color entry).
    private struct BurstColor {
        let jpeg: Data
        let w: Int, h: Int
        let fx: Float, fy: Float, cx: Float, cy: Float
        let worldToCamera: simd_double4x4
    }
    private var burstColor: BurstColor?
    /// Frames seen during a photo-only station capture (no depth to count, so
    /// the burst length is tracked explicitly).
    private var burstRGBCount = 0
    /// Photo-only capture: one guided "station" photo per pose (named
    /// `key_<pose>`), uploaded as color frames since there are no depth poses.
    private var stationFrames: [ColorFrameCapture] = []
    /// Motion metric of the frame `burstColor` was grabbed from — we keep the
    /// stillest (sharpest) RGB frame of the burst, not just the first.
    private var burstColorMotion: Double = .greatestFiniteMagnitude
    private static let ciContext = CIContext()
    private var alignedSince: Date?
    private var lastAlignedAt: Date?
    /// When the current pose's burst started, for the capture timeout.
    private var captureStart: Date?

    // Stillness tracking: smoothed camera-in-face motion so we only lock a pose
    // (and grab its photo) when the phone + head are steady. Motion blur from
    // capturing mid-movement is what makes skin look noisily "textured".
    private var lastCamPosMM: SIMD3<Double>?
    private var lastYawDeg: Float?
    private var lastPitchDeg: Float?
    private var lastMotionTime: TimeInterval?
    private var motionMetric: Double?   // EMA of mm/s + 2·deg/s; nil until known

    private var viewportSize = CGSize(width: 390, height: 600)
    private var demoTimer: Timer?
    private var demoStart: Date?

    private let burstFrames = 8
    /// Fewest frames we'll still accept if a burst can't fill before timing out
    /// (steep profile angles can thin the depth stream).
    private let minBurstFrames = 3
    /// Give up filling a burst after this long and use/abort what we have, so a
    /// pose that can't return depth doesn't freeze the capture.
    private let captureTimeout: TimeInterval = 4.0
    private let yawTolDeg: Float = 12
    private let pitchTolDeg: Float = 10
    private let rollTolDeg: Float = 10
    private let minDistMM: Float = 250
    // TrueDepth degrades with range but stays usable past 60 cm (burst-averaging
    // recovers the extra noise). A profile must be framed from farther back than
    // a front view — the whole head has to fit — so the cap has to allow it, or
    // the profile poses can never clear the distance gate. (Server depth-trunc is
    // raised to match so the far side of the head isn't clipped at this range.)
    private let maxDistMM: Float = 620
    private let holdSeconds = 0.6
    /// How long alignment may briefly drop out without restarting the hold
    /// countdown — absorbs single-frame tracking/pose jitter.
    private let holdGraceSeconds = 0.35
    /// Combined camera-in-face motion (mm/s + 2·deg/s) must stay under this for
    /// a pose to count as held. Tuned to allow a steadily-held phone through
    /// while rejecting active turning/drift; raise it if poses won't lock.
    private let stillThreshold: Double = 55

    // MARK: free-orbit harvest grid
    //
    // After the guided depth keyframes, the operator slowly circles the face and
    // the app auto-grabs one quality-gated colour frame per yaw×pitch cell. These
    // are depth-LESS: the server projects them for texture only (no ICP/TSDF), so
    // the set can be large without slowing geometry. yaw spans ±90° in 15° steps
    // (index -6…+6); pitch snaps to three bands (below / level / above).
    static let orbitYawStepDeg: Float = 15
    static let orbitYawIndexRange = -6...6        // -90°…+90°
    static let orbitPitchBuckets: [Float] = [-25, 0, 25]
    static let orbitPitchCount = 3
    static func orbitCellKey(_ yi: Int, _ pi: Int) -> String { "y\(yi)_p\(pi)" }
    static var orbitTotalCells: Int { orbitYawIndexRange.count * orbitPitchCount }

    /// Backing store for `coverageCells` (mutated off the published value so we
    /// only publish on the main queue when a new cell actually fills). The grid
    /// is now just a COARSE progress map — actual capture is dense (below).
    private var coverage: Set<String> = []
    private var orbitColorFrames: [ColorFrameCapture] = []
    /// A harvested frame must be at least this sharp (variance of a Laplacian on
    /// the downsampled luma) — rejects motion-blurred/defocused frames so we can
    /// capture WHILE sweeping (no need to stop at each angle). TUNE ON DEVICE —
    /// Laplacian variance scales with content/exposure, so this is conservative.
    private let orbitSharpnessMin: Double = 8.0

    // DENSE photogrammetry harvest. We want as many sharp, overlapping frames as
    // possible, so the orbit grabs a frame wherever the camera is at least
    // `orbitMinStepDeg` from EVERY frame captured so far (a Poisson-disk in
    // yaw×pitch). That captures continuously as you sweep, yet rejects
    // near-duplicates when you pause or re-cross covered ground — bounded by
    // `maxOrbitFrames` for upload/storage. Smaller step = denser; raise the cap
    // for an even richer set (≈0.4 MB/frame).
    private let orbitMinStepDeg: Float = 4
    private let maxOrbitFrames = 180
    /// Reject only fast whips (the sharpness gate handles the rest), so frames
    /// can be grabbed mid-sweep without stopping.
    private let orbitMotionCeiling: Double = 180
    /// Yaw/pitch (deg) of every frame captured this orbit, for the spacing test.
    private var orbitYaws: [Float] = []
    private var orbitPitches: [Float] = []

    // Full-resolution stills (iOS 16 `captureHighResolutionFrame`, ~12 MP vs the
    // ~1.5-2 MP stream). One per newly-filled coverage cell, rate-limited; each
    // still carries its OWN ARFrame's pose + intrinsics so it drops into the
    // existing color_frames schema as `still_###`.
    private var stillFrames: [ColorFrameCapture] = []
    private var lastStillAt: Date?
    private let maxStillFrames = 30
    private let minStillInterval: TimeInterval = 1.5

    func setViewportSize(_ size: CGSize) { viewportSize = size }

    /// Turn the camera on and show a live preview WITHOUT the alignment guides
    /// or auto-shutter. The operator frames the subject, then taps Start (which
    /// asks for a patient ID and calls `beginGuidedCapture`).
    func startPreview() {
        guard let newBackend = mode.makeBackend() else {
            backend = nil
            statusText = unsupportedMessage
            return
        }
        backend = newBackend
        stopDemo()
        isDemo = false
        captured = []
        stationFrames = []
        burst = []
        alignedSince = nil
        lastAlignedAt = nil
        resetMotion()
        finishedSession = nil
        guidance = GuidanceState()
        let config = newBackend.makeConfiguration()
        session.delegate = self
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        phase = .preview
        statusText = "Frame the subject, then tap Start"
    }

    /// Begin the guided 5-pose capture for the given patient. Called after the
    /// operator taps Start and enters a patient ID. Reuses the already-running
    /// preview session.
    func beginGuidedCapture(patientId: String) {
        guard captureSupported else { return }
        self.patientId = patientId
        stopDemo()
        isDemo = false
        captured = []
        stationFrames = []
        burst = []
        alignedSince = nil
        lastAlignedAt = nil
        resetMotion()
        finishedSession = nil
        if session.delegate == nil { startPreview() }   // safety: ensure camera is live
        phase = .aligning(pose: .front)
        statusText = phrasing(Pose.front.instruction)
    }

    func cancel() {
        stopDemo()
        cues.stop()
        isDemo = false
        alignedSince = nil
        lastAlignedAt = nil
        resetMotion()
        // Drop back to a live preview (camera stays on) rather than a dead idle
        // screen, so the operator can immediately reframe and start again.
        if captureSupported {
            startPreview()
        } else {
            session.pause()
            phase = .idle
            statusText = "Tap Start"
        }
    }

    /// Called when the Capture tab appears: turn the camera on in preview mode.
    /// Guides/auto-shutter only begin once the operator taps Start.
    func autoStart() {
        guard captureSupported, phase == .idle, !isDemo else { return }
        startPreview()
    }

    /// Called when the Capture tab disappears: free the camera. A capture in
    /// progress is reset so re-opening the tab starts cleanly; a finished
    /// session is preserved (its preview just pauses).
    func leaveCaptureTab() {
        guard !isDemo else { return }
        stopDemo()
        cues.stop()
        session.pause()
        // A finished session is preserved (its preview just pauses). Anything in
        // progress (preview or mid-capture) is reset so re-opening the tab
        // starts cleanly with a fresh live preview.
        if phase != .done {
            captured = []
            burst = []
            alignedSince = nil
            lastAlignedAt = nil
            resetMotion()
            phase = .idle
            statusText = "Tap Start"
        }
    }

    // MARK: - Demo mode (no camera)

    /// Runs the full guided-capture experience without a TrueDepth sensor:
    /// scripts the alignment/hold/capture animation through the three poses,
    /// then writes a synthetic head session so the result shows up — and can
    /// be uploaded — just like a real capture. Lets the app be explored on the
    /// Simulator or any device without a front depth camera.
    func startDemo() {
        stopDemo()
        session.pause()
        isDemo = true
        captured = []
        burst = []
        finishedSession = nil
        guidance = GuidanceState()
        phase = .aligning(pose: .front)
        statusText = phrasing(Pose.front.instruction)
        demoStart = Date()
        demoTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0,
                                         repeats: true) { [weak self] _ in
            self?.demoTick()
        }
    }

    private func stopDemo() {
        demoTimer?.invalidate()
        demoTimer = nil
        demoStart = nil
    }

    private func demoTick() {
        guard let start = demoStart else { return }
        let t = Date().timeIntervalSince(start)
        let search = 0.9, capture = 0.5
        let perPose = search + holdSeconds + capture
        let idx = Int(t / perPose)
        if idx >= Pose.allCases.count { finishDemo(); return }

        let pose = Pose(rawValue: idx)!
        let local = t - Double(idx) * perPose
        let w = viewportSize.width, h = viewportSize.height
        let eyeY = h * 0.42, sep: CGFloat = 62, midX = w / 2

        if local < search {
            // Drift the eye line in from an off-target, slightly tilted start.
            let k = CGFloat(1 - local / search)        // 1 -> 0
            let slide = k * 70 * (idx % 2 == 0 ? 1 : -1)
            let tilt = k * 16
            guidance = GuidanceState(
                yawDeg: pose.targetYawDeg, pitchDeg: Float(tilt),
                rollDeg: Float(tilt), distanceMM: 360 + Float(k) * 130,
                eyeLeft: CGPoint(x: midX - sep + slide, y: eyeY + tilt),
                eyeRight: CGPoint(x: midX + sep + slide, y: eyeY - tilt),
                hasFace: t > 0.15, angleOK: true, aligned: false,
                expressionNeutral: true)
            phase = .aligning(pose: pose)
            statusText = phrasing(pose.instruction)
        } else if local < search + holdSeconds {
            guidance = alignedGuidance(pose: pose, midX: midX, eyeY: eyeY, sep: sep)
            phase = .holding(pose: pose, progress: (local - search) / holdSeconds)
            statusText = "Hold still…"
        } else {
            guidance = alignedGuidance(pose: pose, midX: midX, eyeY: eyeY, sep: sep)
            phase = .capturing(pose: pose)
            statusText = "Capturing…"
        }
    }

    private func alignedGuidance(pose: Pose, midX: CGFloat, eyeY: CGFloat,
                                 sep: CGFloat) -> GuidanceState {
        GuidanceState(
            yawDeg: pose.targetYawDeg, pitchDeg: 0, rollDeg: 0, distanceMM: 350,
            eyeLeft: CGPoint(x: midX - sep, y: eyeY),
            eyeRight: CGPoint(x: midX + sep, y: eyeY),
            hasFace: true, angleOK: true, aligned: true, expressionNeutral: true)
    }

    private func finishDemo() {
        stopDemo()
        phase = .done
        statusText = "Rendering demo scan…"
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let poses = DemoData.makePoses()
            do {
                let url = try SessionWriter.write(poses: poses, patientId: self?.patientId ?? "")
                DispatchQueue.main.async {
                    self?.finishedSession = url
                    self?.statusText =
                        "Demo scan saved ✓ — open Sessions to view or upload"
                }
            } catch {
                DispatchQueue.main.async {
                    self?.statusText = "Demo save failed: \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // The backend does the camera-specific work (face anchor or Vision head
        // anchor, depth, extrinsics); everything below is shared gating and
        // state-machine logic.
        guard let backend else { return }
        guard let sample = backend.subjectSample(for: frame,
                                                 viewportSize: viewportSize) else {
            DispatchQueue.main.async { self.guidance.hasFace = false }
            return
        }
        let yaw = sample.yawDeg
        let pitch = sample.pitchDeg
        let roll = sample.rollDeg
        let distMM = sample.distanceMM
        let neutral = sample.expressionNeutral
        let eyes = (sample.eyeLeft, sample.eyeRight)

        // Smoothed motion of the camera relative to the subject: linear speed
        // (mm/s) plus angular speed (deg/s, weighted) of the head pose. Used to
        // gate the hold so we only ever capture a steady frame.
        let camPosMM = sample.cameraPositionMM
        if let p0 = lastCamPosMM, let y0 = lastYawDeg, let pi0 = lastPitchDeg,
           let t0 = lastMotionTime, frame.timestamp > t0 {
            let dt = frame.timestamp - t0
            let lin = simd_length(camPosMM - p0) / dt
            let ang = (Double(abs(yaw - y0)) + Double(abs(pitch - pi0))) / dt
            let inst = lin + 2.0 * ang
            motionMetric = motionMetric.map { $0 * 0.7 + inst * 0.3 } ?? inst
        }
        lastCamPosMM = camPosMM
        lastYawDeg = yaw
        lastPitchDeg = pitch
        lastMotionTime = frame.timestamp
        let still = (motionMetric ?? .greatestFiniteMagnitude) < stillThreshold

        // Alignment + hold are evaluated while either ALIGNING or HOLDING the
        // pose. (Previously this only matched `.aligning`, so the first aligned
        // frame flipped the phase to `.holding` and every later frame bailed
        // out here — the hold countdown never advanced and capture never fired.)
        let pose: Pose
        switch phaseForProcessing() {
        case let .aligning(p): pose = p
        case let .holding(p, _): pose = p
        case .preview, .idle:
            // Camera is live but the operator hasn't tapped Start: show the feed
            // only, no guide lines and no auto-shutter.
            DispatchQueue.main.async { self.guidance.hasFace = true }
            return
        case .orbiting:
            // Free orbit: no single target pose. Show the live readouts and try
            // to harvest a colour frame for whatever yaw×pitch cell we're in.
            updateGuidance(yaw: yaw, pitch: pitch, roll: roll, dist: distMM,
                           eyes: eyes, angleOK: true, aligned: still,
                           neutral: neutral)
            harvestOrbitFrame(frame: frame, sample: sample,
                              yaw: yaw, pitch: pitch, still: still,
                              neutral: neutral, distMM: distMM)
            return
        default:   // .capturing, .done
            updateGuidance(yaw: yaw, pitch: pitch, roll: roll,
                           dist: distMM, eyes: eyes,
                           angleOK: true, aligned: false, neutral: neutral)
            collectBurstFrame(frame: frame, sample: sample)
            return
        }

        // A profile is shot one-handed at arm's length, which is harder to hold
        // perfectly level than a close-up front view — and the elevation is read
        // off the LOCKED face frame, which carries a little extra slack. Give the
        // profile poses a looser "Level" gate so it's actually reachable; the
        // small residual pitch is well within what the server fuse tolerates.
        let pitchTol = pose.needsLockedFrame ? pitchTolDeg + 5 : pitchTolDeg
        let yawTol = yawTolDeg + pose.yawTolBonusDeg + huntRelaxDeg(for: pose)
        let angleOK = abs(yaw - pose.targetYawDeg) < yawTol
        let aligned = angleOK
            && abs(pitch - pose.targetPitchDeg) < pitchTol
            && abs(roll) < rollTolDeg
            && distMM > minDistMM && distMM < maxDistMM
            && neutral
            && still
        updateGuidance(yaw: yaw, pitch: pitch, roll: roll,
                       dist: distMM, eyes: eyes,
                       angleOK: angleOK, aligned: aligned, neutral: neutral)

        if aligned {
            lastAlignedAt = Date()
            if alignedSince == nil { alignedSince = Date() }
            let held = Date().timeIntervalSince(alignedSince!)
            if held >= holdSeconds {
                beginBurst(pose: pose)
            } else {
                cues.holding()
                DispatchQueue.main.async {
                    self.phase = .holding(pose: pose, progress: held / self.holdSeconds)
                    self.statusText = "Hold still…"
                }
            }
        } else {
            // Hunt-beep: direction by pitch, tempo by closeness on the dominant
            // unmet axis (only matters while neutral — otherwise the gate is the
            // expression, which the on-screen hint covers).
            cues.updateAlignment(yawErr: yaw - pose.targetYawDeg,
                                 pitchErr: pitch - pose.targetPitchDeg,
                                 aligned: false)
            // Ignore a brief dropout (jitter) — keep the countdown running
            // through the grace window so the hold doesn't restart on noise.
            let inGrace = lastAlignedAt.map {
                Date().timeIntervalSince($0) < holdGraceSeconds } ?? false
            if !inGrace {
                alignedSince = nil
                DispatchQueue.main.async {
                    if case .holding = self.phase { self.phase = .aligning(pose: pose) }
                    self.statusText = self.phrasing(self.alignmentHint(
                        pose: pose, yaw: yaw, pitch: pitch, roll: roll,
                        distMM: distMM, neutral: neutral, still: still,
                        subjectDriftMM: sample.subjectDriftMM))
                }
            }
        }
    }

    // MARK: - capture internals

    /// ARSession delivers delegate callbacks on the main queue (we never set
    /// `delegateQueue`), so reading published state directly is safe here —
    /// a main.sync would deadlock.
    private func phaseForProcessing() -> Phase { phase }

    private func beginBurst(pose: Pose) {
        alignedSince = nil
        lastAlignedAt = nil
        burst = []
        burstGeometry = nil
        burstColor = nil
        burstRGBCount = 0
        burstColorMotion = .greatestFiniteMagnitude
        captureStart = Date()
        DispatchQueue.main.async {
            self.phase = .capturing(pose: pose)
            self.statusText = "Capturing…"
        }
    }

    private func collectBurstFrame(frame: ARFrame, sample: SubjectSample) {
        guard case let .capturing(pose) = phaseForProcessing(),
              let backend else { return }

        // Photo-only rear capture (no LiDAR): the "burst" is just picking the
        // stillest RGB frame at this station — there is no depth to accumulate.
        let photoOnly = !backend.providesDepth

        // Don't wait forever for a full burst. At steep profile angles the
        // depth stream can thin out (or, if the OS withholds depth without a
        // tracked face, stop entirely) — finish with the frames we have if
        // there are enough, otherwise drop back to aligning with a hint rather
        // than freezing the whole capture on this pose.
        if let start = captureStart, Date().timeIntervalSince(start) > captureTimeout {
            if photoOnly, burstColor != nil {
                finishStationPhoto(pose: pose)
            } else if burst.count >= minBurstFrames, burstGeometry != nil {
                finishBurst(pose: pose)
            } else {
                abortBurst(pose: pose)
            }
            return
        }

        if photoOnly {
            burstRGBCount += 1
        } else {
            guard let depth = backend.depthSample(for: frame) else { return }
            burst.append(depth.depthMM)
            if burstGeometry == nil {
                burstGeometry = (depth.width, depth.height,
                                 depth.fx, depth.fy, depth.cx, depth.cy,
                                 sample.worldToCameraCV)
            }
        }

        // Keep the RGB frame from the STILLEST moment of the burst (lowest
        // motion → least motion blur), not just the first — sharper texture.
        let motion = motionMetric ?? .greatestFiniteMagnitude
        if motion < burstColorMotion, let jpeg = Self.jpegData(from: frame.capturedImage) {
            let res = frame.camera.imageResolution
            let k = frame.camera.intrinsics
            burstColor = BurstColor(
                jpeg: jpeg, w: Int(res.width), h: Int(res.height),
                fx: k.columns.0.x, fy: k.columns.1.y,
                cx: k.columns.2.x, cy: k.columns.2.y,
                worldToCamera: sample.worldToCameraCV)
            burstColorMotion = motion
        }

        if photoOnly {
            if burstRGBCount >= burstFrames, burstColor != nil {
                finishStationPhoto(pose: pose)
            }
        } else if burst.count >= burstFrames {
            finishBurst(pose: pose)
        }
    }

    private func finishBurst(pose: Pose) {
        guard let geo = burstGeometry, let first = burst.first else { return }
        var avg = [Float](repeating: 0, count: first.count)
        var counts = [Float](repeating: 0, count: first.count)
        for frame in burst {
            for i in 0..<frame.count where frame[i] > 0 {
                avg[i] += frame[i]
                counts[i] += 1
            }
        }
        // Require the pixel valid in most frames; flickering edge pixels are
        // exactly the unreliable ones. Scaled to the frames we actually got
        // (a timed-out profile burst may hold fewer than `burstFrames`).
        let minCount = max(1.0, Float(burst.count) * 0.6)
        for i in 0..<avg.count {
            avg[i] = counts[i] >= minCount ? avg[i] / counts[i] : 0
        }
        captured.append(CapturedPose(
            name: pose.name, depthMM: avg, width: geo.w, height: geo.h,
            fx: geo.fx, fy: geo.fy, cx: geo.cx, cy: geo.cy,
            worldToCamera: geo.worldToCamera,
            colorJPEG: burstColor?.jpeg,
            rgbWidth: burstColor?.w ?? 0, rgbHeight: burstColor?.h ?? 0,
            rgbFx: burstColor?.fx ?? 0, rgbFy: burstColor?.fy ?? 0,
            rgbCx: burstColor?.cx ?? 0, rgbCy: burstColor?.cy ?? 0))
        burst = []
        burstGeometry = nil
        burstColor = nil
        burstRGBCount = 0
        captureStart = nil
        advanceFromCapturedPose(pose)
    }

    /// Photo-only station capture: save the stillest RGB frame of the hold as
    /// a `key_<pose>` color frame (there is no depth pose to write), request a
    /// high-res still from the same vantage, then advance.
    private func finishStationPhoto(pose: Pose) {
        guard let color = burstColor else { return }
        stationFrames.append(ColorFrameCapture(
            name: "key_\(pose.name)", jpeg: color.jpeg,
            width: color.w, height: color.h,
            fx: color.fx, fy: color.fy, cx: color.cx, cy: color.cy,
            worldToCamera: color.worldToCamera))
        burst = []
        burstGeometry = nil
        burstColor = nil
        burstRGBCount = 0
        burstColorMotion = .greatestFiniteMagnitude
        captureStart = nil
        captureStill()
        advanceFromCapturedPose(pose)
    }

    /// Shared tail of a completed keyframe (depth burst or station photo):
    /// freeze the rear head anchor after the front pose, then advance to the
    /// next pose, or into the free-orbit phase after the last one. The orbit
    /// is where dense colour frames are auto-harvested; the operator ends it
    /// (or it ends itself) and only THEN do we save.
    private func advanceFromCapturedPose(_ pose: Pose) {
        if pose == .front { backend?.didCaptureFrontPose() }
        cues.captured()
        DispatchQueue.main.async {
            if let next = Pose(rawValue: pose.rawValue + 1) {
                self.phase = .aligning(pose: next)
                self.statusText = self.phrasing(next.instruction)
            } else {
                self.beginOrbit()
            }
        }
    }

    // MARK: - free-orbit harvest

    /// Enter the free-orbit phase: the operator slowly circles the face while we
    /// auto-grab one colour frame per yaw×pitch cell. Camera keeps running.
    ///
    /// AE/AWB/focus are LOCKED for the duration of the orbit via iOS 16's
    /// `configurableCaptureDeviceForPrimaryCamera` — the supported way to reach
    /// the capture device under ARKit. Exposure drift across the sweep was the
    /// main source of texture seams in the projected atlas.
    private func beginOrbit() {
        coverage = []
        orbitColorFrames = []
        stillFrames = []
        lastStillAt = nil
        orbitYaws = []
        orbitPitches = []
        coverageCells = []
        orbitCapturedCount = 0
        alignedSince = nil
        lastAlignedAt = nil
        phase = .orbiting
        cues.stop()   // end the hold/hunt cues; orbit uses per-frame ticks
        backend?.setCameraLocked(true)
        statusText = phrasing(
            "Last step: slowly sweep the phone around their face — it captures as you move")
    }

    /// Called from the UI when the operator taps Done during the orbit phase.
    func finishOrbit() {
        guard phase == .orbiting else { return }
        backend?.setCameraLocked(false)
        session.pause()
        phase = .done
        statusText = "Saving session…"
        saveSession()
    }

    /// Densely harvest colour frames for photogrammetry: grab a frame wherever
    /// the camera is at least `orbitMinStepDeg` from EVERY frame already taken
    /// this orbit (a Poisson-disk in yaw×pitch), gated by distance, neutral
    /// expression, a loose motion ceiling and a sharpness floor. This captures
    /// continuously as the phone sweeps, while the spacing test rejects
    /// near-duplicates when you pause or re-cross covered ground. Bounded by
    /// `maxOrbitFrames` for upload/storage.
    private func harvestOrbitFrame(frame: ARFrame, sample: SubjectSample,
                                   yaw: Float, pitch: Float, still: Bool,
                                   neutral: Bool, distMM: Float) {
        guard orbitColorFrames.count < maxOrbitFrames else { return }
        guard neutral, distMM > minDistMM, distMM < maxDistMM else { return }
        // Reject fast whips (the sharpness gate handles residual blur); capturing
        // mid-sweep is fine, so we use a loose ceiling rather than full stillness.
        let motion = motionMetric ?? .greatestFiniteMagnitude
        guard motion < orbitMotionCeiling else { return }
        // Spatial diversity: skip if within orbitMinStepDeg of any captured frame.
        let minStepSq = orbitMinStepDeg * orbitMinStepDeg
        for i in 0..<orbitYaws.count {
            let dy = yaw - orbitYaws[i], dp = pitch - orbitPitches[i]
            if dy * dy + dp * dp < minStepSq { return }
        }
        guard Self.sharpness(of: frame.capturedImage) >= orbitSharpnessMin else { return }
        guard let jpeg = Self.jpegData(from: frame.capturedImage) else { return }

        let res = frame.camera.imageResolution
        let k = frame.camera.intrinsics
        let idx = orbitColorFrames.count
        let frameCapture = ColorFrameCapture(
            name: String(format: "orbit_%03d", idx), jpeg: jpeg,
            width: Int(res.width), height: Int(res.height),
            fx: k.columns.0.x, fy: k.columns.1.y,
            cx: k.columns.2.x, cy: k.columns.2.y,
            worldToCamera: sample.worldToCameraCV)

        orbitYaws.append(yaw)
        orbitPitches.append(pitch)
        orbitColorFrames.append(frameCapture)
        // Light up the coarse grid cell (progress map only — not a capture limit);
        // a NEWLY filled cell also triggers a full-res still from that vantage.
        if let (yi, pi) = orbitBucket(yaw: yaw, pitch: pitch),
           coverage.insert(Self.orbitCellKey(yi, pi)).inserted {
            captureStill()
        }
        let count = orbitColorFrames.count
        let cells = coverage
        let full = count >= maxOrbitFrames
        if full { cues.orbitFull() } else { cues.orbitTick() }
        DispatchQueue.main.async {
            self.coverageCells = cells
            self.orbitCapturedCount = count
            self.statusText = full
                ? "Coverage full (\(count) photos) — tap Done"
                : "Capturing as you sweep — \(count) photos"
        }
    }

    /// Grab a ~12 MP still (vs the ~2 MP stream) from the current vantage. The
    /// completion's ARFrame carries the still's OWN camera pose + intrinsics, so
    /// its extrinsics are exact even though it lands ~100 ms after the trigger
    /// (the subject anchor is world-locked and the head is held still).
    private func captureStill() {
        guard #available(iOS 16.0, *),
              let backend, backend.stillCaptureSupported else { return }
        guard stillFrames.count < maxStillFrames else { return }
        if let last = lastStillAt, Date().timeIntervalSince(last) < minStillInterval {
            return
        }
        lastStillAt = Date()
        session.captureHighResolutionFrame { [weak self] frame, error in
            guard let self, let frame else {
                if let error { print("[capture] still failed: \(error)") }
                return
            }
            guard let jpeg = Self.jpegData(from: frame.capturedImage, quality: 0.9),
                  self.stillFrames.count < self.maxStillFrames,
                  let worldToCamera = self.backend?.worldToCameraCV(for: frame)
            else { return }
            let res = frame.camera.imageResolution
            let k = frame.camera.intrinsics
            let cf = ColorFrameCapture(
                name: String(format: "still_%03d", self.stillFrames.count), jpeg: jpeg,
                width: Int(res.width), height: Int(res.height),
                fx: k.columns.0.x, fy: k.columns.1.y,
                cx: k.columns.2.x, cy: k.columns.2.y,
                worldToCamera: worldToCamera)
            self.stillFrames.append(cf)
            print("[capture] still \(self.stillFrames.count) "
                  + "@ \(Int(res.width))×\(Int(res.height))")
        }
    }

    /// Map a camera yaw/pitch (degrees) to a coverage grid cell, or nil if it
    /// falls outside the captured range. yaw → nearest 15° index in ±90°; pitch
    /// → nearest of the three bands, rejected if more than ~18° off any band.
    private func orbitBucket(yaw: Float, pitch: Float) -> (yi: Int, pi: Int)? {
        let yi = Int((yaw / Self.orbitYawStepDeg).rounded())
        guard Self.orbitYawIndexRange.contains(yi) else { return nil }
        var pi = 0
        var bestD = Float.greatestFiniteMagnitude
        for (i, band) in Self.orbitPitchBuckets.enumerated() {
            let d = abs(pitch - band)
            if d < bestD { bestD = d; pi = i }
        }
        guard bestD <= 18 else { return nil }
        return (yi, pi)
    }

    /// Variance of a Laplacian over the (subsampled) luma plane — a cheap focus
    /// metric. Higher = sharper. Reads the Y plane of ARKit's biplanar YCbCr
    /// frame directly; samples a centred window every few pixels.
    private static func sharpness(of pixelBuffer: CVPixelBuffer) -> Double {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else {
            return .greatestFiniteMagnitude   // can't measure → don't block
        }
        let w = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let h = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        let stride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let ptr = base.assumingMemoryBound(to: UInt8.self)
        let step = 4
        let x0 = w / 4, x1 = 3 * w / 4
        let y0 = h / 4, y1 = 3 * h / 4
        guard x1 - x0 > 2 * step, y1 - y0 > 2 * step else {
            return .greatestFiniteMagnitude
        }
        var sum = 0.0, sumSq = 0.0, n = 0.0
        var y = y0 + step
        while y < y1 - step {
            var x = x0 + step
            while x < x1 - step {
                let c = Int(ptr[y * stride + x])
                let up = Int(ptr[(y - step) * stride + x])
                let dn = Int(ptr[(y + step) * stride + x])
                let lf = Int(ptr[y * stride + x - step])
                let rt = Int(ptr[y * stride + x + step])
                let lap = Double(4 * c - up - dn - lf - rt)
                sum += lap
                sumSq += lap * lap
                n += 1
                x += step
            }
            y += step
        }
        guard n > 1 else { return .greatestFiniteMagnitude }
        let mean = sum / n
        return sumSq / n - mean * mean
    }

    /// A burst that couldn't gather enough depth (e.g. the sensor returned no
    /// frames at this profile angle): discard it and re-arm the SAME pose so the
    /// operator can settle and try again, rather than freezing on it.
    private func abortBurst(pose: Pose) {
        burst = []
        burstGeometry = nil
        burstColor = nil
        burstRGBCount = 0
        burstColorMotion = .greatestFiniteMagnitude
        captureStart = nil
        alignedSince = nil
        lastAlignedAt = nil
        DispatchQueue.main.async {
            self.phase = .aligning(pose: pose)
            self.statusText = "Couldn't read depth — hold steady and reframe"
        }
    }

    private func saveSession() {
        do {
            let url = try SessionWriter.write(
                poses: captured,
                colorFrames: stationFrames + orbitColorFrames + stillFrames,
                patientId: patientId,
                device: backend?.deviceTag ?? "iphone-truedepth",
                photoOnly: backend.map { !$0.providesDepth } ?? false)
            finishedSession = url
            statusText = "Captured ✓ — ready to upload from Sessions tab"
            cues.done()
        } catch {
            cues.stop()
            statusText = "Save failed: \(error.localizedDescription)"
        }
    }

    /// JPEG-encode an ARFrame's RGB buffer (camera-native landscape orientation).
    private static func jpegData(from pixelBuffer: CVPixelBuffer,
                                 quality: Double = 0.9) -> Data? {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        return ciContext.jpegRepresentation(
            of: image, colorSpace: CGColorSpaceCreateDeviceRGB(),
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: quality])
    }

    private func updateGuidance(yaw: Float, pitch: Float, roll: Float,
                                dist: Float, eyes: (CGPoint, CGPoint),
                                angleOK: Bool, aligned: Bool, neutral: Bool) {
        DispatchQueue.main.async {
            self.guidance = GuidanceState(
                yawDeg: yaw, pitchDeg: pitch, rollDeg: roll, distanceMM: dist,
                eyeLeft: eyes.0, eyeRight: eyes.1, hasFace: true,
                angleOK: angleOK, aligned: aligned, expressionNeutral: neutral)
        }
    }

    private func alignmentHint(pose: Pose, yaw: Float, pitch: Float,
                               roll: Float, distMM: Float, neutral: Bool,
                               still: Bool, subjectDriftMM: Float = 0) -> String {
        // Rear capture: the head anchor was frozen at the front pose, so a
        // subject who shifts afterwards silently corrupts every extrinsic.
        // The backend measures the drift whenever Vision re-sees the face.
        if subjectDriftMM > 20 { return "The subject moved — ask them to hold still" }
        if !neutral { return "Relax your face (neutral expression)" }
        if distMM < minDistMM { return "Move the phone a little farther away" }
        if distMM > maxDistMM { return "Move the phone a little closer" }
        // Phone-orbit: the subject's head stays still and the PHONE moves around
        // it. yaw is the camera's angle in the face frame (subject-right = +).
        // dyaw > 0 means the camera is too far toward the subject's right, so it
        // must move back toward their left to reach the target (and vice versa).
        let dyaw = yaw - pose.targetYawDeg
        if abs(dyaw) >= yawTolDeg {
            let dir = dyaw > 0 ? "left" : "right"
            return abs(dyaw) > 20 ? "Keep moving the phone to their \(dir)"
                                  : "Move the phone a little to their \(dir)"
        }
        if abs(roll) >= rollTolDeg { return "Hold the phone level" }
        let pitchTol = pose.needsLockedFrame ? pitchTolDeg + 5 : pitchTolDeg
        let dpitch = pitch - pose.targetPitchDeg
        if abs(dpitch) >= pitchTol {
            // dpitch > 0 means the camera sits higher than the pose wants.
            return dpitch > 0 ? "Lower the phone a little" : "Raise the phone a little"
        }
        // On target but still moving: the only thing left is to settle.
        if !still { return "Hold steady…" }
        return pose.instruction
    }

    // Auto-relax: a pose that's been hunted for >10 s gently widens its yaw
    // gate (up to +6°) instead of stonewalling. A long hunt drags out the whole
    // session, and session length is itself a capture-quality risk — expression
    // drift across the photo set is what melts Object Capture geometry.
    private var huntPose: Pose?
    private var huntStart: Date?
    private func huntRelaxDeg(for pose: Pose) -> Float {
        if huntPose != pose {
            huntPose = pose
            huntStart = Date()
        }
        let hunting = Date().timeIntervalSince(huntStart ?? Date())
        return Float(min(max(hunting - 10, 0), 12) / 2)
    }

    /// Clears the smoothed-motion state between captures so a stale velocity
    /// from a previous pose/session can't briefly pass the stillness gate.
    private func resetMotion() {
        lastCamPosMM = nil
        lastYawDeg = nil
        lastPitchDeg = nil
        lastMotionTime = nil
        motionMetric = nil
        backend?.reset()
        captureStart = nil
        coverage = []
        orbitColorFrames = []
        orbitYaws = []
        orbitPitches = []
        coverageCells = []
        orbitCapturedCount = 0
    }
}

extension simd_double4x4 {
    init(_ m: simd_float4x4) {
        self.init(
            SIMD4<Double>(m.columns.0), SIMD4<Double>(m.columns.1),
            SIMD4<Double>(m.columns.2), SIMD4<Double>(m.columns.3))
    }

    /// ARKit transforms are in meters; the pipeline expects millimeters.
    func scaledTranslationMM() -> simd_double4x4 {
        var out = self
        out.columns.3 = SIMD4<Double>(columns.3.x * 1000, columns.3.y * 1000,
                                      columns.3.z * 1000, 1)
        return out
    }
}

extension SIMD4 where Scalar == Double {
    init(_ v: SIMD4<Float>) {
        self.init(Double(v.x), Double(v.y), Double(v.z), Double(v.w))
    }
}

/// Generates a synthetic 3-pose "head" capture with no camera, matching the
/// processing pipeline's conventions exactly (640x480 depth in mm, OpenCV
/// `world_to_camera` extrinsics, cameras orbiting the origin at 350 mm). The
/// head is a union of ellipsoids (skull + nose/brow/chin/cheeks) intersected
/// analytically per ray, so the depth is exact and the session fuses on the
/// server like a real one.
enum DemoData {
    private struct Ellipsoid { let c: SIMD3<Double>; let r: SIMD3<Double> }

    // Sensor model shared with vectra3d/cameras.py.
    private static let width = 640, height = 480
    private static let fx = 580.0, fy = 580.0
    private static let captureDistance = 350.0
    // Mirror CaptureController.Pose: front + a 3/4 view and a near-profile each
    // side (front first — the server treats poses[0] as the reference).
    private static let yawsDeg: [Double] = [0, -35, -72, 35, 72]
    private static let names = ["front", "left_half", "left", "right_half", "right"]

    private static let head: [Ellipsoid] = [
        Ellipsoid(c: SIMD3<Double>(0, 0, 0),     r: SIMD3<Double>(75, 105, 85)),  // skull
        Ellipsoid(c: SIMD3<Double>(0, -8, 72),   r: SIMD3<Double>(13, 20, 26)),   // nose
        Ellipsoid(c: SIMD3<Double>(0, 40, 74),   r: SIMD3<Double>(42, 14, 16)),   // brow
        Ellipsoid(c: SIMD3<Double>(0, -88, 52),  r: SIMD3<Double>(26, 22, 30)),   // chin
        Ellipsoid(c: SIMD3<Double>(38, -28, 68), r: SIMD3<Double>(26, 26, 26)),   // cheek L
        Ellipsoid(c: SIMD3<Double>(-38, -28, 68), r: SIMD3<Double>(26, 26, 26)),  // cheek R
    ]

    static func makePoses() -> [CapturedPose] {
        let cx = Double(width - 1) / 2, cy = Double(height - 1) / 2
        var out: [CapturedPose] = []
        for (i, yawDeg) in yawsDeg.enumerated() {
            let yaw = yawDeg * .pi / 180
            let camPos = SIMD3<Double>(captureDistance * sin(yaw), 0,
                                       captureDistance * cos(yaw))
            let (rW2C, rC2W, t) = lookAt(camPos: camPos)

            var depth = [Float](repeating: 0, count: width * height)
            for v in 0..<height {
                for u in 0..<width {
                    let dCam = simd_normalize(SIMD3<Double>(
                        (Double(u) - cx) / fx, (Double(v) - cy) / fy, 1))
                    let dWorld = rC2W * dCam
                    var best = Double.greatestFiniteMagnitude
                    for e in head {
                        if let s = intersect(origin: camPos, dir: dWorld, e: e),
                           s < best { best = s }
                    }
                    if best < Double.greatestFiniteMagnitude {
                        let pWorld = camPos + dWorld * best
                        let z = (rW2C * pWorld + t).z
                        if z > 0 { depth[v * width + u] = Float(z) }
                    }
                }
            }
            out.append(CapturedPose(
                name: names[i], depthMM: depth, width: width, height: height,
                fx: Float(fx), fy: Float(fy), cx: Float(cx), cy: Float(cy),
                worldToCamera: matrix(rW2C: rW2C, t: t)))
        }
        return out
    }

    /// Mirrors cameras.look_at_extrinsic: camera at `camPos` looking at the
    /// origin, OpenCV axes (x right, y down, z forward).
    private static func lookAt(camPos: SIMD3<Double>)
        -> (rW2C: simd_double3x3, rC2W: simd_double3x3, t: SIMD3<Double>) {
        let up = SIMD3<Double>(0, 1, 0)
        let forward = simd_normalize(-camPos)            // target is the origin
        let right = simd_normalize(simd_cross(-up, forward))
        let down = simd_cross(forward, right)
        let rC2W = simd_double3x3(columns: (right, down, forward))
        let rW2C = rC2W.transpose
        let t = -(rW2C * camPos)
        return (rW2C, rC2W, t)
    }

    /// Nearest positive ray/ellipsoid intersection distance, or nil.
    private static func intersect(origin: SIMD3<Double>, dir: SIMD3<Double>,
                                  e: Ellipsoid) -> Double? {
        let o = (origin - e.c) / e.r
        let d = dir / e.r
        let a = simd_dot(d, d)
        let b = simd_dot(o, d)
        let c = simd_dot(o, o) - 1
        let disc = b * b - a * c
        if disc < 0 { return nil }
        let sq = disc.squareRoot()
        let s0 = (-b - sq) / a
        if s0 > 1e-6 { return s0 }
        let s1 = (-b + sq) / a
        return s1 > 1e-6 ? s1 : nil
    }

    private static func matrix(rW2C: simd_double3x3,
                               t: SIMD3<Double>) -> simd_double4x4 {
        var m = matrix_identity_double4x4
        m.columns.0 = SIMD4<Double>(rW2C.columns.0.x, rW2C.columns.0.y, rW2C.columns.0.z, 0)
        m.columns.1 = SIMD4<Double>(rW2C.columns.1.x, rW2C.columns.1.y, rW2C.columns.1.z, 0)
        m.columns.2 = SIMD4<Double>(rW2C.columns.2.x, rW2C.columns.2.y, rW2C.columns.2.z, 0)
        m.columns.3 = SIMD4<Double>(t.x, t.y, t.z, 1)
        return m
    }
}

// MARK: - Eyes-free capture feedback (tones + haptics, no speech)

/// Tiny sine-tone player. Buffers are synthesised on demand (with short fades to
/// avoid clicks) and cached by frequency+duration, so a known pitch can be played
/// repeatedly with no assets and minimal latency.
private final class ToneEngine {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1)!
    private var cache: [Int: AVAudioPCMBuffer] = [:]
    private var running = false

    init() {
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
    }

    private func ensureRunning() {
        guard !running else { return }
        do { try engine.start(); player.play(); running = true }
        catch { running = false }
    }

    private func buffer(freq: Double, ms: Double) -> AVAudioPCMBuffer {
        let key = Int(freq) * 10_000 + Int(ms)
        if let b = cache[key] { return b }
        let n = AVAudioFrameCount(format.sampleRate * ms / 1000)
        let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: n)!
        buf.frameLength = n
        let ch = buf.floatChannelData![0]
        let w = 2.0 * Double.pi * freq / format.sampleRate
        let fade = format.sampleRate * 0.008   // 8 ms ramp in/out
        for i in 0..<Int(n) {
            var amp = 0.22
            if Double(i) < fade { amp *= Double(i) / fade }
            let tail = Double(Int(n) - i)
            if tail < fade { amp *= tail / fade }
            ch[i] = Float(sin(w * Double(i)) * amp)
        }
        cache[key] = buf
        return buf
    }

    func play(freq: Double, ms: Double = 70) {
        ensureRunning()
        guard running else { return }
        player.scheduleBuffer(buffer(freq: freq, ms: ms), at: nil)
    }

    func stop() {
        guard running else { return }
        player.stop(); engine.stop(); running = false
    }
}

/// Eyes-free capture feedback: direction is encoded in tone PITCH (which way to
/// move the phone), proximity in tone TEMPO (how close to the target), so a user
/// shooting their own profile — phone turned away, screen unseen — can home in by
/// ear. Discrete cues confirm each capture and completion. No-op when disabled.
final class CaptureCues {
    var enabled = true { didSet { if !enabled { stop() } } }

    private let tone = ToneEngine()
    private let impact = UIImpactFeedbackGenerator(style: .light)
    private let notify = UINotificationFeedbackGenerator()

    private var beepTimer: Timer?
    private var beepFreq: Double = 0
    private var beepInterval: TimeInterval = 0
    private var lastHoldTick = Date.distantPast

    init() {
        // Mix + duck other audio so cues are audible even with music playing and
        // through the silent switch while actively capturing.
        try? AVAudioSession.sharedInstance().setCategory(
            .playback, options: [.mixWithOthers, .duckOthers])
        try? AVAudioSession.sharedInstance().setActive(true)
        impact.prepare(); notify.prepare()
    }

    /// While aligning: pick a frequency (direction of the dominant unmet axis) and
    /// a repeat interval (shrinking as the axis nears tolerance), and run the beep.
    /// Errors are signed degrees; `aligned` silences the hunt beep.
    func updateAlignment(yawErr: Float, pitchErr: Float, aligned: Bool) {
        guard enabled else { return }
        if aligned { stopBeep(); return }
        let yawTol: Float = 6, pitchTol: Float = 8
        let freq: Double
        let closeness: Float   // 0 (far) … 1 (at tolerance edge)
        if abs(yawErr) > yawTol {
            freq = yawErr > 0 ? 523 : 880                 // low vs high = which way
            closeness = max(0, 1 - (abs(yawErr) - yawTol) / 60)
        } else if abs(pitchErr) > pitchTol {
            freq = pitchErr > 0 ? 392 : 698               // lower vs raise the phone
            closeness = max(0, 1 - (abs(pitchErr) - pitchTol) / 40)
        } else {
            freq = 330; closeness = 0.4                   // fine-tune (level/distance)
        }
        setBeep(freq: freq, interval: 0.55 - 0.42 * TimeInterval(closeness))
    }

    /// Locked on, counting down the hold: quick steady ticks (no hunt beep).
    func holding() {
        guard enabled else { return }
        stopBeep()
        let now = Date()
        guard now.timeIntervalSince(lastHoldTick) > 0.16 else { return }
        lastHoldTick = now
        tone.play(freq: 1046, ms: 32)
    }

    /// A pose's depth burst finished — the key "got it, move on" confirmation.
    func captured() {
        guard enabled else { return }
        stopBeep()
        notify.notificationOccurred(.success)
        tone.play(freq: 880, ms: 90)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            self?.tone.play(freq: 1320, ms: 120)
        }
    }

    /// One dense orbit frame harvested: a light blip so progress is felt.
    func orbitTick() {
        guard enabled else { return }
        impact.impactOccurred(intensity: 0.5)
        tone.play(freq: 1500, ms: 26)
    }

    func orbitFull() {
        guard enabled else { return }
        notify.notificationOccurred(.warning)
        tone.play(freq: 660, ms: 150)
    }

    /// Scan saved — a short rising three-note flourish.
    func done() {
        guard enabled else { return }
        stopBeep()
        notify.notificationOccurred(.success)
        let notes: [Double] = [660, 880, 1175]
        for (i, f) in notes.enumerated() {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.13 * Double(i)) { [weak self] in
                self?.tone.play(freq: f, ms: i == notes.count - 1 ? 200 : 110)
            }
        }
    }

    func stop() { stopBeep(); tone.stop() }

    private func setBeep(freq: Double, interval: TimeInterval) {
        if beepTimer != nil, beepFreq == freq, abs(beepInterval - interval) < 0.02 { return }
        beepFreq = freq; beepInterval = interval
        beepTimer?.invalidate()
        tone.play(freq: freq, ms: 55)   // beep immediately, then on the timer
        beepTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            guard let self, self.enabled else { return }
            self.tone.play(freq: self.beepFreq, ms: 55)
        }
    }

    private func stopBeep() {
        beepTimer?.invalidate(); beepTimer = nil
        beepFreq = 0; beepInterval = 0
    }
}
