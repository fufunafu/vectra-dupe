import ARKit
import SwiftUI

struct CaptureView: View {
    @StateObject private var controller = CaptureController()
    @State private var modelToView: IdentifiedURL?
    @State private var askingPatientID = false
    @State private var patientIDField = ""
    /// Operator mode: photographing someone else. Mirrors the preview so the
    /// subject reads naturally to the operator, and enlarges the live readouts
    /// for arm's-length use. Purely cosmetic — depth capture (front TrueDepth)
    /// and the alignment gates are unaffected. Selfie mode is the mirrored
    /// self-view people expect from a front camera.
    @AppStorage("captureOperatorMode") private var operatorMode = false
    /// Eyes-free tone + haptic guidance during capture (default on). Toggle lives
    /// in Settings; mirrored here via the shared UserDefaults key.
    @AppStorage("soundGuidance") private var soundGuidance = true

    /// Horizontal flip applied to the camera + guide layer. Selfie = mirrored
    /// (familiar self-view); operator = un-mirrored (a real view of the subject).
    private var previewFlipX: CGFloat { operatorMode ? 1 : -1 }

    var body: some View {
        GeometryReader { geo in
            ZStack {
                ARPreview(session: controller.session)
                    .ignoresSafeArea()
                    .scaleEffect(x: previewFlipX, y: 1)

                // Legibility scrims top and bottom over the live camera.
                VStack {
                    LinearGradient(colors: [.black.opacity(0.55), .clear],
                                   startPoint: .top, endPoint: .bottom)
                        .frame(height: 220)
                    Spacer()
                    LinearGradient(colors: [.clear, .black.opacity(0.65)],
                                   startPoint: .top, endPoint: .bottom)
                        .frame(height: 320)
                }
                .ignoresSafeArea()

                if controller.phase == .orbiting {
                    // Free orbit: no eye-line guide (the face anchor is gone at
                    // wide angles). A live yaw direction nudge instead.
                    OrbitGuideOverlay(yawDeg: controller.guidance.yawDeg)
                        .allowsHitTesting(false)
                } else if controller.phase.isActive {
                    // Flipped with the SAME transform as the camera so the eye
                    // markers stay registered to the mirrored video.
                    GuideOverlay(guidance: controller.guidance, phase: controller.phase)
                        .scaleEffect(x: previewFlipX, y: 1)
                        .allowsHitTesting(false)
                }

                VStack(spacing: 0) {
                    HStack {
                        Spacer()
                        modeToggle
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, controller.phase.isActive ? 56 : 12)

                    if controller.phase.isActive {
                        PoseStepper(phase: controller.phase)
                            .padding(.horizontal, 20)
                            .padding(.top, 8)
                    }
                    Spacer()
                    statusCard
                        .padding(.horizontal, 16)
                        .padding(.bottom, 8)
                }
            }
            .onAppear {
                controller.setViewportSize(fullScreenSize(geo))
                controller.isSelfie = !operatorMode
                controller.cues.enabled = soundGuidance
                controller.autoStart()
            }
            .onDisappear { controller.leaveCaptureTab() }
            .onChange(of: geo.size) { _, _ in controller.setViewportSize(fullScreenSize(geo)) }
            .onChange(of: operatorMode) { _, isOperator in controller.isSelfie = !isOperator }
            .onChange(of: soundGuidance) { _, on in controller.cues.enabled = on }
            .sheet(item: $modelToView) { item in
                NavigationStack {
                    Model3DView(sessionDir: item.url)
                        .toolbar {
                            ToolbarItem(placement: .topBarLeading) {
                                Button("Done") { modelToView = nil }
                            }
                        }
                }
                .preferredColorScheme(.dark)
            }
            .alert("New capture session", isPresented: $askingPatientID) {
                TextField("Patient ID (no spaces)", text: $patientIDField)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                Button("Start") { beginCapture() }
                Button("Cancel", role: .cancel) { }
            } message: {
                Text("Enter a patient ID for this session. The capture time is recorded automatically.")
            }
        }
    }

    /// Strip whitespace from the entered patient ID (the ID must contain no
    /// spaces) and begin the guided capture if anything is left.
    private func beginCapture() {
        let cleaned = patientIDField
            .components(separatedBy: .whitespacesAndNewlines)
            .joined()
        guard !cleaned.isEmpty else { return }
        patientIDField = cleaned
        controller.beginGuidedCapture(patientId: cleaned)
    }

    /// The overlay and camera preview both fill the whole screen (they ignore
    /// the safe area), so ARKit's `projectPoint` must use the full-screen size,
    /// not the safe-area-inset layout size — otherwise projected points (the
    /// eye line) sit too high by the top inset.
    private func fullScreenSize(_ geo: GeometryProxy) -> CGSize {
        CGSize(width: geo.size.width + geo.safeAreaInsets.leading + geo.safeAreaInsets.trailing,
               height: geo.size.height + geo.safeAreaInsets.top + geo.safeAreaInsets.bottom)
    }

    /// Return to a live preview (real camera) or re-run the demo, so the
    /// operator can frame the next subject before tapping Start again.
    private func restart() {
        if CaptureController.hasTrueDepth { controller.startPreview() }
        else { controller.startDemo() }
    }

    /// Show the patient-ID prompt; on confirm it begins the guided capture.
    private func promptStart() {
        patientIDField = controller.patientId   // prefill with the last ID used
        askingPatientID = true
    }

    // MARK: - Selfie / Operator toggle

    /// Switches between mirrored self-view and an un-mirrored operator view for
    /// photographing someone else. Both use the front TrueDepth camera.
    private var modeToggle: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.2)) { operatorMode.toggle() }
        } label: {
            HStack(spacing: 7) {
                Image(systemName: operatorMode ? "person.2.fill" : "person.crop.square")
                Text(operatorMode ? "Operator" : "Selfie")
            }
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().strokeBorder(Color.white.opacity(0.14), lineWidth: 1))
        }
    }

    // MARK: - Bottom status card

    private var statusCard: some View {
        VStack(spacing: 14) {
            // Headline instruction + live pose tag.
            VStack(spacing: 6) {
                if let pose = controller.phase.activePose {
                    Text(pose.name.replacingOccurrences(of: "_", with: " ").uppercased() + " VIEW")
                        .font((operatorMode ? Font.subheadline : Font.caption).weight(.bold))
                        .tracking(2)
                        .foregroundStyle(Theme.accentBright)
                }
                // Operator mode enlarges the live instruction so it's readable
                // at arm's length while the phone faces the subject.
                Text(controller.statusText)
                    .font(operatorMode ? Font.title.weight(.bold) : Font.title3.weight(.semibold))
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if controller.phase == .orbiting {
                coveragePanel
            } else if controller.phase.isActive {
                alignmentChips
            } else if controller.phase == .done {
                doneBadge
            }

            actionButton
        }
        .padding(18)
        .frame(maxWidth: .infinity)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .strokeBorder(Color.white.opacity(0.12), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.4), radius: 18, x: 0, y: 10)
    }

    private var alignmentChips: some View {
        HStack(spacing: 8) {
            CriterionChip(title: "Framing",
                          ok: controller.guidance.hasFace && controller.guidance.angleOK,
                          icon: "face.dashed")
            CriterionChip(title: distanceLabel, ok: controller.guidance.distanceOK,
                          icon: "ruler")
            CriterionChip(title: "Level", ok: controller.guidance.levelOK,
                          icon: "level")
            CriterionChip(title: "Neutral", ok: controller.guidance.expressionNeutral,
                          icon: "face.smiling")
        }
    }

    private var distanceLabel: String {
        let cm = controller.guidance.distanceMM / 10
        guard controller.guidance.hasFace, cm > 1 else { return "Distance" }
        return String(format: "%.0f cm", cm)
    }

    /// Live yaw×pitch coverage map for the free-orbit phase: each cell lights up
    /// as a colour frame is harvested for that angle, plus a running count.
    private var coveragePanel: some View {
        VStack(spacing: 6) {
            CoverageGrid(cells: controller.coverageCells)
            // Orientation so the grid reads: columns are camera angle around the
            // head, rows are tilt (top = looking down, bottom = looking up).
            HStack {
                Text("◀ their left")
                Spacer()
                Text("front")
                Spacer()
                Text("their right ▶")
            }
            .font(.system(size: 10, weight: .semibold))
            .foregroundStyle(.white.opacity(0.5))
            Text("\(controller.orbitCapturedCount) photos · "
                 + "\(controller.coverageCells.count)/\(CaptureController.orbitTotalCells) regions covered")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.white.opacity(0.85))
            Text("Captures automatically as you sweep. Cover the face from every angle, then tap Done — more photos = sharper texture and better 3D.")
                .font(.system(size: 11))
                .foregroundStyle(.white.opacity(0.55))
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var doneBadge: some View {
        HStack(spacing: 10) {
            Image(systemName: "checkmark.seal.fill")
                .foregroundStyle(Theme.success)
                .font(.title3)
            Text("Scan complete — open Sessions to upload")
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.9))
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Theme.success.opacity(0.14), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    @ViewBuilder private var actionButton: some View {
        switch controller.phase {
        case .idle, .preview, .done:
            VStack(spacing: 10) {
                // After a capture, the headline action is to inspect the 3D model.
                if controller.phase == .done, let dir = controller.finishedSession {
                    Button { modelToView = IdentifiedURL(url: dir) } label: {
                        Label("View 3D model", systemImage: "cube.transparent")
                    }
                    .buttonStyle(PrimaryButtonStyle())

                    Button { restart() } label: {
                        Label("Capture again",
                              systemImage: CaptureController.hasTrueDepth
                                ? "arrow.clockwise" : "wand.and.stars")
                    }
                    .buttonStyle(GhostButtonStyle())
                } else if CaptureController.hasTrueDepth {
                    Button { promptStart() } label: {
                        Label("Start capture", systemImage: "viewfinder")
                    }
                    .buttonStyle(PrimaryButtonStyle())

                    Button { controller.startDemo() } label: {
                        Label("Run demo (no camera)", systemImage: "wand.and.stars")
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(.white.opacity(0.75))
                    }
                } else {
                    Button { controller.startDemo() } label: {
                        Label("Run demo capture", systemImage: "wand.and.stars")
                    }
                    .buttonStyle(PrimaryButtonStyle())

                    Label("No TrueDepth camera detected — demo mode renders a sample scan.",
                          systemImage: "info.circle")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.6))
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: .infinity)
                }
            }
        case .orbiting:
            VStack(spacing: 10) {
                Button { controller.finishOrbit() } label: {
                    Label("Finish & save scan", systemImage: "checkmark.circle")
                }
                .buttonStyle(PrimaryButtonStyle())

                Button(role: .cancel) { controller.cancel() } label: {
                    Label("Cancel", systemImage: "xmark")
                }
                .buttonStyle(GhostButtonStyle())
            }
        default:
            Button(role: .cancel) {
                controller.cancel()
            } label: {
                Label("Cancel", systemImage: "xmark")
            }
            .buttonStyle(GhostButtonStyle())
        }
    }
}

/// Live yaw×pitch coverage map for the free orbit. Columns are yaw (−90°…+90°
/// left→right), rows are the three pitch bands (above / level / below, top→
/// bottom). A cell fills green once a colour frame has been harvested for it.
private struct CoverageGrid: View {
    let cells: Set<String>
    private let yawRange = CaptureController.orbitYawIndexRange
    private let pitchCount = CaptureController.orbitPitchCount

    var body: some View {
        VStack(spacing: 3) {
            ForEach(Array((0..<pitchCount).reversed()), id: \.self) { pi in
                HStack(spacing: 3) {
                    ForEach(Array(yawRange), id: \.self) { yi in
                        let on = cells.contains(CaptureController.orbitCellKey(yi, pi))
                        RoundedRectangle(cornerRadius: 2, style: .continuous)
                            .fill(on ? AnyShapeStyle(Theme.success)
                                     : AnyShapeStyle(Color.white.opacity(0.12)))
                            .frame(height: 12)
                            .frame(maxWidth: .infinity)
                            .animation(.easeOut(duration: 0.2), value: on)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity)
    }
}

/// Minimal free-orbit guide: a dashed orbit ring with a marker at the camera's
/// current yaw (0° = front, at the top), so the operator can see where they've
/// been and keep circling smoothly. No eye line — the face anchor is gone at
/// the wide angles this phase reaches.
private struct OrbitGuideOverlay: View {
    let yawDeg: Float

    var body: some View {
        GeometryReader { geo in
            let center = CGPoint(x: geo.size.width / 2, y: geo.size.height * 0.40)
            let radius = min(geo.size.width, geo.size.height) * 0.26
            let angle = Double(yawDeg) * .pi / 180 - .pi / 2
            ZStack {
                Circle()
                    .strokeBorder(Color.white.opacity(0.25),
                                  style: StrokeStyle(lineWidth: 2, dash: [4, 8]))
                    .frame(width: radius * 2, height: radius * 2)
                    .position(center)
                Circle()
                    .fill(Theme.accentBright)
                    .frame(width: 16, height: 16)
                    .position(x: center.x + CGFloat(cos(angle)) * radius,
                              y: center.y + CGFloat(sin(angle)) * radius)
            }
        }
        .ignoresSafeArea()
    }
}

/// One alignment criterion shown as a lit/unlit chip.
private struct CriterionChip: View {
    let title: String
    let ok: Bool
    let icon: String

    var body: some View {
        VStack(spacing: 5) {
            Image(systemName: ok ? "checkmark" : icon)
                .font(.caption.weight(.bold))
            Text(title)
                .font(.system(size: 10, weight: .semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .foregroundStyle(ok ? Theme.ink : .white.opacity(0.75))
        .frame(maxWidth: .infinity)
        .padding(.vertical, 9)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(ok ? Theme.success.opacity(0.95) : Color.white.opacity(0.08))
        )
        .animation(.easeOut(duration: 0.2), value: ok)
    }
}

/// Top progress rail across the capture poses.
private struct PoseStepper: View {
    let phase: CaptureController.Phase

    var body: some View {
        // Nine steps don't fit the screen width, so the rail scrolls and keeps
        // the active step centred. The pill (background) stays screen-width;
        // only the nodes inside scroll. Connectors are fixed-width here — a
        // flexible `maxWidth: .infinity` is unbounded inside a horizontal
        // ScrollView and would blow up the layout.
        ScrollViewReader { proxy in
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 0) {
                    ForEach(Array(CaptureController.Pose.allCases.enumerated()), id: \.element.rawValue) { idx, pose in
                        node(for: pose).id(pose.rawValue)
                        if idx < CaptureController.Pose.allCases.count - 1 {
                            Rectangle()
                                .fill(connectorFilled(after: pose) ? Theme.accent : Color.white.opacity(0.18))
                                .frame(width: 16, height: 2)
                        }
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
            }
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().strokeBorder(Color.white.opacity(0.12), lineWidth: 1))
            .onChange(of: activeRawValue) { _, new in
                withAnimation(.easeInOut(duration: 0.3)) {
                    proxy.scrollTo(new, anchor: .center)
                }
            }
            .onAppear { proxy.scrollTo(activeRawValue, anchor: .center) }
        }
    }

    /// Raw value of the step the rail should centre on. Orbit/done sit on the
    /// last keyframe (all captured by then).
    private var activeRawValue: Int {
        switch phase {
        case let .aligning(p), let .capturing(p): return p.rawValue
        case let .holding(p, _): return p.rawValue
        case .orbiting, .done: return CaptureController.Pose.allCases.count - 1
        default: return 0
        }
    }

    private func node(for pose: CaptureController.Pose) -> some View {
        let state = poseState(pose)
        return VStack(spacing: 4) {
            ZStack {
                Circle()
                    .fill(state == .pending
                          ? AnyShapeStyle(Color.white.opacity(0.12))
                          : AnyShapeStyle(Theme.brand))
                    .frame(width: 30, height: 30)
                if state == .active {
                    Circle().strokeBorder(Theme.accentBright, lineWidth: 2)
                        .frame(width: 38, height: 38)
                }
                Image(systemName: icon(for: pose, state: state))
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(state == .pending ? .white.opacity(0.7) : Theme.ink)
            }
            Text(label(for: pose))
                .font(.system(size: 10, weight: .semibold))
                .lineLimit(1)
                .foregroundStyle(state == .pending ? .white.opacity(0.5) : .white)
        }
        .frame(width: 48)
    }

    private func icon(for pose: CaptureController.Pose, state: PoseState) -> String {
        if state == .done { return "checkmark" }
        switch pose {
        case .front:     return "person.fill"
        case .leftHalf:  return "arrow.left"
        case .left:      return "arrow.left.to.line"
        case .rightHalf: return "arrow.right"
        case .right:     return "arrow.right.to.line"
        case .brow:      return "arrow.up"
        case .jaw:       return "arrow.down"
        case .earLeft:   return "ear"
        case .earRight:  return "ear"
        }
    }

    private func label(for pose: CaptureController.Pose) -> String {
        switch pose {
        case .front:     return "Front"
        case .leftHalf:  return "L ½"
        case .left:      return "Left"
        case .rightHalf: return "R ½"
        case .right:     return "Right"
        case .brow:      return "Brow"
        case .jaw:       return "Jaw"
        case .earLeft:   return "Ear L"
        case .earRight:  return "Ear R"
        }
    }

    private enum PoseState { case pending, active, done }

    private func poseState(_ pose: CaptureController.Pose) -> PoseState {
        switch phase {
        case .idle, .preview: return .pending
        // All depth keyframes are captured by the time the orbit/done phases
        // are reached, so every node reads as done.
        case .orbiting, .done: return .done
        case let .aligning(current), let .capturing(current):
            return order(pose, current)
        case let .holding(current, _):
            return order(pose, current)
        }
    }

    private func order(_ pose: CaptureController.Pose,
                       _ current: CaptureController.Pose) -> PoseState {
        if pose.rawValue < current.rawValue { return .done }
        if pose.rawValue == current.rawValue { return .active }
        return .pending
    }

    private func connectorFilled(after pose: CaptureController.Pose) -> Bool {
        poseState(pose) == .done
    }
}

/// Camera preview backed by ARSCNView (shows the TrueDepth video feed).
struct ARPreview: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView()
        view.session = session
        view.automaticallyUpdatesLighting = true
        return view
    }

    func updateUIView(_ view: ARSCNView, context: Context) {}
}

/// The VECTRA-style alignment guides: a camera-style viewfinder frame with
/// corner ticks, the target eye line + midline, the live line through the
/// subject's eyes, and a gradient hold-progress ring.
struct GuideOverlay: View {
    let guidance: CaptureController.GuidanceState
    let phase: CaptureController.Phase

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { timeline in
            Canvas { ctx, size in
                let t = timeline.date.timeIntervalSinceReferenceDate
                draw(in: &ctx, size: size, time: t)
            }
        }
        .ignoresSafeArea()
    }

    private func draw(in ctx: inout GraphicsContext, size: CGSize, time: Double) {
        let aligned = guidance.aligned
        let lineColor: Color = aligned ? Theme.success : Theme.accentBright
        let midX = size.width / 2
        let eyeY = size.height * 0.42

        // Breathing pulse while still searching for alignment.
        let pulse = aligned ? 1.0 : 0.55 + 0.45 * (0.5 + 0.5 * sin(time * 2.2))

        // 1) Viewfinder frame with corner ticks.
        let inset: CGFloat = 28
        let frame = CGRect(x: inset, y: size.height * 0.14,
                           width: size.width - inset * 2,
                           height: size.height * 0.62)
        drawCornerTicks(in: &ctx, rect: frame,
                        color: lineColor.opacity(aligned ? 0.95 : 0.6 * pulse))

        // 2) Target guide lines (the VECTRA's two alignment lines).
        var guides = Path()
        guides.move(to: CGPoint(x: frame.minX, y: eyeY))
        guides.addLine(to: CGPoint(x: frame.maxX, y: eyeY))
        guides.move(to: CGPoint(x: midX, y: frame.minY))
        guides.addLine(to: CGPoint(x: midX, y: frame.maxY))
        ctx.stroke(guides, with: .color(.white.opacity(0.30)),
                   style: StrokeStyle(lineWidth: 1, dash: [5, 7]))

        guard guidance.hasFace else {
            drawHint(in: &ctx, center: CGPoint(x: midX, y: eyeY),
                     text: "Looking for a face…", color: .white.opacity(0.8))
            return
        }

        // 3) Live line through the eyes + eye markers.
        var live = Path()
        live.move(to: guidance.eyeLeft)
        live.addLine(to: guidance.eyeRight)
        ctx.stroke(live, with: .color(lineColor),
                   style: StrokeStyle(lineWidth: 2, lineCap: .round))
        for eye in [guidance.eyeLeft, guidance.eyeRight] {
            let r: CGFloat = aligned ? 5 : 4
            if aligned {
                let glow = Path(ellipseIn: CGRect(x: eye.x - r - 3, y: eye.y - r - 3,
                                                  width: (r + 3) * 2, height: (r + 3) * 2))
                ctx.fill(glow, with: .color(Theme.success.opacity(0.25)))
            }
            let dot = Path(ellipseIn: CGRect(x: eye.x - r, y: eye.y - r,
                                             width: r * 2, height: r * 2))
            ctx.fill(dot, with: .color(lineColor))
            ctx.stroke(dot, with: .color(.white.opacity(0.9)), lineWidth: 1.2)
        }

        // 4) Hold-progress ring while locking the pose.
        if case let .holding(_, progress) = phase {
            let center = CGPoint(x: midX, y: eyeY)
            let radius: CGFloat = 42
            let track = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius,
                                               width: radius * 2, height: radius * 2))
            ctx.stroke(track, with: .color(.white.opacity(0.2)), lineWidth: 6)
            var arc = Path()
            arc.addArc(center: center, radius: radius,
                       startAngle: .degrees(-90),
                       endAngle: .degrees(-90 + 360 * progress), clockwise: false)
            ctx.stroke(arc, with: .linearGradient(
                Gradient(colors: [Theme.accentBright, Theme.success]),
                startPoint: CGPoint(x: center.x - radius, y: center.y),
                endPoint: CGPoint(x: center.x + radius, y: center.y)),
                style: StrokeStyle(lineWidth: 6, lineCap: .round))
        }

        // 5) Capture flash ring.
        if case .capturing = phase {
            let center = CGPoint(x: midX, y: eyeY)
            let radius: CGFloat = 46
            let ring = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius,
                                              width: radius * 2, height: radius * 2))
            ctx.stroke(ring, with: .color(Theme.accentBright.opacity(pulse)),
                       style: StrokeStyle(lineWidth: 5, lineCap: .round))
        }
    }

    private func drawCornerTicks(in ctx: inout GraphicsContext, rect: CGRect, color: Color) {
        let len: CGFloat = 26
        let corners: [(CGPoint, CGPoint, CGPoint)] = [
            (CGPoint(x: rect.minX, y: rect.minY + len), CGPoint(x: rect.minX, y: rect.minY), CGPoint(x: rect.minX + len, y: rect.minY)),
            (CGPoint(x: rect.maxX - len, y: rect.minY), CGPoint(x: rect.maxX, y: rect.minY), CGPoint(x: rect.maxX, y: rect.minY + len)),
            (CGPoint(x: rect.minX, y: rect.maxY - len), CGPoint(x: rect.minX, y: rect.maxY), CGPoint(x: rect.minX + len, y: rect.maxY)),
            (CGPoint(x: rect.maxX - len, y: rect.maxY), CGPoint(x: rect.maxX, y: rect.maxY), CGPoint(x: rect.maxX, y: rect.maxY - len)),
        ]
        for (a, b, c) in corners {
            var p = Path()
            p.move(to: a); p.addLine(to: b); p.addLine(to: c)
            ctx.stroke(p, with: .color(color),
                       style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
        }
    }

    private func drawHint(in ctx: inout GraphicsContext, center: CGPoint,
                          text: String, color: Color) {
        var resolved = ctx.resolve(Text(text).font(.subheadline.weight(.medium)))
        resolved.shading = .color(color)
        ctx.draw(resolved, at: center, anchor: .center)
    }
}

// MARK: - Phase / guidance view conveniences

extension CaptureController.Phase {
    var isActive: Bool {
        switch self {
        case .idle, .preview, .done: return false
        default: return true
        }
    }

    var activePose: CaptureController.Pose? {
        switch self {
        case let .aligning(p), let .capturing(p): return p
        case let .holding(p, _): return p
        default: return nil
        }
    }
}

extension CaptureController.GuidanceState {
    /// View-local mirrors of the controller's gates, used only for the
    /// per-criterion feedback chips (the controller's `aligned` stays
    /// authoritative for the capture trigger).
    var distanceOK: Bool { distanceMM > 250 && distanceMM < 500 }
    var levelOK: Bool { abs(rollDeg) < 10 && abs(pitchDeg) < 10 }
}
