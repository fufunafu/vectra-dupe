import SwiftUI
import UIKit

struct SessionsListView: View {
    @EnvironmentObject private var settings: AppSettings
    @State private var sessions: [URL] = []
    @State private var uploadState: [String: UploadStatus] = [:]
    @State private var exporting: IdentifiedURL?
    @State private var exportError: String?

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.appBackground.ignoresSafeArea()

                if sessions.isEmpty {
                    emptyState
                } else {
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(sessions, id: \.absoluteString) { dir in
                                SessionCard(
                                    dir: dir,
                                    status: uploadState[dir.lastPathComponent] ?? .onDevice,
                                    onUpload: { upload(dir: dir) },
                                    onExport: { export(dir: dir) },
                                    onDelete: { delete(dir: dir) })
                            }
                        }
                        .padding(.horizontal, 16)
                        .padding(.top, 8)
                        .padding(.bottom, 24)
                    }
                }
            }
            .navigationTitle("Sessions")
            .navigationDestination(for: IdentifiedURL.self) { item in
                Model3DView(sessionDir: item.url)
            }
            .sheet(item: $exporting) { item in
                ActivityView(items: [item.url])
            }
            .alert("Export failed", isPresented: Binding(
                get: { exportError != nil }, set: { if !$0 { exportError = nil } })) {
                Button("OK", role: .cancel) { exportError = nil }
            } message: { Text(exportError ?? "") }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { refresh() } label: { Image(systemName: "arrow.clockwise") }
                }
            }
            .onAppear(perform: refresh)
            .refreshable { refresh() }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            ZStack {
                Circle().fill(Theme.accent.opacity(0.12)).frame(width: 96, height: 96)
                Image(systemName: "square.stack.3d.up.slash")
                    .font(.system(size: 38, weight: .light))
                    .foregroundStyle(Theme.accent)
            }
            Text("No captures yet")
                .font(.title3.weight(.semibold))
                .foregroundStyle(.white)
            Text("Head to the Capture tab and record a 3-view\nscan. It will appear here, ready to upload.")
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundStyle(.white.opacity(0.6))
        }
        .padding(40)
    }

    private func refresh() {
        sessions = SessionWriter.listLocalSessions()
    }

    private func delete(dir: URL) {
        try? FileManager.default.removeItem(at: dir)
        refresh()
    }

    private func export(dir: URL) {
        do {
            let zip = try SessionExport.zip(dir)
            exporting = IdentifiedURL(url: zip)
        } catch {
            exportError = error.localizedDescription
        }
    }

    private func upload(dir: URL) {
        let name = dir.lastPathComponent
        guard let base = URL(string: settings.serverURL) else {
            uploadState[name] = .failed("Invalid server URL — check Settings")
            return
        }
        uploadState[name] = .uploading(fraction: 0, sent: 0, total: 0)
        Task {
            do {
                let sid = try await Uploader(baseURL: base).uploadSession(
                    directory: dir, label: name, settings: settings,
                    onUploadProgress: { frac, sent, total in
                        Task { @MainActor in
                            uploadState[name] = .uploading(fraction: frac, sent: sent, total: total)
                        }
                    },
                    onProcessing: { await MainActor.run {
                        uploadState[name] = .processing(startedAt: Date())
                    } })
                await MainActor.run { uploadState[name] = .processed(sid) }
            } catch {
                await MainActor.run {
                    uploadState[name] = .failed(error.localizedDescription)
                }
            }
        }
    }
}

enum UploadStatus {
    case onDevice
    case uploading(fraction: Double, sent: Int64, total: Int64)
    case processing(startedAt: Date)
    case processed(String)
    case failed(String)

    var pill: (text: String, color: Color) {
        switch self {
        case .onDevice:        return ("On device", .white.opacity(0.55))
        case .uploading:       return ("Uploading…", Theme.warn)
        case .processing:      return ("Processing…", Theme.warn)
        case .processed:       return ("Processed", Theme.success)
        case .failed:          return ("Failed", Theme.danger)
        }
    }

    var detail: String? {
        switch self {
        case .processing:         return "Reconstructing on the server — this can take a few minutes. Keep the app open."
        case let .processed(sid): return "Server session \(sid)"
        case let .failed(msg):    return msg
        default:                  return nil
        }
    }

    /// Upload button is disabled while either uploading or processing.
    var isBusy: Bool {
        switch self {
        case .uploading, .processing: return true
        default:                      return false
        }
    }
    var isProcessed: Bool { if case .processed = self { return true }; return false }
}

private struct SessionCard: View {
    let dir: URL
    let status: UploadStatus
    let onUpload: () -> Void
    let onExport: () -> Void
    let onDelete: () -> Void

    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 12) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(Theme.accent.opacity(0.15))
                            .frame(width: 44, height: 44)
                        Image(systemName: "cube.transparent")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundStyle(Theme.accent)
                    }
                    VStack(alignment: .leading, spacing: 3) {
                        Text(Self.prettyDate(dir.lastPathComponent))
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.white)
                        Text("3-view TrueDepth scan")
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.5))
                    }
                    Spacer()
                    StatusPill(text: status.pill.text, color: status.pill.color,
                               filled: status.isProcessed)
                }

                if let detail = status.detail {
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.55))
                        .lineLimit(2)
                }

                UploadProgressBlock(status: status)

                NavigationLink(value: IdentifiedURL(url: dir)) {
                    Label("View 3D model", systemImage: "cube.transparent")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 11)
                        .background(Theme.brand,
                                    in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                        .foregroundStyle(Theme.ink)
                }

                HStack(spacing: 10) {
                    Button(action: onUpload) {
                        Label(status.isProcessed ? "Re-upload" : "Upload to server",
                              systemImage: "arrow.up.to.line")
                            .font(.subheadline.weight(.semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(Color.white.opacity(status.isBusy ? 0.04 : 0.08),
                                        in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                            .foregroundStyle(.white.opacity(0.85))
                    }
                    .disabled(status.isBusy)

                    Button(action: onExport) {
                        Image(systemName: "square.and.arrow.up")
                            .font(.subheadline.weight(.semibold))
                            .frame(width: 44)
                            .padding(.vertical, 10)
                            .background(Color.white.opacity(0.08),
                                        in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                            .foregroundStyle(.white.opacity(0.85))
                    }

                    Button(role: .destructive, action: onDelete) {
                        Image(systemName: "trash")
                            .font(.subheadline.weight(.semibold))
                            .frame(width: 44)
                            .padding(.vertical, 10)
                            .background(Color.white.opacity(0.06),
                                        in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                            .foregroundStyle(Theme.danger)
                    }
                }
            }
        }
    }

    /// Session dir names are ISO8601 timestamps with ':' → '-'. Render as a
    /// friendly local date/time, falling back to the raw name.
    static func prettyDate(_ raw: String) -> String {
        // Restore the colons the writer stripped from the time part for
        // filesystem safety (e.g. 2026-06-12T14-30-05Z -> ...T14:30:05Z).
        var s = raw
        if let tIdx = s.firstIndex(of: "T") {
            let timePart = s[s.index(after: tIdx)...]
                .replacingOccurrences(of: "-", with: ":")
            s = String(s[..<s.index(after: tIdx)]) + timePart
        }
        guard let date = ISO8601DateFormatter().date(from: s) else { return raw }
        let out = DateFormatter()
        out.dateFormat = "MMM d, yyyy · h:mm a"
        return out.string(from: date)
    }
}

/// Progress bar for the two long phases of an upload:
///  • uploading — a real byte-% bar (dense captures are tens of MB).
///  • processing — a time-based bar that eases toward ~95% over the typical
///    ~2.5 min reconstruction (the server poll reports no percentage), with a
///    live elapsed counter.
private struct UploadProgressBlock: View {
    let status: UploadStatus

    // Typical OC reconstruction wall-clock; the bar approaches 95% asymptotically.
    private static let expectedProcessing: TimeInterval = 150

    var body: some View {
        switch status {
        case let .uploading(fraction, sent, total):
            VStack(alignment: .leading, spacing: 5) {
                ProgressView(value: fraction.isFinite ? min(max(fraction, 0), 1) : 0)
                    .tint(Theme.accent)
                Text(total > 0
                     ? "Uploading \(Self.mb(sent)) / \(Self.mb(total))"
                     : "Uploading…")
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.5))
            }
        case let .processing(startedAt):
            TimelineView(.periodic(from: .now, by: 1)) { context in
                let elapsed = max(0, context.date.timeIntervalSince(startedAt))
                let frac = 0.95 * (1 - exp(-elapsed / Self.expectedProcessing))
                VStack(alignment: .leading, spacing: 5) {
                    ProgressView(value: min(frac, 0.99)).tint(Theme.accent)
                    HStack {
                        Text("\(Self.clock(elapsed)) elapsed · ~2–3 min")
                        Spacer()
                        Text("\(Int(frac * 100))%")
                    }
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.5))
                }
            }
        default:
            EmptyView()
        }
    }

    private static func mb(_ bytes: Int64) -> String {
        String(format: "%.0f MB", Double(bytes) / 1_000_000)
    }

    private static func clock(_ t: TimeInterval) -> String {
        let s = Int(t)
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}

struct SettingsView: View {
    @EnvironmentObject private var settings: AppSettings
    @AppStorage("soundGuidance") private var soundGuidance = true

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.appBackground.ignoresSafeArea()
                ScrollView {
                    VStack(spacing: 16) {
                        header
                        serverCard
                        guidanceCard
                        patientCard
                        helpCard
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                }
            }
            .navigationTitle("Settings")
        }
    }

    private var header: some View {
        GlassCard {
            HStack(spacing: 14) {
                BrandMark(size: 46)
                VStack(alignment: .leading, spacing: 3) {
                    Text("Vectra-dupe")
                        .font(.title3.weight(.bold))
                        .foregroundStyle(.white)
                    Text("3D facial volume capture")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.55))
                }
                Spacer()
                Text("v1.0")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.white.opacity(0.4))
            }
        }
    }

    private var serverCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                SectionLabel(icon: "server.rack", title: "Processing server")
                HStack(spacing: 10) {
                    Image(systemName: "link").foregroundStyle(Theme.accent)
                    TextField("http://host:8008", text: settings.$serverURL)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .foregroundStyle(.white)
                }
                .padding(12)
                .background(Color.white.opacity(0.06),
                            in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                Text("The LAN address of the machine running the FastAPI server.")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.5))
            }
        }
    }

    private var guidanceCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                SectionLabel(icon: "speaker.wave.2.fill", title: "Capture guidance")
                Toggle(isOn: $soundGuidance) {
                    Text("Sound & haptic guidance")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.white)
                }
                .tint(Theme.accent)
                Text("Tones and vibration guide you when you can't see the screen — "
                     + "pitch tells you which way to move, faster beeps mean you're "
                     + "closer, and a chime + buzz confirm each shot. Ideal for selfie capture.")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.5))
            }
        }
    }

    private var patientCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                SectionLabel(icon: "person.text.rectangle", title: "Patient")
                HStack(spacing: 10) {
                    Image(systemName: "person").foregroundStyle(Theme.accent)
                    TextField("Patient name", text: settings.$patientName)
                        .foregroundStyle(.white)
                }
                .padding(12)
                .background(Color.white.opacity(0.06),
                            in: RoundedRectangle(cornerRadius: 12, style: .continuous))

                HStack {
                    Text("Server patient ID")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.55))
                    Spacer()
                    Text(settings.patientId.isEmpty ? "created on first upload"
                         : settings.patientId)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(settings.patientId.isEmpty
                                         ? .white.opacity(0.4) : Theme.accent)
                }

                if !settings.patientId.isEmpty {
                    Button(role: .destructive) {
                        settings.patientId = ""
                    } label: {
                        Label("Reset patient link", systemImage: "arrow.counterclockwise")
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(Theme.danger)
                    }
                    .padding(.top, 2)
                }
            }
        }
    }

    private var helpCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                SectionLabel(icon: "lightbulb", title: "Capture tips")
                ForEach(tips, id: \.self) { tip in
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.caption)
                            .foregroundStyle(Theme.accent)
                            .padding(.top, 2)
                        Text(tip)
                            .font(.footnote)
                            .foregroundStyle(.white.opacity(0.7))
                    }
                }
            }
        }
    }

    private let tips = [
        "Hold the phone at arm's length, eyes level with the guide line.",
        "Keep a relaxed, neutral expression — no smiling or brow raise.",
        "Each pose captures automatically once the line turns green.",
        "Use a hair net or headband; loose hair ruins the reconstruction.",
    ]
}

private struct SectionLabel: View {
    let icon: String
    let title: String
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.caption.weight(.semibold))
                .foregroundStyle(Theme.accent)
            Text(title.uppercased())
                .font(.caption.weight(.bold))
                .tracking(1.2)
                .foregroundStyle(.white.opacity(0.6))
        }
    }
}

// MARK: - Export

/// Bundles a session directory (depth maps + session.json) into a .zip the
/// user can AirDrop to a Mac or save to Files, then run through the full
/// `vectra3d` pipeline at native fidelity.
enum SessionExport {
    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    static func zip(_ dir: URL) throws -> URL {
        let coordinator = NSFileCoordinator()
        var coordError: NSError?
        var produced: URL?
        var copyError: Error?

        // `.forUploading` hands back a temporary .zip of the folder that the
        // system deletes when the block returns, so copy it somewhere stable.
        coordinator.coordinate(readingItemAt: dir, options: [.forUploading],
                               error: &coordError) { zippedURL in
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent(dir.lastPathComponent + ".zip")
            do {
                try? FileManager.default.removeItem(at: dest)
                try FileManager.default.copyItem(at: zippedURL, to: dest)
                produced = dest
            } catch {
                copyError = error
            }
        }

        if let coordError { throw coordError }
        if let copyError { throw copyError }
        guard let produced else { throw Failure(message: "Could not create the archive.") }
        return produced
    }
}

/// Thin SwiftUI wrapper around the system share sheet.
struct ActivityView: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
