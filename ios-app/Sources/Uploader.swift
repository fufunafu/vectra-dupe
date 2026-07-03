import Foundation

/// Talks to the vectra-dupe processing server.
struct Uploader {
    let baseURL: URL

    struct ServerError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// Dense hybrid captures upload ~150 JPEGs and the server's `/process` runs
    /// ICP + TSDF + texture projection over all of them synchronously, which far
    /// exceeds URLSession's 60 s default. Give every call a generous budget:
    /// 120 s of idle (no-byte) tolerance and 10 min total per resource.
    private static let session: URLSession = {
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 120
        cfg.timeoutIntervalForResource = 600
        return URLSession(configuration: cfg)
    }()

    private func post(_ path: String, json: [String: Any]) async throws -> [String: Any] {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: json)
        let (data, response) = try await Self.session.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode < 300 else {
            throw ServerError(message: String(data: data, encoding: .utf8) ?? "server error")
        }
        return try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    }

    private func get(_ path: String) async throws -> [String: Any] {
        let request = URLRequest(url: baseURL.appendingPathComponent(path))
        let (data, response) = try await Self.session.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode < 300 else {
            throw ServerError(message: String(data: data, encoding: .utf8) ?? "server error")
        }
        return try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    }

    func ensurePatient(settings: AppSettings) async throws -> String {
        if !settings.patientId.isEmpty { return settings.patientId }
        let name = settings.patientName.isEmpty ? "iPhone Patient" : settings.patientName
        let patient = try await post("api/patients", json: ["name": name])
        guard let pid = patient["id"] as? String else {
            throw ServerError(message: "no patient id in response")
        }
        await MainActor.run { settings.patientId = pid }
        return pid
    }

    /// Create a session, multipart-upload every file in the directory, trigger
    /// processing (which now runs in the background on the server), then poll the
    /// session until it reports done/failed. `onProcessing` fires once the upload
    /// is in and processing has begun, so the UI can switch its pill. Returns the
    /// server session id.
    func uploadSession(directory: URL, label: String, settings: AppSettings,
                       onUploadProgress: @escaping @Sendable (Double, Int64, Int64) -> Void = { _, _, _ in },
                       onProcessing: @Sendable () async -> Void = {}) async throws -> String {
        let pid = try await ensurePatient(settings: settings)
        let session = try await post("api/patients/\(pid)/sessions",
                                     json: ["label": label])
        guard let sid = session["id"] as? String else {
            throw ServerError(message: "no session id in response")
        }

        let boundary = "vectra-\(UUID().uuidString)"
        var body = Data()
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        for file in files {
            let payload = try Data(contentsOf: file)
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append(("Content-Disposition: form-data; name=\"files\"; "
                         + "filename=\"\(file.lastPathComponent)\"\r\n"
                         + "Content-Type: application/octet-stream\r\n\r\n")
                .data(using: .utf8)!)
            body.append(payload)
            body.append("\r\n".data(using: .utf8)!)
        }
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        var request = URLRequest(
            url: baseURL.appendingPathComponent("api/patients/\(pid)/sessions/\(sid)/upload"))
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)",
                         forHTTPHeaderField: "Content-Type")
        // Per-task delegate surfaces byte-level upload progress (bodies are tens of
        // MB for dense captures) without changing the shared session.
        let progress = UploadProgressDelegate { sent, total in
            let frac = total > 0 ? Double(sent) / Double(total) : 0
            onUploadProgress(frac, sent, total)
        }
        let (data, response) = try await Self.session.upload(
            for: request, from: body, delegate: progress)
        guard let http = response as? HTTPURLResponse, http.statusCode < 300 else {
            throw ServerError(message: String(data: data, encoding: .utf8) ?? "upload failed")
        }

        // Processing now runs server-side in the background and returns at once;
        // poll the session meta until it finishes so we never hold a request open
        // across the (possibly multi-minute) reconstruction.
        _ = try await post("api/patients/\(pid)/sessions/\(sid)/process", json: [:])
        await onProcessing()
        return try await pollUntilDone(pid: pid, sid: sid)
    }

    /// Poll the session meta every few seconds until status is done/failed.
    /// Bounded so a wedged job can't poll forever (≈12 min at 2.5 s/poll).
    private func pollUntilDone(pid: String, sid: String) async throws -> String {
        let maxAttempts = 300
        for _ in 0..<maxAttempts {
            try await Task.sleep(nanoseconds: 2_500_000_000)
            let meta = try await get("api/patients/\(pid)/sessions/\(sid)")
            switch meta["status"] as? String {
            case "done":
                return sid
            case "failed":
                throw ServerError(message: meta["error"] as? String ?? "processing failed")
            default:
                continue   // "processing" / "new" — keep waiting
            }
        }
        throw ServerError(message: "processing timed out — still running on the server")
    }
}

/// Task-scoped delegate that reports multipart upload byte progress. Passed to
/// `URLSession.upload(for:from:delegate:)` so the shared session is untouched.
private final class UploadProgressDelegate: NSObject, URLSessionTaskDelegate {
    let onProgress: @Sendable (Int64, Int64) -> Void   // (totalSent, totalExpected)

    init(onProgress: @escaping @Sendable (Int64, Int64) -> Void) {
        self.onProgress = onProgress
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didSendBodyData bytesSent: Int64,
                    totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        onProgress(totalBytesSent, totalBytesExpectedToSend)
    }
}
