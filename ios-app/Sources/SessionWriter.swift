import Foundation
import simd

/// Writes a captured session to Documents/sessions/<timestamp>/ in the
/// shared `vectra-dupe-session/1` format the server ingests.
enum SessionWriter {
    static var sessionsRoot: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("sessions", isDirectory: true)
    }

    /// Row-major 4x4 (the server parses `world_to_camera` as rows of columns).
    private static func rows(_ m: simd_double4x4) -> [[Double]] {
        (0..<4).map { row in
            [m.columns.0[row], m.columns.1[row], m.columns.2[row], m.columns.3[row]]
        }
    }

    static func write(poses: [CapturedPose],
                      colorFrames: [ColorFrameCapture] = [],
                      patientId: String = "") throws -> URL {
        let stamp = ISO8601DateFormatter().string(from: Date())
        let dirName = stamp.replacingOccurrences(of: ":", with: "-")
        let dir = sessionsRoot.appendingPathComponent(dirName, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        var poseEntries: [[String: Any]] = []
        for pose in poses {
            let depthFile = "depth_\(pose.name).bin"
            var data = Data(capacity: pose.depthMM.count * 4)
            pose.depthMM.withUnsafeBufferPointer { buf in
                data.append(UnsafeBufferPointer(
                    start: UnsafeRawPointer(buf.baseAddress!)
                        .assumingMemoryBound(to: UInt8.self),
                    count: buf.count * 4))
            }
            try data.write(to: dir.appendingPathComponent(depthFile))

            var entry: [String: Any] = [
                "name": pose.name,
                "depth_file": depthFile,
                "width": pose.width,
                "height": pose.height,
                "intrinsics": ["fx": pose.fx, "fy": pose.fy,
                               "cx": pose.cx, "cy": pose.cy],
                "world_to_camera": Self.rows(pose.worldToCamera),
                "depth_unit_mm": 1.0,
            ]
            // The color photo + its intrinsics let the mesh be textured (on
            // device and on the computer). Absent for demo captures.
            if let jpeg = pose.colorJPEG {
                let colorFile = "color_\(pose.name).jpg"
                try jpeg.write(to: dir.appendingPathComponent(colorFile))
                entry["color_file"] = colorFile
                entry["rgb_width"] = pose.rgbWidth
                entry["rgb_height"] = pose.rgbHeight
                entry["rgb_intrinsics"] = ["fx": pose.rgbFx, "fy": pose.rgbFy,
                                           "cx": pose.rgbCx, "cy": pose.rgbCy]
            }
            poseEntries.append(entry)
        }

        // Dense, depth-LESS colour frames auto-harvested over the free orbit.
        // Written as a top-level `color_frames` array (server schema: name,
        // color_file, rgb_width/height, rgb_intrinsics, world_to_camera); the
        // server projects them for texture only, never through ICP/TSDF.
        var colorEntries: [[String: Any]] = []
        for cf in colorFrames {
            let colorFile = "color_\(cf.name).jpg"
            try cf.jpeg.write(to: dir.appendingPathComponent(colorFile))
            colorEntries.append([
                "name": cf.name,
                "color_file": colorFile,
                "rgb_width": cf.width,
                "rgb_height": cf.height,
                "rgb_intrinsics": ["fx": cf.fx, "fy": cf.fy,
                                   "cx": cf.cx, "cy": cf.cy],
                "world_to_camera": Self.rows(cf.worldToCamera),
            ])
        }

        var meta: [String: Any] = [
            "format": "vectra-dupe-session/1",
            "label": dirName,
            "device": "iphone-truedepth",
            "captured_at": stamp,
            "poses": poseEntries,
        ]
        if !colorEntries.isEmpty { meta["color_frames"] = colorEntries }
        // Operator-entered identifier for this capture session (no spaces).
        if !patientId.isEmpty { meta["patient_id"] = patientId }
        let json = try JSONSerialization.data(
            withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
        try json.write(to: dir.appendingPathComponent("session.json"))
        return dir
    }

    static func listLocalSessions() -> [URL] {
        (try? FileManager.default.contentsOfDirectory(
            at: sessionsRoot, includingPropertiesForKeys: nil))?
            .filter { $0.hasDirectoryPath }
            .sorted { $0.lastPathComponent > $1.lastPathComponent } ?? []
    }
}
