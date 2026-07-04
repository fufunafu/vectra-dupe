// ocrecon — wrap Apple's Object Capture (RealityKit PhotogrammetrySession) as a
// command-line tool the Python server can shell out to.
//
//   ocrecon <input-image-folder> <output.(obj|usdz)>
//           [--detail full|medium|reduced|raw] [--feature-sensitivity normal|high]
//           [--poses <poses.json>]
//
// --poses additionally writes the per-image camera poses OC estimated, in the
// SAME coordinate space as the output model:
//   {"poses":[{"image":"color_orbit_000.jpg","camera_to_world":[[...4]x4]},…]}
// (Used to drop photo-only reconstructions into the metric ARKit frame by
// fitting these camera centres to the ARKit-tracked ones.)
//
// Contract (so Python can parse it robustly):
//   * progress / diagnostics -> stderr
//   * the LAST stdout line on success is a one-line JSON object:
//       {"output":"...","images_used":N,"seconds":T}
//   * exit 0 on success; non-zero (with a one-line stderr message) on any failure,
//     so the caller can fall back to the depth-fusion (TSDF) path.
//
// This is Apple's HelloPhotogrammetry sample re-expressed as a dependency-free
// SwiftPM executable (argv parsed by hand instead of swift-argument-parser).

import Foundation
import RealityKit
import ModelIO

func warn(_ msg: String) {
    FileHandle.standardError.write(Data(("ocrecon: " + msg + "\n").utf8))
}

func fail(_ msg: String) -> Never {
    warn(msg)
    exit(2)
}

let args = CommandLine.arguments
guard args.count >= 3 else {
    fail("usage: ocrecon <input-folder> <output.(obj|usdz)> "
         + "[--detail full|medium|reduced|raw] [--feature-sensitivity normal|high]")
}

let inputURL = URL(fileURLWithPath: args[1], isDirectory: true)
let outputURL = URL(fileURLWithPath: args[2])

var detail: PhotogrammetrySession.Request.Detail = .full
var sensitivity: PhotogrammetrySession.Configuration.FeatureSensitivity = .high
var posesURL: URL? = nil

var i = 3
while i < args.count {
    let flag = args[i]
    i += 1
    guard i < args.count else { fail("\(flag) needs a value") }
    let rawValue = args[i]
    let value = rawValue.lowercased()
    i += 1
    switch flag {
    case "--detail":
        switch value {
        case "preview": detail = .preview
        case "reduced": detail = .reduced
        case "medium": detail = .medium
        case "full": detail = .full
        case "raw": detail = .raw
        default: fail("unknown --detail \(value)")
        }
    case "--feature-sensitivity":
        switch value {
        case "normal": sensitivity = .normal
        case "high": sensitivity = .high
        default: fail("unknown --feature-sensitivity \(value)")
        }
    case "--poses":
        posesURL = URL(fileURLWithPath: rawValue)
    default:
        fail("unknown argument \(flag)")
    }
}

guard PhotogrammetrySession.isSupported else {
    fail("PhotogrammetrySession is not supported on this machine")
}

// Count usable input images for the success JSON (and to fail fast on an empty set).
let imageExts: Set<String> = ["jpg", "jpeg", "png", "heic"]
let imageCount = (try? FileManager.default.contentsOfDirectory(
    at: inputURL, includingPropertiesForKeys: nil))?
    .filter { imageExts.contains($0.pathExtension.lowercased()) }.count ?? 0
guard imageCount > 0 else { fail("no images found in \(inputURL.path)") }

var config = PhotogrammetrySession.Configuration()
// Faces are low-texture/smooth skin; high sensitivity finds more correspondences.
config.featureSensitivity = sensitivity
// Our orbit frames are spatially adjacent but Object Capture's own ordering is
// robust; .unordered avoids penalising a non-clean sweep order.
config.sampleOrdering = .unordered

let start = Date()

// PhotogrammetrySession on this macOS build only validates USD/USDZ output
// (it rejects .obj with `invalidOutput`). So we always reconstruct to a USDZ,
// then — if the caller asked for .obj — convert USDZ -> OBJ via Model I/O
// (MDLAsset), which Open3D can read with texture + UVs intact.
let wantsObj = outputURL.pathExtension.lowercased() == "obj"
let modelURL: URL = wantsObj
    ? URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("ocrecon-\(ProcessInfo.processInfo.globallyUniqueString).usdz")
    : outputURL

// Model I/O's OBJ exporter writes the .mtl referencing texture files
// (`map_Kd <material>_diffuseColor.png` …) but does NOT write the PNGs to disk.
// So we pull each texture out of the asset's materials and write it ourselves,
// using the same `<materialName>_<suffix>.png` naming the exporter referenced.
func writeMaterialTextures(_ asset: MDLAsset, objDir: URL) {
    let semantics: [(MDLMaterialSemantic, String)] = [
        (.baseColor, "_diffuseColor.png"),
        (.ambientOcclusion, "_occlusion.png"),
        (.tangentSpaceNormal, "_normal.png"),
    ]
    var materials: [MDLMaterial] = []
    func collect(_ object: MDLObject) {
        if let mesh = object as? MDLMesh, let subs = mesh.submeshes {
            for case let sub as MDLSubmesh in subs {
                if let m = sub.material { materials.append(m) }
            }
        }
        for child in object.children.objects { collect(child) }
    }
    for i in 0..<asset.count { collect(asset.object(at: i)) }

    for m in materials {
        let name = m.name.isEmpty ? "Texture" : m.name
        for (sem, suffix) in semantics {
            guard let prop = m.property(with: sem),
                  let tex = prop.textureSamplerValue?.texture else { continue }
            let url = objDir.appendingPathComponent(name + suffix)
            if tex.write(to: url) {
                warn("wrote texture \(url.lastPathComponent)")
            } else {
                warn("failed to write texture \(url.lastPathComponent)")
            }
        }
    }
}

// Convert the reconstructed USDZ at `modelURL` into the requested OBJ at
// `outputURL`, exporting texture images alongside it. Returns false on failure.
func convertUSDZtoOBJ() -> Bool {
    guard MDLAsset.canExportFileExtension("obj") else {
        warn("Model I/O cannot export obj on this system")
        return false
    }
    let asset = MDLAsset(url: modelURL)
    asset.loadTextures()  // pull embedded USDZ textures into memory
    do {
        try asset.export(to: outputURL)
    } catch {
        warn("USDZ->OBJ conversion failed: \(error)")
        return false
    }
    writeMaterialTextures(asset, objDir: outputURL.deletingLastPathComponent())
    return FileManager.default.fileExists(atPath: outputURL.path)
}

// Serialize OC's per-image camera poses (model space) to `url` as JSON.
func writePoses(_ poses: PhotogrammetrySession.Poses, to url: URL) {
    var entries: [String] = []
    for (sampleID, pose) in poses.posesBySample {
        guard let imageURL = poses.urlsBySample[sampleID] else { continue }
        let R = simd_float3x3(pose.rotation)
        let t = pose.translation
        // Row-major camera->world (same layout the Python side parses).
        let rows: [[Float]] = [
            [R.columns.0.x, R.columns.1.x, R.columns.2.x, t.x],
            [R.columns.0.y, R.columns.1.y, R.columns.2.y, t.y],
            [R.columns.0.z, R.columns.1.z, R.columns.2.z, t.z],
            [0, 0, 0, 1],
        ]
        let matrix = rows.map { "[" + $0.map { String($0) }.joined(separator: ",") + "]" }
            .joined(separator: ",")
        entries.append("{\"image\":\"\(imageURL.lastPathComponent)\","
                       + "\"camera_to_world\":[\(matrix)]}")
    }
    let json = "{\"poses\":[" + entries.joined(separator: ",") + "]}"
    do {
        try json.write(to: url, atomically: true, encoding: .utf8)
        warn("wrote \(entries.count) camera poses to \(url.lastPathComponent)")
    } catch {
        warn("failed to write poses: \(error)")
    }
}

do {
    let session = try PhotogrammetrySession(input: inputURL, configuration: config)
    var requests: [PhotogrammetrySession.Request] = [
        .modelFile(url: modelURL, detail: detail)]
    if posesURL != nil { requests.append(.poses) }
    warn("starting \(detail) reconstruction on \(imageCount) images")
    try session.process(requests: requests)

    for try await output in session.outputs {
        switch output {
        case .processingComplete:
            if wantsObj && !convertUSDZtoOBJ() {
                fail("reconstruction succeeded but USDZ->OBJ conversion failed")
            }
            let secs = Int(Date().timeIntervalSince(start))
            // Final stdout line = machine-readable result.
            print("{\"output\":\"\(outputURL.path)\",\"images_used\":\(imageCount),\"seconds\":\(secs)}")
            exit(0)
        case .requestError(_, let error):
            fail("request error: \(error)")
        case .requestProgress(_, let fraction):
            warn("progress \(Int(fraction * 100))%")
        case .requestComplete(_, let result):
            warn("request complete")
            if case .poses(let poses) = result, let posesURL {
                writePoses(poses, to: posesURL)
            }
        case .inputComplete:
            warn("input complete")
        case .invalidSample(let id, let reason):
            warn("invalid sample \(id): \(reason)")
        case .skippedSample(let id):
            warn("skipped sample \(id)")
        case .automaticDownsampling:
            warn("automatic downsampling applied")
        case .processingCancelled:
            fail("processing cancelled")
        case .requestProgressInfo(_, let info):
            warn("progress info: \(info)")
        case .stitchingIncomplete:
            warn("stitching incomplete")
        @unknown default:
            warn("unhandled output")
        }
    }
} catch {
    fail("processing failed: \(error)")
}

fail("session ended without completing")
