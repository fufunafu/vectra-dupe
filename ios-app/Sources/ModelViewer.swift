import SwiftUI
import SceneKit
import UIKit
import simd

/// A URL wrapped for use with `.sheet(item:)` and value-based navigation.
struct IdentifiedURL: Identifiable, Hashable {
    let id = UUID()
    let url: URL
}

/// Loads a saved capture session from disk and turns each pose's depth map into
/// a shaded triangle-mesh surface (in the face frame), rendered interactively
/// with SceneKit. Viewing defaults to the Front pose — a clean, recognisable
/// face — with a toggle to overlay the side views. The on-device render is a
/// preview only; the precise fused volume analysis still runs in the Python
/// pipeline (the on-device overlay has no ICP alignment, so the three views may
/// not line up perfectly).
struct Model3DView: View {
    let sessionDir: URL

    @State private var scene: SCNScene?
    @State private var poseNames: [String] = []
    @State private var selected = "front"
    @State private var errorText: String?

    var body: some View {
        ZStack {
            Theme.appBackground.ignoresSafeArea()

            if let scene {
                ScanSceneView(scene: scene, selected: selected, poseNames: poseNames)
                    .ignoresSafeArea(edges: .bottom)

                VStack {
                    if poseNames.count > 1 {
                        // A handful of poses fit a segmented control; more (the
                        // 5-view capture) overflow it, so fall back to a menu.
                        Group {
                            if poseNames.count > 3 {
                                posePicker.pickerStyle(.menu)
                            } else {
                                posePicker.pickerStyle(.segmented)
                            }
                        }
                        .padding(.horizontal, 16)
                        .padding(.top, 8)
                    }
                    Spacer()
                    Text("drag to rotate · pinch to zoom")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.6))
                        .padding(.vertical, 10)
                        .padding(.horizontal, 16)
                        .background(.ultraThinMaterial, in: Capsule())
                        .padding(.bottom, 16)
                }
            } else if let errorText {
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 34, weight: .light))
                        .foregroundStyle(Theme.warn)
                    Text("Couldn't build the model")
                        .font(.headline).foregroundStyle(.white)
                    Text(errorText)
                        .font(.caption).foregroundStyle(.white.opacity(0.6))
                        .multilineTextAlignment(.center)
                }
                .padding(40)
            } else {
                VStack(spacing: 14) {
                    ProgressView().tint(Theme.accent)
                    Text("Building 3D model…")
                        .font(.subheadline).foregroundStyle(.white.opacity(0.7))
                }
            }
        }
        .navigationTitle("3D Model")
        .navigationBarTitleDisplayMode(.inline)
        .task(id: sessionDir) { await build() }
    }

    private var posePicker: some View {
        Picker("View", selection: $selected) {
            Text("All").tag("All")
            ForEach(poseNames, id: \.self) {
                Text(Self.poseLabel($0)).tag($0)
            }
        }
    }

    /// Human-readable pose label, e.g. "left_half" -> "Left ½", "front" -> "Front".
    static func poseLabel(_ name: String) -> String {
        name
            .replacingOccurrences(of: "_half", with: " ½")
            .split(separator: "_")
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }

    private func build() async {
        let dir = sessionDir
        do {
            let model = try await Task.detached(priority: .userInitiated) {
                try ScanModel.load(from: dir)
            }.value
            await MainActor.run {
                scene = model.scene
                poseNames = model.poseNames
                selected = model.poseNames.contains("front") ? "front" : "All"
            }
        } catch {
            await MainActor.run { errorText = error.localizedDescription }
        }
    }
}

/// SceneKit host with orbit/zoom controls; shows/hides pose meshes by name.
private struct ScanSceneView: UIViewRepresentable {
    let scene: SCNScene
    let selected: String
    let poseNames: [String]

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = scene
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true
        view.backgroundColor = .clear
        view.antialiasingMode = .multisampling4X
        applyVisibility(in: view)
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        if view.scene !== scene { view.scene = scene }
        applyVisibility(in: view)
    }

    private func applyVisibility(in view: SCNView) {
        let names = Set(poseNames)
        for node in view.scene?.rootNode.childNodes ?? [] {
            guard let name = node.name, names.contains(name) else { continue }
            node.isHidden = (selected != "All" && selected != name)
        }
    }
}

// MARK: - Model construction

private enum ScanError: LocalizedError {
    case unreadable(String)
    var errorDescription: String? {
        switch self { case let .unreadable(m): return m }
    }
}

private struct ScanModel {
    let scene: SCNScene
    let poseNames: [String]

    /// ~40k depth samples per pose keeps the mesh detailed but light.
    private static let targetSamplesPerPose = 40_000
    /// Don't connect neighbouring samples across a depth jump bigger than this
    /// (mm) — avoids stretched triangles between the face and the background.
    private static let discontinuityMM: Float = 12
    /// Keep only geometry within this radius (mm) of the ARKit face-anchor
    /// origin (≈ head centre), dropping shoulders, clothing, and background.
    private static let headRadiusMM: Float = 135

    private struct RawPose {
        let name: String
        var verts: [SIMD3<Float>]   // face frame, mm
        var tris: [UInt32]
        var colors: [SIMD4<Float>]? // per-vertex RGBA sampled from the photo
    }

    static func load(from dir: URL) throws -> ScanModel {
        let metaData = try Data(contentsOf: dir.appendingPathComponent("session.json"))
        guard let json = try JSONSerialization.jsonObject(with: metaData) as? [String: Any],
              let poses = json["poses"] as? [[String: Any]], !poses.isEmpty else {
            throw ScanError.unreadable("session.json has no poses")
        }

        var raw: [RawPose] = []
        for pose in poses {
            guard let name = pose["name"] as? String,
                  let depthFile = pose["depth_file"] as? String,
                  let w = (pose["width"] as? NSNumber)?.intValue,
                  let h = (pose["height"] as? NSNumber)?.intValue,
                  let intr = pose["intrinsics"] as? [String: Any],
                  let fx = (intr["fx"] as? NSNumber)?.floatValue,
                  let fy = (intr["fy"] as? NSNumber)?.floatValue,
                  let cx = (intr["cx"] as? NSNumber)?.floatValue,
                  let cy = (intr["cy"] as? NSNumber)?.floatValue,
                  let rows = pose["world_to_camera"] as? [[Any]] else { continue }

            let worldToCamera = matrix(from: rows)
            let depthData = try Data(contentsOf: dir.appendingPathComponent(depthFile))
            let depth: [Float] = depthData.withUnsafeBytes {
                Array($0.bindMemory(to: Float32.self))
            }
            guard depth.count >= w * h else { continue }

            let (verts, tris) = buildPose(
                depth: depth, w: w, h: h, fx: fx, fy: fy, cx: cx, cy: cy,
                camToWorld: worldToCamera.inverse)
            guard !verts.isEmpty else { continue }

            // Texture: project each vertex into the pose's photo for its colour.
            var colors: [SIMD4<Float>]? = nil
            if let colorFile = pose["color_file"] as? String,
               let rgb = pose["rgb_intrinsics"] as? [String: Any],
               let rfx = (rgb["fx"] as? NSNumber)?.floatValue,
               let rfy = (rgb["fy"] as? NSNumber)?.floatValue,
               let rcx = (rgb["cx"] as? NSNumber)?.floatValue,
               let rcy = (rgb["cy"] as? NSNumber)?.floatValue,
               let img = loadRGB(dir.appendingPathComponent(colorFile)) {
                colors = sampleColors(verts: verts, worldToCamera: worldToCamera,
                                      img: img, fx: rfx, fy: rfy, cx: rcx, cy: rcy)
            }
            raw.append(RawPose(name: name, verts: verts, tris: tris, colors: colors))
        }

        guard !raw.isEmpty else { throw ScanError.unreadable("no valid depth points") }

        // Common centroid + radius so all poses share one centred frame.
        var centroid = SIMD3<Float>(repeating: 0)
        var total = 0
        for p in raw { for v in p.verts { centroid += v }; total += p.verts.count }
        centroid /= Float(max(1, total))
        // Robust radius: 92nd-percentile distance, so a few stray points can't
        // blow up the bounding sphere and shrink the face to a dot.
        var dists: [Float] = []
        dists.reserveCapacity(total)
        for p in raw { for v in p.verts { dists.append(simd_length(v - centroid)) } }
        dists.sort()
        let radius = dists.isEmpty ? 100
            : dists[min(dists.count - 1, Int(Float(dists.count) * 0.92))]

        let scene = SCNScene()
        for p in raw {
            let node = SCNNode(geometry: geometry(for: p, centroid: centroid))
            node.name = p.name
            scene.rootNode.addChildNode(node)
        }

        // Camera in front of the face (face +z points toward the viewer).
        let camera = SCNCamera()
        camera.zNear = 0.001
        camera.zFar = 100
        let camNode = SCNNode()
        camNode.camera = camera
        camNode.position = SCNVector3(0, 0, max(0.05, radius / 1000) * 2.1)
        camNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(camNode)

        return ScanModel(scene: scene, poseNames: raw.map { $0.name })
    }

    /// Edge-preserving depth smoothing (a small bilateral filter on range only).
    /// TrueDepth has ~0.5 mm per-pixel noise that makes the surface read as
    /// "bumps everywhere"; averaging each pixel with same-depth neighbours kills
    /// that grain while the depth guard keeps real edges (nose, jaw) crisp.
    private static func denoiseDepth(_ src: [Float], w: Int, h: Int) -> [Float] {
        guard w > 2, h > 2 else { return src }
        let maxJumpMM: Float = 6   // don't blend across a step bigger than this
        let radius = 2
        var depth = src
        for _ in 0..<2 {
            var out = depth
            for y in 0..<h {
                for x in 0..<w {
                    let c = depth[y * w + x]
                    if c <= 0 { continue }
                    var sum: Float = 0, n: Float = 0
                    for dy in -radius...radius {
                        let yy = y + dy
                        if yy < 0 || yy >= h { continue }
                        for dx in -radius...radius {
                            let xx = x + dx
                            if xx < 0 || xx >= w { continue }
                            let v = depth[yy * w + xx]
                            if v > 0 && abs(v - c) < maxJumpMM { sum += v; n += 1 }
                        }
                    }
                    if n > 0 { out[y * w + x] = sum / n }
                }
            }
            depth = out
        }
        return depth
    }

    /// Back-project a depth map to a triangulated surface in the face frame (mm).
    private static func buildPose(depth rawDepth: [Float], w: Int, h: Int,
                                  fx: Float, fy: Float, cx: Float, cy: Float,
                                  camToWorld: simd_double4x4) -> ([SIMD3<Float>], [UInt32]) {
        let depth = denoiseDepth(rawDepth, w: w, h: h)
        let stride = max(1, Int((Double(w * h) /
            Double(targetSamplesPerPose)).squareRoot().rounded()))

        var us: [Int] = []; var u = 0; while u < w { us.append(u); u += stride }
        var vs: [Int] = []; var v = 0; while v < h { vs.append(v); v += stride }
        let cols = us.count, rowsN = vs.count

        var vidx = [Int32](repeating: -1, count: rowsN * cols)
        var zgrid = [Float](repeating: 0, count: rowsN * cols)
        var verts: [SIMD3<Float>] = []

        for (r, vv) in vs.enumerated() {
            for (c, uu) in us.enumerated() {
                let z = depth[vv * w + uu]
                if z > 0 {
                    let xc = (Float(uu) - cx) / fx * z
                    let yc = (Float(vv) - cy) / fy * z
                    let pw = camToWorld * SIMD4<Double>(Double(xc), Double(yc), Double(z), 1)
                    let p = SIMD3<Float>(Float(pw.x), Float(pw.y), Float(pw.z))
                    // Crop to the head: the world origin is the face-anchor
                    // centre, so anything far from it is shoulders/background.
                    if simd_length(p) > headRadiusMM { continue }
                    vidx[r * cols + c] = Int32(verts.count)
                    zgrid[r * cols + c] = z
                    verts.append(p)
                }
            }
        }

        var tris: [UInt32] = []
        for r in 0..<max(0, rowsN - 1) {
            for c in 0..<max(0, cols - 1) {
                let i00 = vidx[r * cols + c]
                let i01 = vidx[r * cols + c + 1]
                let i10 = vidx[(r + 1) * cols + c]
                let i11 = vidx[(r + 1) * cols + c + 1]
                if i00 < 0 || i01 < 0 || i10 < 0 || i11 < 0 { continue }
                let z00 = zgrid[r * cols + c], z01 = zgrid[r * cols + c + 1]
                let z10 = zgrid[(r + 1) * cols + c], z11 = zgrid[(r + 1) * cols + c + 1]
                let span = max(max(z00, z01), max(z10, z11))
                         - min(min(z00, z01), min(z10, z11))
                if span > discontinuityMM { continue }
                tris.append(contentsOf: [UInt32(i00), UInt32(i10), UInt32(i11)])
                tris.append(contentsOf: [UInt32(i00), UInt32(i11), UInt32(i01)])
            }
        }
        return postProcess(verts: verts, tris: tris)
    }

    /// Drop disconnected speckle (sensor-noise islands) and Laplacian-smooth the
    /// surface so it reads as a clean face instead of a grainy blob.
    private static func postProcess(verts: [SIMD3<Float>],
                                    tris: [UInt32]) -> ([SIMD3<Float>], [UInt32]) {
        guard !verts.isEmpty, tris.count >= 3 else { return (verts, tris) }

        // Vertex adjacency from triangle edges.
        var adj = [[Int]](repeating: [], count: verts.count)
        var seen = [Set<Int>](repeating: [], count: verts.count)
        func link(_ a: Int, _ b: Int) {
            if !seen[a].contains(b) { seen[a].insert(b); adj[a].append(b) }
        }
        var t = 0
        while t + 2 < tris.count {
            let a = Int(tris[t]), b = Int(tris[t + 1]), c = Int(tris[t + 2])
            link(a, b); link(b, a); link(a, c); link(c, a); link(b, c); link(c, b)
            t += 3
        }
        seen = []

        // Connected components; keep only sizeable ones (removes float speckle).
        var comp = [Int](repeating: -1, count: verts.count)
        var sizes: [Int] = []
        for s in 0..<verts.count where comp[s] == -1 && !adj[s].isEmpty {
            var stack = [s]; comp[s] = sizes.count; var count = 0
            while let n = stack.popLast() {
                count += 1
                for m in adj[n] where comp[m] == -1 { comp[m] = sizes.count; stack.append(m) }
            }
            sizes.append(count)
        }
        let minSize = max(80, verts.count / 100)

        // Laplacian smoothing (a few gentle passes).
        var pos = verts
        for _ in 0..<3 {
            var next = pos
            for i in 0..<pos.count where !adj[i].isEmpty {
                var sum = SIMD3<Float>(repeating: 0)
                for nbr in adj[i] { sum += pos[nbr] }
                let avg = sum / Float(adj[i].count)
                next[i] = pos[i] + 0.5 * (avg - pos[i])
            }
            pos = next
        }

        // Remap surviving vertices and rebuild the triangle list.
        var remap = [Int32](repeating: -1, count: verts.count)
        var outVerts: [SIMD3<Float>] = []
        for i in 0..<verts.count {
            let cidx = comp[i]
            if cidx >= 0 && sizes[cidx] >= minSize {
                remap[i] = Int32(outVerts.count)
                outVerts.append(pos[i])
            }
        }
        var outTris: [UInt32] = []
        t = 0
        while t + 2 < tris.count {
            let a = remap[Int(tris[t])], b = remap[Int(tris[t + 1])], c = remap[Int(tris[t + 2])]
            if a >= 0 && b >= 0 && c >= 0 {
                outTris.append(UInt32(a)); outTris.append(UInt32(b)); outTris.append(UInt32(c))
            }
            t += 3
        }
        return (outVerts, outTris)
    }

    /// Build a lit SCNGeometry for one pose: centre to metres, compute smooth
    /// vertex normals, tint per pose.
    private static func geometry(for pose: RawPose,
                                 centroid: SIMD3<Float>) -> SCNGeometry {
        let n = pose.verts.count
        var positions = [SCNVector3](); positions.reserveCapacity(n)
        var centred = [SIMD3<Float>](); centred.reserveCapacity(n)
        for v in pose.verts {
            let c = (v - centroid) / 1000           // mm → m
            centred.append(c)
            positions.append(SCNVector3(c.x, c.y, c.z))
        }

        var nAccum = [SIMD3<Float>](repeating: .zero, count: n)
        var t = 0
        while t + 2 < pose.tris.count {
            let a = Int(pose.tris[t]), b = Int(pose.tris[t + 1]), c = Int(pose.tris[t + 2])
            let face = simd_cross(centred[b] - centred[a], centred[c] - centred[a])
            nAccum[a] += face; nAccum[b] += face; nAccum[c] += face
            t += 3
        }
        let normals: [SCNVector3] = nAccum.map {
            let len = simd_length($0)
            let u = len > 1e-8 ? $0 / len : SIMD3<Float>(0, 0, 1)
            return SCNVector3(u.x, u.y, u.z)
        }

        let vSource = SCNGeometrySource(vertices: positions)
        let nSource = SCNGeometrySource(normals: normals)
        var sources = [vSource, nSource]

        // Photo texture (per-vertex colour) when available, else flat tint.
        let textured = (pose.colors?.count == n)
        if textured, let colors = pose.colors {
            let colorData = colors.withUnsafeBufferPointer { Data(buffer: $0) }
            sources.append(SCNGeometrySource(
                data: colorData, semantic: .color, vectorCount: n,
                usesFloatComponents: true, componentsPerVector: 4,
                bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
                dataStride: MemoryLayout<SIMD4<Float>>.stride))
        }

        let indexData = pose.tris.withUnsafeBufferPointer { Data(buffer: $0) }
        let element = SCNGeometryElement(
            data: indexData, primitiveType: .triangles,
            primitiveCount: pose.tris.count / 3, bytesPerIndex: 4)

        let geo = SCNGeometry(sources: sources, elements: [element])
        let material = SCNMaterial()
        if textured {
            // The photo already carries real lighting; show it faithfully
            // (white base so per-vertex colour displays at full brightness).
            material.diffuse.contents = UIColor.white
            material.lightingModel = .constant
        } else {
            material.diffuse.contents = tint(for: pose.name)
            material.lightingModel = .blinn
        }
        material.isDoubleSided = true
        geo.firstMaterial = material
        return geo
    }

    /// Decode a JPEG into a top-left-origin RGBA8 buffer for texture sampling.
    private static func loadRGB(_ url: URL) -> (px: [UInt8], w: Int, h: Int)? {
        guard let data = try? Data(contentsOf: url),
              let cg = UIImage(data: data)?.cgImage else { return nil }
        let w = cg.width, h = cg.height
        guard w > 0, h > 0 else { return nil }
        var px = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(
            data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        // Flip so buffer row 0 is the top of the image (matches intrinsics).
        ctx.translateBy(x: 0, y: CGFloat(h))
        ctx.scaleBy(x: 1, y: -1)
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return (px, w, h)
    }

    /// Project each face-frame vertex into the pose's photo and read its colour.
    private static func sampleColors(verts: [SIMD3<Float>],
                                     worldToCamera: simd_double4x4,
                                     img: (px: [UInt8], w: Int, h: Int),
                                     fx: Float, fy: Float, cx: Float, cy: Float)
        -> [SIMD4<Float>] {
        let fallback = SIMD4<Float>(0.80, 0.73, 0.67, 1)   // skin, for misses
        var out = [SIMD4<Float>](repeating: fallback, count: verts.count)
        // Intrinsics are in the photo's own pixel space, which equals the
        // decoded buffer size, so no rescaling is needed — just bounds checks.
        let sx = Float(img.w), sy = Float(img.h)
        for i in 0..<verts.count {
            let v = verts[i]
            let p = worldToCamera * SIMD4<Double>(Double(v.x), Double(v.y), Double(v.z), 1)
            let z = Float(p.z)
            guard z > 1 else { continue }
            let u = fx * Float(p.x) / z + cx
            let w = fy * Float(p.y) / z + cy
            guard u >= 0, w >= 0, u < sx, w < sy else { continue }
            let xi = min(img.w - 1, Int(u)), yi = min(img.h - 1, Int(w))
            let o = (yi * img.w + xi) * 4
            out[i] = SIMD4<Float>(Float(img.px[o]) / 255, Float(img.px[o + 1]) / 255,
                                  Float(img.px[o + 2]) / 255, 1)
        }
        return out
    }

    private static func tint(for name: String) -> UIColor {
        switch name {
        case "left":  return UIColor(red: 0.55, green: 0.80, blue: 0.84, alpha: 1)
        case "right": return UIColor(red: 0.60, green: 0.83, blue: 0.62, alpha: 1)
        default:      return UIColor(red: 0.87, green: 0.80, blue: 0.74, alpha: 1)  // skin
        }
    }

    /// Build a 4×4 from the saved row-major `[[row][col]]` list.
    private static func matrix(from rows: [[Any]]) -> simd_double4x4 {
        func d(_ a: Any) -> Double { (a as? NSNumber)?.doubleValue ?? 0 }
        var cols = [SIMD4<Double>](repeating: SIMD4<Double>(repeating: 0), count: 4)
        for r in 0..<min(4, rows.count) {
            for c in 0..<min(4, rows[r].count) { cols[c][r] = d(rows[r][c]) }
        }
        return simd_double4x4(cols[0], cols[1], cols[2], cols[3])
    }
}
