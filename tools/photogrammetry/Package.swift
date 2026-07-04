// swift-tools-version: 5.9
import PackageDescription

// Standalone Object Capture CLI. Builds with `swift build -c release` — no Xcode
// project, no xcodegen. RealityKit is a system framework, so there are no
// external package dependencies (the build is hermetic/offline).
let package = Package(
    name: "ocrecon",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "ocrecon", path: "Sources/ocrecon"),
    ]
)
