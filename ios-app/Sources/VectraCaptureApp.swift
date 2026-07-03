import SwiftUI

@main
struct VectraCaptureApp: App {
    @StateObject private var settings = AppSettings()

    init() { Theme.applyGlobalAppearance() }

    var body: some Scene {
        WindowGroup {
            RootTabs()
                .environmentObject(settings)
                .tint(Theme.accent)
                .preferredColorScheme(.dark)
        }
    }
}

private struct RootTabs: View {
    @State private var selection = 0

    var body: some View {
        TabView(selection: $selection) {
            CaptureView()
                .tag(0)
                .tabItem { Label("Capture", systemImage: "viewfinder") }
            SessionsListView()
                .tag(1)
                .tabItem { Label("Sessions", systemImage: "square.stack.3d.up.fill") }
            SettingsView()
                .tag(2)
                .tabItem { Label("Settings", systemImage: "slider.horizontal.3") }
        }
    }
}

final class AppSettings: ObservableObject {
    @AppStorage("serverURL") var serverURL: String = "http://192.168.1.10:8008"
    @AppStorage("patientId") var patientId: String = ""
    @AppStorage("patientName") var patientName: String = ""
}

// MARK: - Design system

/// Shared visual language: a calibrated-instrument look — deep ink surfaces,
/// a signature teal accent, glass cards, and monospaced live readouts.
enum Theme {
    static let accent       = Color(red: 0.22, green: 0.80, blue: 0.74)   // calibrated teal
    static let accentBright = Color(red: 0.40, green: 0.92, blue: 0.84)
    static let accentDeep   = Color(red: 0.07, green: 0.42, blue: 0.50)
    static let success      = Color(red: 0.32, green: 0.86, blue: 0.55)
    static let warn         = Color(red: 1.00, green: 0.80, blue: 0.33)
    static let danger       = Color(red: 0.98, green: 0.44, blue: 0.44)

    static let ink     = Color(red: 0.05, green: 0.07, blue: 0.10)   // deepest background
    static let surface = Color(red: 0.10, green: 0.13, blue: 0.17)   // card background
    static let hairline = Color.white.opacity(0.09)

    static var brand: LinearGradient {
        LinearGradient(colors: [accentBright, accentDeep],
                       startPoint: .topLeading, endPoint: .bottomTrailing)
    }

    static var appBackground: LinearGradient {
        LinearGradient(colors: [ink, Color(red: 0.07, green: 0.11, blue: 0.15)],
                       startPoint: .top, endPoint: .bottom)
    }

    /// Tint the UIKit tab/navigation chrome to match the dark theme.
    static func applyGlobalAppearance() {
        let tab = UITabBarAppearance()
        tab.configureWithTransparentBackground()
        tab.backgroundEffect = UIBlurEffect(style: .systemUltraThinMaterialDark)
        tab.backgroundColor = UIColor(ink).withAlphaComponent(0.55)
        UITabBar.appearance().standardAppearance = tab
        UITabBar.appearance().scrollEdgeAppearance = tab

        let nav = UINavigationBarAppearance()
        nav.configureWithTransparentBackground()
        nav.backgroundEffect = UIBlurEffect(style: .systemUltraThinMaterialDark)
        nav.titleTextAttributes = [.foregroundColor: UIColor.white]
        nav.largeTitleTextAttributes = [.foregroundColor: UIColor.white]
        UINavigationBar.appearance().standardAppearance = nav
        UINavigationBar.appearance().scrollEdgeAppearance = nav
    }
}

// MARK: - Reusable components

/// A frosted surface card used across the data screens.
struct GlassCard<Content: View>: View {
    var padding: CGFloat = 16
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(padding)
            .background(Theme.surface.opacity(0.85), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .strokeBorder(Theme.hairline, lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.35), radius: 14, x: 0, y: 8)
    }
}

/// A small status chip with a tinted dot.
struct StatusPill: View {
    let text: String
    let color: Color
    var filled = false

    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 7, height: 7)
            Text(text)
                .font(.caption.weight(.semibold))
                .foregroundStyle(filled ? .white : color)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(
            Capsule().fill(filled ? color.opacity(0.9) : color.opacity(0.14))
        )
    }
}

/// Filled, gradient primary action.
struct PrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .foregroundStyle(Theme.ink)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 15)
            .background(Theme.brand, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .shadow(color: Theme.accent.opacity(0.45), radius: 12, x: 0, y: 6)
            .opacity(configuration.isPressed ? 0.85 : 1)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}

/// Outlined secondary action (e.g. Cancel).
struct GhostButtonStyle: ButtonStyle {
    var tint: Color = .white
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .foregroundStyle(tint)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 15)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.18), lineWidth: 1)
            )
            .opacity(configuration.isPressed ? 0.7 : 1)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}

/// The small wordmark used in headers.
struct BrandMark: View {
    var size: CGFloat = 34
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: size * 0.28, style: .continuous)
                .fill(Theme.brand)
            Image(systemName: "cube.transparent.fill")
                .font(.system(size: size * 0.52, weight: .semibold))
                .foregroundStyle(Theme.ink)
        }
        .frame(width: size, height: size)
        .shadow(color: Theme.accent.opacity(0.4), radius: 6, x: 0, y: 3)
    }
}
