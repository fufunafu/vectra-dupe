"""Server-side processing: session -> mesh, and mesh pair -> comparison.

All meshes are stored in a normalized frame: face centroid at the origin,
y up, face looking along +z (derived from the front-pose camera). This keeps
viewer orientation and heatmap projections consistent across capture sources.
"""

import glob
import json
import os
import shutil
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vectra3d import analyze, compare, fuse, io_session, photogrammetry  # noqa: E402


# Keep only geometry within this radius (mm) of the face-anchor centre (≈ head
# centre), dropping shoulders, clothing, and background. Used for the MEASUREMENT
# mesh (mesh.ply); generous so nothing measurable is clipped.
HEAD_RADIUS_MM = 135.0

# Tighter crop for the Object Capture DISPLAY meshes (mesh.glb / mesh_textured.glb).
# OC reconstructs the full hair/neck periphery, which it renders as a gray,
# low-texture blob halo with a ragged outline. A face-focused sphere trims that
# halo and gives a clean boundary. Display-only — never applied to mesh.ply.
# Overridable via VECTRA_DISPLAY_CROP_MM while tuning visually.
DISPLAY_CROP_RADIUS_MM = float(os.environ.get("VECTRA_DISPLAY_CROP_MM", "110.0"))

# Extra Taubin iterations applied to the VIEWER meshes only (mesh.glb /
# mesh_textured.glb), on top of the fusion-mesh smoothing. The measurement mesh
# (mesh.ply) is left at the fusion level; this is purely cosmetic — it irons out
# the residual ripple so a straight nose looks straight, without affecting volume.
DISPLAY_SMOOTH_ITERS = int(os.environ.get("VECTRA_DISPLAY_SMOOTH", "12"))

# Light Taubin for the OC display mesh — OC geometry is already sharp, so just a
# few iterations to take the edge off the periphery without melting detail.
OC_DISPLAY_TAUBIN_ITERS = max(0, int(os.environ.get("VECTRA_OC_DISPLAY_TAUBIN", "3")))


def crop_to_head(mesh: o3d.geometry.TriangleMesh,
                 center: np.ndarray, radius_mm: float) -> o3d.geometry.TriangleMesh:
    verts = np.asarray(mesh.vertices)
    if len(verts) == 0:
        return mesh
    keep = np.linalg.norm(verts - center, axis=1) <= radius_mm
    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_vertices_by_mask(~keep)
    return out


def keep_main_components(mesh: o3d.geometry.TriangleMesh,
                         min_frac: float = 0.05,
                         max_offset_mm: float = 70.0) -> o3d.geometry.TriangleMesh:
    """Drop disconnected islands that aren't the face: TrueDepth speckle, and the
    larger background/hair slabs a profile view sometimes fuses into a separate
    blob off to one side. The face reliably fuses into ONE big component, so we
    keep the largest, plus any other component that is both sizeable (>= min_frac
    of it) AND centred near it (within max_offset_mm of its centroid). A big blob
    sitting off to the side — too far to be a facial feature — is dropped even
    though it clears the size threshold. Also helps UV-atlas unwrapping, which
    fails on meshes riddled with tiny components."""
    out = o3d.geometry.TriangleMesh(mesh)
    if len(out.triangles) == 0:
        return out
    labels, counts, _ = out.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if len(counts) == 0:
        return out
    verts = np.asarray(out.vertices)
    tris = np.asarray(out.triangles)
    biggest = int(counts.argmax())
    face_center = verts[np.unique(tris[labels == biggest])].mean(axis=0)
    size_ok = counts >= max(1, int(counts.max() * min_frac))
    keep = np.zeros(len(counts), dtype=bool)
    keep[biggest] = True
    for ci in np.nonzero(size_ok)[0]:
        if ci == biggest:
            continue
        center = verts[np.unique(tris[labels == ci])].mean(axis=0)
        if np.linalg.norm(center - face_center) <= max_offset_mm:
            keep[ci] = True
    drop_tri = ~keep[labels]
    out.remove_triangles_by_mask(drop_tri)
    out.remove_unreferenced_vertices()
    return out


def normalize_to_front_frame(mesh: o3d.geometry.TriangleMesh,
                             front_world_to_cam: np.ndarray,
                             radius_mm: float = HEAD_RADIUS_MM
                             ) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """Returns the normalized+cropped mesh and the 4x4 world->normalized
    transform applied (so photo extrinsics can be moved into the same frame).

    The fused mesh is already in the ARKit face-anchor frame, which is the
    normalized frame we want: +y points to the crown, +z points out of the face
    toward the camera, and the origin sits ~at the head centre. (Verified on a
    real capture: head spans y≈[-120,+150], face front at +z, shoulders at -y.)
    So there is NO rotation to apply — earlier flip/roll guesses are exactly what
    rotated the head 90°/top-down on real scans. We only crop to the head sphere
    (around the anchor origin) and recentre on the centroid for viewing.

    `front_world_to_cam` is unused now but kept in the signature for callers.
    """
    del front_world_to_cam  # intentionally unused; frame is already normalized
    out = o3d.geometry.TriangleMesh(mesh)
    # The ARKit face-anchor origin is world (0,0,0) ≈ the head centre; crop a
    # head-sized sphere around it to drop shoulders, clothing, and background.
    out = crop_to_head(out, np.zeros(3), radius_mm)
    out = keep_main_components(out)
    center = out.get_center()
    out.translate(-center)
    out.compute_vertex_normals()
    recenter = np.eye(4)
    recenter[:3, 3] = -center
    return out, recenter


def process_session(raw_dir: str, out_dir: str,
                    texture_mode: str = "both") -> dict:
    """Process a session into mesh.ply (the measurement mesh) and the viewer's
    display meshes.

    HYBRID geometry (deliberate split of measurement vs display):
      * mesh.ply (measurement) is ALWAYS the depth-fusion (TSDF) mesh. That is the
        geometry on which ±0.2 mL volume accuracy was validated. Apple Object
        Capture's photogrammetry mesh, even after metric landmark alignment,
        deviates ~7 mm from the TrueDepth surface and aligns non-deterministically,
        so it is NOT trusted for measurement.
      * The display meshes (mesh.glb / mesh_textured.glb) PREFER Object Capture —
        a clean, photoreal, textured surface — falling back to the TSDF mesh when
        OC is unavailable or fails a metric guard. The user sees a Kiri-tier model;
        the clinic measures on the validated depth mesh.

    Display variants (built in one pass so the viewer toggles instantly):
      mesh.glb          — per-vertex colour, no UV texture.
      mesh_textured.glb — sharp photo texture (OC's single UV atlas, or the TSDF
                          cylindrical projection in fallback).
    `texture_mode` is kept for callers but "both" is the default and the only path
    the viewer uses. VECTRA_DISABLE_OC=1 forces the TSDF display path too (used by
    the e2e test, whose synthetic renders aren't a valid OC input).
    """
    poses, color_frames, meta = io_session.load_session(raw_dir)
    os.makedirs(out_dir, exist_ok=True)
    # stale OC scratch dirs survive a killed process; clear them so they don't pile up
    for stale in glob.glob(os.path.join(out_dir, "oc_*")):
        if os.path.isdir(stale):
            shutil.rmtree(stale, ignore_errors=True)

    # --- Measurement mesh: ALWAYS depth-fusion (TSDF) — the validated ±0.2 mL path.
    extrinsics = fuse.view_extrinsics(poses)
    # Colour frames are depth-less: they never enter ICP/TSDF (so the dense RGB
    # set can be large without slowing geometry). They carry raw ARKit poses;
    # texture projection is forgiving and gates by facing angle.
    col_ext_world = [cf.world_to_camera for cf in color_frames]
    world_mesh = fuse.integrate(poses, extrinsics, color_frames, col_ext_world)
    mesh, world_to_norm = normalize_to_front_frame(world_mesh, poses[0].world_to_camera)
    # mesh.ply: geometry + per-vertex colour — drives the volume measurement.
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "mesh.ply"), mesh)

    # --- Display geometry: prefer photoreal Object Capture; fall back to TSDF.
    display_source = "tsdf"
    oc_stats: dict = {}
    oc = None
    # record the preconditions so a TSDF fallback is never silent about why
    oc_avail = {
        "tool": photogrammetry.tool_available(),
        "landmarks": photogrammetry.landmark_tooling_available(),
        "disabled": os.environ.get("VECTRA_DISABLE_OC") == "1",
        "has_color_frames": bool(color_frames),
    }
    use_oc = (color_frames and poses
              and oc_avail["tool"] and oc_avail["landmarks"]
              and not oc_avail["disabled"])
    if use_oc:
        try:
            oc = photogrammetry.reconstruct_metric(raw_dir, poses, color_frames, out_dir)
            oc_stats = oc.stats
            display_source = "object_capture"
        except Exception as e:  # noqa: BLE001
            print(f"[process_session] Object Capture display skipped, using TSDF: {e}")
            oc = None
            oc_stats = {"oc_error": str(e)}

    vertex_ok = textured_ok = False
    if oc is not None:
        # OC geometry is already clean — no cosmetic smoothing needed. Normalize it
        # into the same canonical face frame the measurement mesh uses (its own
        # recentre, since OC and TSDF centroids differ by a few mm — irrelevant for
        # a standalone display model).
        # Tighter face-focused crop (+ keep_main_components inside
        # normalize_to_front_frame) trims OC's gray hair/neck blob halo and ragged
        # outline. The SAME crop radius + recenter (oc_w2n) is reused for the
        # textured mesh below so the Smooth/Textured toggle stays aligned.
        oc_disp, oc_w2n = normalize_to_front_frame(
            oc.mesh, poses[0].world_to_camera, radius_mm=DISPLAY_CROP_RADIUS_MM)
        if OC_DISPLAY_TAUBIN_ITERS:
            # Taubin drops vertex colours; vertex count/order is preserved, so
            # re-attach the pre-smoothing colours.
            colors = oc_disp.vertex_colors
            oc_disp = oc_disp.filter_smooth_taubin(
                number_of_iterations=OC_DISPLAY_TAUBIN_ITERS)
            oc_disp.vertex_colors = colors
            oc_disp.compute_vertex_normals()
        if texture_mode in ("vertex", "both"):
            try:
                tmesh = o3d.t.geometry.TriangleMesh.from_legacy(oc_disp)
                o3d.t.io.write_triangle_mesh(os.path.join(out_dir, "mesh.glb"), tmesh)
                vertex_ok = oc_disp.has_vertex_colors()
            except Exception as e:  # noqa: BLE001
                print(f"[process_session] OC vertex glb skipped: {e}")
        if texture_mode in ("cylindrical", "both"):
            # OC carries a sharp single-atlas UV texture; clean (crop + floater
            # removal + light smoothing) and move into the normalized frame
            # (UVs + albedo untouched).
            textured_ok = photogrammetry.write_normalized_textured_glb(
                oc.textured, oc_w2n, os.path.join(out_dir, "mesh_textured.glb"),
                crop_center=np.zeros(3), crop_radius_mm=DISPLAY_CROP_RADIUS_MM,
                smooth_iters=OC_DISPLAY_TAUBIN_ITERS)
    else:
        # TSDF display: an extra cosmetic Taubin pass (shrink-free, keeps features)
        # so the viewer surface reads as smooth. Vertex count/order preserved, so
        # the per-vertex colours stay valid. Measurement (mesh.ply) is untouched.
        display = o3d.geometry.TriangleMesh(mesh)
        if DISPLAY_SMOOTH_ITERS:
            display = display.filter_smooth_taubin(number_of_iterations=DISPLAY_SMOOTH_ITERS)
            display.vertex_colors = mesh.vertex_colors
            display.compute_vertex_normals()
        if texture_mode in ("vertex", "both"):
            try:
                tmesh = o3d.t.geometry.TriangleMesh.from_legacy(display)
                o3d.t.io.write_triangle_mesh(os.path.join(out_dir, "mesh.glb"), tmesh)
                vertex_ok = display.has_vertex_colors()
            except Exception as e:  # noqa: BLE001
                print(f"[process_session] vertex glb skipped: {e}")
        if texture_mode in ("cylindrical", "both"):
            try:
                inv = np.linalg.inv(world_to_norm)
                ext_norm = [e @ inv for e in extrinsics]
                col_ext_norm = [e @ inv for e in col_ext_world]
                tmesh = fuse.build_cylindrical_textured_mesh(
                    display, poses, ext_norm,
                    color_frames=color_frames, color_extrinsics=col_ext_norm)
                o3d.t.io.write_triangle_mesh(
                    os.path.join(out_dir, "mesh_textured.glb"), tmesh)
                textured_ok = True
            except Exception as e:  # noqa: BLE001
                print(f"[process_session] textured glb skipped: {e}")

    stats = {
        "reconstruction": "tsdf",          # measurement geometry (always)
        "display_source": display_source,  # geometry shown in the viewer
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "surface_area_mm2": round(float(mesh.get_surface_area()), 1),
        "depth_keyframes": len(poses),
        "color_frames": len(color_frames),
        "textured": vertex_ok,
        "has_textured_glb": textured_ok,
        "label": meta.get("label", ""),
        "captured_at": meta.get("captured_at", ""),
        "device": meta.get("device", ""),
        "patient_id": meta.get("patient_id", ""),
    }
    # OC scale/rms/ipd display diagnostics (its "reconstruction" key would clobber
    # the measurement source, so drop it — the display geometry is display_source).
    stats.update({k: v for k, v in oc_stats.items() if k != "reconstruction"})
    stats["oc_available"] = oc_avail
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def compare_sessions_on_disk(before_mesh_path: str, after_mesh_path: str,
                             out_dir: str) -> dict:
    before = o3d.io.read_triangle_mesh(before_mesh_path)
    after = o3d.io.read_triangle_mesh(after_mesh_path)
    before.compute_vertex_normals()
    after.compute_vertex_normals()

    result = compare.compare_sessions(before, after)
    os.makedirs(out_dir, exist_ok=True)

    analyze.save_colored_mesh(before, result.field,
                              os.path.join(out_dir, "heatmap.ply"))
    significant = [r for r in result.regions if abs(r.volume_mm3) >= compare.NOISE_FLOOR_MM3]
    total_ml = sum(r.volume_mm3 for r in significant) / 1000.0
    analyze.save_heatmap_png(
        result.field, os.path.join(out_dir, "heatmap.png"),
        f"net change in detected regions: {total_ml:+.2f} mL "
        f"({len(significant)} significant region(s))")

    valid = np.isfinite(result.field.distances)
    summary = {
        "regions": [r.to_dict() for r in result.regions],
        "net_significant_volume_ml": round(total_ml, 3),
        "noise_floor_ml": compare.NOISE_FLOOR_MM3 / 1000.0,
        "surface_rms_mm": round(float(
            np.sqrt(np.mean(result.field.distances[valid] ** 2))), 3) if valid.any() else None,
        "transform": np.asarray(result.transform).tolist(),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
