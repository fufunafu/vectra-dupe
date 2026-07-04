"""Fuse the depth views of a session into a single surface mesh.

Steps: back-project each depth map -> refine the (drifted) side-view poses with
colored ICP, chaining from front outward so wide profiles register against the
adjacent 3/4 view -> TSDF-integrate all views. Each ICP correction is bounded to
a drift-sized shift; a divergent correction is discarded in favour of the raw
ARKit pose (an unbounded "refine" slides whole views off and shreds the mesh).
"""

import os

import numpy as np
import open3d as o3d

from .io_session import ColorFrame, PoseCapture


def _color_sources(poses, extrinsics, color_frames=(), color_extrinsics=()):
    """Iterate (source, extrinsic) over every camera that carries colour: the
    depth poses (whose colour is optional) AND the dense colour-only frames.
    Both PoseCapture and ColorFrame expose `.color` and `.rgb_intrinsics`, so
    the texture code treats them uniformly. Depth poses without colour are
    skipped by the callers' `pose.color is None` guard."""
    yield from zip(poses, extrinsics)
    yield from zip(color_frames, color_extrinsics)

# Fusion resolution. The defaults are tuned for TrueDepth (640x480, ~1-2 mm
# noise); rear-LiDAR sessions (256x192, noisier) get a wider truncation via
# processing.py's per-device override. Env-tunable for experiments.
VOXEL_MM = float(os.environ.get("VECTRA_FUSE_VOXEL_MM", "1.0"))
SDF_TRUNC_MM = float(os.environ.get("VECTRA_FUSE_SDF_TRUNC_MM", "6.0"))
# Depth past this (mm) is dropped before fusion. Must exceed the largest
# camera-to-far-face distance — a profile is captured from ~60 cm, so the back
# of the head can sit near 70 cm; truncating at 60 cm would clip it. Background
# beyond the head is removed later by the head-sphere crop, so a loose cap here
# is safe. (Front-pose synthetic data sits well under this, so e2e is unaffected.)
DEPTH_TRUNC_MM = 800.0
# A genuine ARKit pose-drift correction is small (sub-degree, ~mm). On a smooth,
# low-texture face, colored ICP can instead slide a whole view tangentially over
# its iterations and "converge" to a pose 10-20 cm off — which shreds the TSDF
# (misaligned views cancel, leaving a lacy, holed surface). So we ACCEPT a
# correction only if it is drift-sized; anything larger means ICP diverged and
# the raw ARKit pose (locked face frame + world tracking) was the better answer.
MAX_ICP_TRANS_MM = 15.0
MAX_ICP_ROT_DEG = 8.0
# Sharpness of the "best view wins" weighting when baking per-vertex colour
# from the original photos: higher => the most head-on camera dominates (sharp,
# but harder seams); lower => softer cross-view blend.
VIEW_SHARPNESS = 4.0


def _aligned_color(pose: PoseCapture) -> np.ndarray:
    """RGB image resampled onto this pose's depth grid (h, w, 3) uint8.

    The depth and color cameras are coincident (ARKit registers depth to the
    RGB image), so a depth pixel maps to a color pixel by a depth-independent
    affine remap of the intrinsics — no per-pixel depth needed. Pixels that
    fall outside the photo get neutral gray so TSDF colour stays defined.
    """
    h, w = pose.depth.shape
    if pose.color is None or pose.rgb_intrinsics is None:
        return np.full((h, w, 3), 180, dtype=np.uint8)

    H, W = pose.color.shape[:2]
    d, r = pose.intrinsics, pose.rgb_intrinsics
    us = np.arange(w)
    vs = np.arange(h)
    ru = np.round((us - d.cx) * (r.fx / d.fx) + r.cx).astype(int)   # (w,)
    rv = np.round((vs - d.cy) * (r.fy / d.fy) + r.cy).astype(int)   # (h,)
    ok_u, ok_v = (ru >= 0) & (ru < W), (rv >= 0) & (rv < H)
    aligned = pose.color[np.clip(rv, 0, H - 1)][:, np.clip(ru, 0, W - 1)]
    aligned = np.ascontiguousarray(aligned, dtype=np.uint8)
    aligned[~(ok_v[:, None] & ok_u[None, :])] = 180
    return aligned


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinearly sample an (H, W, 3) image at float pixel coords -> (N, 3)."""
    H, W = img.shape[:2]
    u0 = np.clip(np.floor(u).astype(int), 0, W - 1)
    v0 = np.clip(np.floor(v).astype(int), 0, H - 1)
    u1 = np.clip(u0 + 1, 0, W - 1)
    v1 = np.clip(v0 + 1, 0, H - 1)
    du = (u - u0)[:, None]
    dv = (v - v0)[:, None]
    c00 = img[v0, u0].astype(np.float64)
    c01 = img[v0, u1].astype(np.float64)
    c10 = img[v1, u0].astype(np.float64)
    c11 = img[v1, u1].astype(np.float64)
    top = c00 * (1 - du) + c01 * du
    bot = c10 * (1 - du) + c11 * du
    return top * (1 - dv) + bot * dv


def bake_vertex_colors(mesh: o3d.geometry.TriangleMesh,
                       poses: list[PoseCapture],
                       extrinsics: list[np.ndarray],
                       color_frames: list[ColorFrame] = (),
                       color_extrinsics: list[np.ndarray] = (),
                       ) -> o3d.geometry.TriangleMesh:
    """Replace the coarse voxel-averaged colour with full-resolution colour
    sampled straight from the original photos. Each vertex is coloured from the
    camera(s) that see it most head-on, so texture detail is limited by the
    photo resolution rather than the TSDF voxel size. The dense colour-only
    frames contribute alongside the depth poses' photos."""
    verts = np.asarray(mesh.vertices)
    if len(verts) == 0:
        return mesh
    normals = np.asarray(mesh.vertex_normals)
    n = len(verts)
    homog = np.c_[verts, np.ones(n)]
    acc = np.zeros((n, 3))
    wsum = np.zeros(n)

    for pose, ext in _color_sources(poses, extrinsics, color_frames, color_extrinsics):
        if pose.color is None or pose.rgb_intrinsics is None:
            continue
        cam = homog @ ext.T                      # world -> camera (OpenCV)
        z = cam[:, 2]
        r = pose.rgb_intrinsics
        with np.errstate(divide="ignore", invalid="ignore"):
            u = r.fx * cam[:, 0] / z + r.cx
            v = r.fy * cam[:, 1] / z + r.cy
        H, W = pose.color.shape[:2]
        in_view = (z > 1.0) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        # Weight by how head-on the surface faces this camera (camera looks +z,
        # so an outward normal facing it has a negative z in the camera frame).
        ncam = normals @ ext[:3, :3].T
        facing = np.clip(-ncam[:, 2], 0.0, None) ** VIEW_SHARPNESS
        w = facing * in_view
        col = _bilinear(pose.color, np.clip(u, 0, W - 1), np.clip(v, 0, H - 1))
        acc += col * w[:, None]
        wsum += w

    baked = np.asarray(mesh.vertex_colors).copy()  # voxel colour as fallback
    seen = wsum > 1e-6
    baked[seen] = np.clip(acc[seen] / wsum[seen, None] / 255.0, 0.0, 1.0)
    mesh.vertex_colors = o3d.utility.Vector3dVector(baked)
    return mesh


def depth_to_cloud(pose: PoseCapture) -> o3d.geometry.PointCloud:
    h, w = pose.depth.shape
    img = o3d.geometry.Image(np.ascontiguousarray(pose.depth))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        img, pose.intrinsics.to_o3d(w, h), pose.world_to_camera,
        depth_scale=1.0, depth_trunc=DEPTH_TRUNC_MM)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=6.0, max_nn=30))
    return pcd


def colored_cloud(pose: PoseCapture,
                  world_to_camera: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    """Point cloud in world coords carrying per-point colour, for colored ICP.

    `world_to_camera` defaults to the pose's stored extrinsic; pass the
    ICP-corrected extrinsic to get the *aligned* cloud (used to grow the
    reference cloud while chaining views).
    """
    h, w = pose.depth.shape
    color = _aligned_color(pose)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(pose.depth)),
        depth_scale=1.0, depth_trunc=DEPTH_TRUNC_MM, convert_rgb_to_intensity=False)
    ext = pose.world_to_camera if world_to_camera is None else world_to_camera
    return o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, pose.intrinsics.to_o3d(w, h), ext)


def _camera_yaw_deg(world_to_camera: np.ndarray) -> float:
    """Yaw of the camera around the face frame, degrees. 0 = front, ± = sides.

    Used only to order the views from front outward so each registers against
    already-aligned geometry.
    """
    r, t = world_to_camera[:3, :3], world_to_camera[:3, 3]
    cam_center = -r.T @ t            # camera position in world (face) frame
    return float(np.degrees(np.arctan2(cam_center[0], cam_center[2])))


def _landmarks_world(pose: PoseCapture) -> tuple[np.ndarray, np.ndarray] | None:
    """MediaPipe's 478 face landmarks unprojected through this view's own depth
    into world coords (raw pose). Returns (points (478,3), valid) or None when
    the tooling is missing or no face is detectable (e.g. profile views)."""
    if pose.color is None or pose.rgb_intrinsics is None:
        return None
    from . import photogrammetry as pg      # lazy: pg imports fuse at top level
    if not pg.landmark_tooling_available():
        return None
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    try:
        o3d.io.write_image(tmp.name,
                           o3d.geometry.Image(np.ascontiguousarray(pose.color)))
        lmk = pg._detect_landmarks(tmp.name)
    finally:
        os.unlink(tmp.name)
    if lmk is None or len(lmk.get("landmarks", [])) != 478:
        return None
    return pg._unproject_landmarks(pose, np.asarray(lmk["landmarks"], float))


def _rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares rigid transform (R, t — no scale) mapping src -> dst.
    Returns (T 4x4, rms of the fit)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mu_d - R @ mu_s
    res = np.linalg.norm((R @ src.T).T + T[:3, 3] - dst, axis=1)
    return T, float(res.mean())


# A landmark-anchored correction is only trusted when the rigid fit is tight —
# a sloppy fit means the landmarks themselves are unreliable on this view.
MAX_LANDMARK_ANCHOR_RMS_MM = 6.0
MIN_LANDMARK_CORRESPONDENCES = 100


def refine_view_pose(pose: PoseCapture,
                     reference: o3d.geometry.PointCloud,
                     init: np.ndarray | None = None,
                     anchor_ext: np.ndarray | None = None) -> np.ndarray:
    """Correct one view's pose drift by ICP onto a reference cloud.

    Faces are smooth, so geometry-only ICP slides tangentially and locks in a
    slightly wrong pose (the false "dent in the nose"). Colored ICP also matches
    the photo texture (eyebrow, nostril, lip edges), pinning the alignment.
    Runs coarse-to-fine; falls back to point-to-plane if colored ICP fails.

    `reference` is the accumulated cloud of already-aligned views, not just the
    front view — a near-profile shares too little surface with the front view
    alone for ICP to converge, so it is registered against the 3/4 view that
    bridges them (see view_extrinsics).

    `init` seeds ICP (e.g. a landmark-anchored correction) instead of identity;
    `anchor_ext` is the pose to fall back to when ICP diverges — a landmark-
    anchored extrinsic beats the raw ARKit pose, which leaves the drift (and its
    misregistration ridges) fully in place.
    """
    src = colored_cloud(pose)
    correction = np.eye(4) if init is None else init.copy()
    try:
        for scale, iters in ((4.0, 60), (2.0, 35), (1.0, 20)):
            s = src.voxel_down_sample(scale)
            t = reference.voxel_down_sample(scale)
            for c in (s, t):
                c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                    radius=scale * 2.0, max_nn=30))
            correction = o3d.pipelines.registration.registration_colored_icp(
                s, t, scale * 1.4, correction,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=iters)).transformation
    except RuntimeError:
        correction = np.eye(4) if init is None else init.copy()
        for max_corr in (8.0, 3.0, 1.5):
            correction = o3d.pipelines.registration.registration_icp(
                src, reference, max_corr, correction,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            ).transformation
    # correction maps assumed-world points to reference-world, so the
    # corrected world-to-camera matrix composes with its inverse.
    corrected = pose.world_to_camera @ np.linalg.inv(correction)
    # Reject a divergent correction: if ICP slid the view farther than a real
    # drift could be, trust the raw ARKit pose instead. (Applying a 100+ mm
    # "refinement" is what shreds the fused mesh.) Measure the *physical* shift
    # of the view — the camera-centre move and the rotation between the raw and
    # corrected poses — not the raw correction matrix, whose translation term
    # mixes with rotation and understates how far the view actually travels.
    def _center(ext):
        return -ext[:3, :3].T @ ext[:3, 3]
    trans_mm = float(np.linalg.norm(_center(corrected) - _center(pose.world_to_camera)))
    dr = pose.world_to_camera[:3, :3].T @ corrected[:3, :3]
    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(dr) - 1.0) / 2.0, -1.0, 1.0))))
    if trans_mm > MAX_ICP_TRANS_MM or rot_deg > MAX_ICP_ROT_DEG:
        return anchor_ext if anchor_ext is not None else pose.world_to_camera
    return corrected


def view_extrinsics(poses: list[PoseCapture]) -> list[np.ndarray]:
    """ICP-refined world->camera matrices, one per pose (front is the anchor).

    Views are aligned from the front outward (by absolute yaw): each is
    registered against the cloud of every view aligned before it, then merged
    in. This chaining lets a ~72 deg profile lock onto the adjacent 3/4 view
    (with which it overlaps) rather than the front view (with which it barely
    does), so the profile — and the nose silhouette it carries — fuses cleanly.
    """
    def normals(cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=6.0, max_nn=30))
        return cloud

    extrinsics: list[np.ndarray | None] = [None] * len(poses)
    extrinsics[0] = poses[0].world_to_camera
    reference = normals(colored_cloud(poses[0]))

    # Landmark anchor: the front view's landmarks (unprojected through its own
    # depth) are the reference positions of 478 physical points. Any other view
    # that can see the face measures the same points; the rigid fit between the
    # two sets IS that view's pose drift — an absolute correction ICP can start
    # from and fall back to, instead of reverting to the drifted ARKit pose.
    ref_lm = _landmarks_world(poses[0])

    order = sorted(range(1, len(poses)),
                   key=lambda i: abs(_camera_yaw_deg(poses[i].world_to_camera)))
    for i in order:
        init = anchor = None
        if ref_lm is not None:
            lm = _landmarks_world(poses[i])
            if lm is not None:
                shared = lm[1] & ref_lm[1]
                if int(shared.sum()) >= MIN_LANDMARK_CORRESPONDENCES:
                    T_lm, rms = _rigid_fit(lm[0][shared], ref_lm[0][shared])
                    if rms <= MAX_LANDMARK_ANCHOR_RMS_MM:
                        init = T_lm
                        anchor = poses[i].world_to_camera @ np.linalg.inv(T_lm)
                        print(f"[fuse] {poses[i].name}: landmark anchor "
                              f"({int(shared.sum())} pts, rms {rms:.1f}mm)",
                              flush=True)
        ext = refine_view_pose(poses[i], reference, init=init, anchor_ext=anchor)
        extrinsics[i] = ext
        # Fold the freshly-aligned view into the reference so the next (wider)
        # view has overlapping geometry to register against. Down-sample to keep
        # the growing reference cheap to ICP against.
        reference = reference + normals(colored_cloud(poses[i], ext))
        reference = normals(reference.voxel_down_sample(VOXEL_MM))
    return [e for e in extrinsics]  # all filled


def integrate(poses: list[PoseCapture],
              extrinsics: list[np.ndarray],
              color_frames: list[ColorFrame] = (),
              color_extrinsics: list[np.ndarray] = (),
              sdf_trunc_mm: float | None = None,
              ) -> o3d.geometry.TriangleMesh:
    """TSDF-fuse the views (with given extrinsics) into a per-vertex-coloured
    world-frame mesh. This is the geometry used for the volume measurement.

    Only the depth `poses` drive geometry (TSDF). The colour-only frames are
    depth-less, so they never enter the TSDF or ICP — they only enrich the
    per-vertex colour via bake_vertex_colors. `color_extrinsics` are their
    world->camera matrices in the same (world) frame as `extrinsics`.
    `sdf_trunc_mm` overrides the TrueDepth-tuned truncation (rear-LiDAR depth
    is noisier and needs a wider band or misaligned noise shreds the surface)."""
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_MM,
        sdf_trunc=SDF_TRUNC_MM if sdf_trunc_mm is None else sdf_trunc_mm,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for pose, ext in zip(poses, extrinsics):
        h, w = pose.depth.shape
        color = _aligned_color(pose)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(np.ascontiguousarray(pose.depth)),
            depth_scale=1.0, depth_trunc=DEPTH_TRUNC_MM, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, pose.intrinsics.to_o3d(w, h), ext)

    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    # Taubin-smooth away residual TrueDepth noise + the structured inter-view
    # misalignment ridges that make a straight nose look "bumpy". Unlike Laplacian
    # it is shrink-free, so the overall shape — and the differential volume
    # measurement — stays intact. Measured on a real scan: at 12 iters the nose
    # tip (and other features) move <0.3 mm while high-freq ripple drops markedly,
    # so this is safely below the volume noise floor. The viewer meshes get an
    # additional cosmetic pass (see processing.DISPLAY_SMOOTH_ITERS).
    # Env-tunable so the phase0 harness can gate reductions (with landmark-
    # anchored registration the ridges are smaller, so less smoothing may keep
    # more genuine detail at the same noise floor).
    mesh = mesh.filter_smooth_taubin(
        number_of_iterations=int(os.environ.get("VECTRA_FUSE_TAUBIN", "12")))
    mesh.compute_vertex_normals()
    # Re-colour from the original full-resolution photos for a sharp texture.
    mesh = bake_vertex_colors(mesh, poses, extrinsics, color_frames, color_extrinsics)
    return mesh


def fuse_session(poses: list[PoseCapture]) -> o3d.geometry.TriangleMesh:
    return integrate(poses, view_extrinsics(poses))


def build_textured_mesh(mesh: o3d.geometry.TriangleMesh,
                        poses: list[PoseCapture],
                        extrinsics: list[np.ndarray],
                        tex_size: int = 2048,
                        color_frames: list[ColorFrame] = (),
                        color_extrinsics: list[np.ndarray] = (),
                        ) -> "o3d.t.geometry.TriangleMesh":
    """Bake the original full-resolution photos onto `mesh` as a real UV texture
    atlas (vs. per-vertex colour). `extrinsics` are world->camera matrices in
    the SAME coordinate frame as `mesh`. Returns a tensor mesh with the texture.

    Open3D's project_images_to_albedo is x86-only, so we do the projection
    ourselves: bake per-texel surface position+normal into the atlas, then
    sample each texel from the photo that sees it most head-on. Texture detail
    is then limited by the photo resolution, not the mesh density.
    """
    clean = o3d.geometry.TriangleMesh(mesh)
    # compute_uvatlas needs a manifold mesh; TSDF + crop can leave bad edges.
    clean.remove_non_manifold_edges()
    clean.remove_degenerate_triangles()
    clean.remove_unreferenced_vertices()
    clean.compute_vertex_normals()

    t = o3d.t.geometry.TriangleMesh.from_legacy(clean)
    t.vertex["coverage"] = o3d.core.Tensor(
        np.ones((len(clean.vertices), 1), dtype=np.float32))
    t.compute_uvatlas(size=tex_size, parallel_partitions=4)
    baked = t.bake_vertex_attr_textures(
        tex_size, {"positions", "normals", "coverage"}, margin=2, fill=0.0)

    pos = baked["positions"].numpy().reshape(-1, 3).astype(np.float64)
    nrm = baked["normals"].numpy().reshape(-1, 3).astype(np.float64)
    covered = baked["coverage"].numpy().reshape(-1) > 0.5
    albedo = _project_photos_to_albedo(pos, nrm, covered, poses, extrinsics, tex_size,
                                       color_frames, color_extrinsics)

    t.material.set_default_properties()
    t.material.material_name = "defaultLit"
    t.material.texture_maps["albedo"] = o3d.t.geometry.Image(o3d.core.Tensor(albedo))
    return t


def _project_photos_to_albedo(pos: np.ndarray, nrm: np.ndarray, covered: np.ndarray,
                              poses: list[PoseCapture], extrinsics: list[np.ndarray],
                              tex_size: int,
                              color_frames: list[ColorFrame] = (),
                              color_extrinsics: list[np.ndarray] = ()) -> np.ndarray:
    """Per-texel winner-take-all photo projection + gutter fill, given each
    texel's baked surface position/normal/coverage. Returns (tex_size, tex_size, 3)
    uint8. Shared by the atlas and the cylindrical-UV texturers.

    Best-view-wins: each texel takes colour from the SINGLE most head-on photo
    that sees it, never an average. Averaging across views blends sub-millimetre
    ICP misalignments into a ghosted/blurred texture even though every source
    photo is sharp; winner-take-all keeps each texel as crisp as its photo.
    """
    n_tex = pos.shape[0]
    homog = np.c_[pos, np.ones(n_tex)]
    albedo = np.full((n_tex, 3), 160.0)   # neutral for unseen texels
    best_w = np.zeros(n_tex)
    for pose, ext in _color_sources(poses, extrinsics, color_frames, color_extrinsics):
        if pose.color is None or pose.rgb_intrinsics is None:
            continue
        cam = homog @ ext.T
        z = cam[:, 2]
        r = pose.rgb_intrinsics
        with np.errstate(divide="ignore", invalid="ignore"):
            u = r.fx * cam[:, 0] / z + r.cx
            v = r.fy * cam[:, 1] / z + r.cy
        H, W = pose.color.shape[:2]
        in_view = covered & (z > 1.0) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        ncam = nrm @ ext[:3, :3].T
        facing = np.clip(-ncam[:, 2], 0.0, None)   # 0..1, head-on = 1
        w = facing * in_view
        take = w > best_w
        if take.any():
            col = _bilinear(pose.color, np.clip(u, 0, W - 1), np.clip(v, 0, H - 1))
            albedo[take] = col[take]
            best_w[take] = w[take]
    albedo = np.clip(albedo, 0, 255).astype(np.uint8).reshape(tex_size, tex_size, 3)

    # Fill the gutters/empty texels (between charts, or facing no camera) with the
    # nearest filled texel's colour (standard atlas dilation) so empty areas carry
    # skin colour instead of flat grey.
    filled = (best_w > 0).reshape(tex_size, tex_size)
    if filled.any() and not filled.all():
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(
            ~filled, return_distances=False, return_indices=True)
        albedo = albedo[idx[0], idx[1]]
    return albedo


def build_cylindrical_textured_mesh(mesh: o3d.geometry.TriangleMesh,
                                    poses: list[PoseCapture],
                                    extrinsics: list[np.ndarray],
                                    tex_size: int = 2048,
                                    color_frames: list[ColorFrame] = (),
                                    color_extrinsics: list[np.ndarray] = (),
                                    ) -> "o3d.t.geometry.TriangleMesh":
    """Bake the photos onto a SINGLE-chart cylindrical UV unwrap, an alternative
    to both the fragmented compute_uvatlas (chart-edge "greyish lines") and the
    seamless-but-softer per-vertex colour.

    The head is unwrapped around its vertical (y) axis: azimuth -> u (front,+z, is
    u=0.5), height -> v. That yields ONE contiguous chart whose only seam is at the
    back of the head (-z), which is hidden/cropped — so the texture is both
    photo-sharp AND free of the atlas's web of internal seams. `extrinsics` are
    world->camera matrices in the SAME frame as `mesh`.
    """
    clean = o3d.geometry.TriangleMesh(mesh)
    clean.remove_non_manifold_edges()
    clean.remove_degenerate_triangles()
    clean.remove_unreferenced_vertices()
    clean.compute_vertex_normals()

    verts = np.asarray(clean.vertices)
    tris = np.asarray(clean.triangles)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    u = np.arctan2(x, z) / (2.0 * np.pi) + 0.5       # 0..1, front (+z) -> 0.5
    spany = max(float(y.max() - y.min()), 1e-6)
    v = (y - float(y.min())) / spany
    vert_uv = np.stack([u, 1.0 - v], axis=1)         # image v: crown at top

    tri_uv = vert_uv[tris].astype(np.float32)        # (T, 3, 2)
    # Triangles straddling the back seam (some u≈0, some u≈1) would rasterize as a
    # band across the whole texture width and clobber face texels — collapse them
    # to a corner. They are hidden back-of-head geometry, so this is invisible.
    uspan = tri_uv[:, :, 0].max(axis=1) - tri_uv[:, :, 0].min(axis=1)
    tri_uv[uspan > 0.5] = 0.0

    t = o3d.t.geometry.TriangleMesh.from_legacy(clean)
    t.triangle["texture_uvs"] = o3d.core.Tensor(tri_uv)
    t.vertex["coverage"] = o3d.core.Tensor(
        np.ones((len(verts), 1), dtype=np.float32))
    baked = t.bake_vertex_attr_textures(
        tex_size, {"positions", "normals", "coverage"}, margin=2, fill=0.0)
    pos = baked["positions"].numpy().reshape(-1, 3).astype(np.float64)
    nrm = baked["normals"].numpy().reshape(-1, 3).astype(np.float64)
    covered = baked["coverage"].numpy().reshape(-1) > 0.5
    albedo = _project_photos_to_albedo(pos, nrm, covered, poses, extrinsics, tex_size,
                                       color_frames, color_extrinsics)

    t.material.set_default_properties()
    t.material.material_name = "defaultLit"
    t.material.texture_maps["albedo"] = o3d.t.geometry.Image(o3d.core.Tensor(albedo))
    return t
