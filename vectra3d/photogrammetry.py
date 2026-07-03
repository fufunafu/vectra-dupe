"""Photoreal reconstruction via Apple Object Capture (PhotogrammetrySession).

The TSDF path (`fuse.py`) fuses 9 low-res TrueDepth depth frames; on smooth skin
the per-frame ICP slides and the frames misregister, shredding the surface. Apple
Object Capture instead solves all the dense RGB orbit frames jointly and produces
a clean, photoreal textured mesh (Kiri/Polycam tier) — but in an ARBITRARY scale
and pose. To use it clinically we must put it in true millimetres and the
canonical ARKit face frame the volume/compare pipeline expects. We do that with
facial landmarks, which give exact metric correspondences:

  1. shell out to the `ocrecon` Swift CLI -> a textured OBJ (arbitrary scale/pose),
  2. ray-cast a textured frontal RENDER of that OBJ and run MediaPipe on it -> 478
     facial landmarks in the OBJ's own frame (orientation found by trying the PCA
     axes and keeping the render MediaPipe actually detects a face in),
  3. unproject the SAME 478 landmarks through the TrueDepth keyframes -> their true
     metric positions in the ARKit world frame,
  4. solve a similarity (Umeyama, with scaling) between the two -> the transform
     that drops the OBJ onto the metric face frame (scale recovered to ~1-2 %,
     validated against inter-pupillary distance).

`processing.py` then runs the existing `normalize_to_front_frame` on the result,
so everything downstream (viewer, compare, volume) is unchanged. If the tool or
the landmark detector is missing, or the alignment is implausible, the caller
falls back to TSDF (which is metric by construction).
"""

import glob
import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from . import fuse
from .io_session import ColorFrame, PoseCapture

# --- external tools -------------------------------------------------------- #
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Object Capture CLI, built by `swift build -c release` in tools/photogrammetry.
OCRECON_BIN = os.path.join(_REPO_ROOT, "tools", "photogrammetry",
                           ".build", "release", "ocrecon")
# MediaPipe lives in an isolated venv (numpy 2 / mediapipe 0.10) with a CLI that
# prints pixel-space landmarks as JSON. Overridable for other layouts.
MP_PYTHON = os.environ.get(
    "VECTRA_MP_PYTHON",
    os.path.join(_REPO_ROOT, "VECTRA-SOFTWARE-ONLY", ".venv-mp", "bin", "python"))
LANDMARK_SCRIPT = os.environ.get(
    "VECTRA_LANDMARK_SCRIPT",
    os.path.join(_REPO_ROOT, "VECTRA-SOFTWARE-ONLY", "tools", "landmark_detect.py"))

# MediaPipe FaceLandmarker indices for the iris centres (used for the IPD guard).
LEFT_IRIS_LM, RIGHT_IRIS_LM = 468, 473

# Detail level for PhotogrammetrySession. `full` is Kiri tier; `medium`/`reduced`
# are faster. Overridable via VECTRA_OC_DETAIL.
OC_DETAIL = os.environ.get("VECTRA_OC_DETAIL", "full")

# Views whose TrueDepth keyframes give the metric landmark reference (front-most
# views have the best depth + the most detectable landmarks).
REF_VIEWS = ("front", "left_half", "right_half")

# Alignment guards — reject (=> TSDF fallback) rather than corrupt measurements.
MIN_CORRESPONDENCES = 60
MAX_ALIGN_RMS_MM = 8.0
IPD_MIN_MM, IPD_MAX_MM = 48.0, 82.0     # human inter-pupillary distance range
HEAD_HEIGHT_MIN_MM, HEAD_HEIGHT_MAX_MM = 140.0, 340.0

# PhotogrammetrySession is non-deterministic: a fresh reconstruction varies, and
# its landmark alignment rms swings run-to-run (observed 2.7–8.3 mm), so a single
# attempt occasionally trips a guard and drops to the TSDF display. We run a few
# attempts and keep the lowest-rms one that passes every guard. Overridable via
# VECTRA_OC_ATTEMPTS. Each attempt re-runs ocrecon (~20 s at full detail).
OC_ATTEMPTS = max(1, int(os.environ.get("VECTRA_OC_ATTEMPTS", "3")))


@dataclass
class OCResult:
    """Object Capture output already in the metric ARKit world frame."""
    mesh: o3d.geometry.TriangleMesh                     # per-vertex colour (measurement + mesh.glb)
    textured: "o3d.t.geometry.TriangleMesh | None"      # UV atlas + albedo (mesh_textured.glb)
    stats: dict = field(default_factory=dict)


def tool_available() -> bool:
    """The OC CLI is built and we're on an Apple-Silicon Mac that can run it."""
    return (os.path.isfile(OCRECON_BIN)
            and os.access(OCRECON_BIN, os.X_OK)
            and platform.system() == "Darwin"
            and platform.machine() == "arm64")


def landmark_tooling_available() -> bool:
    """The MediaPipe venv + landmark CLI needed for metric alignment exist."""
    return os.path.isfile(MP_PYTHON) and os.path.isfile(LANDMARK_SCRIPT)


# --------------------------------------------------------------------------- #
# OBJ + texture loading (Model I/O's OBJ trips Open3D's ASSIMP importer, so we
# parse the simple single-submesh format ourselves).
# --------------------------------------------------------------------------- #

def _load_obj_with_texture(obj_path: str):
    """Parse the ocrecon OBJ -> (V, F, tri_uv, albedo).

    `V` (n,3) vertices, `F` (t,3) triangle vertex indices, `tri_uv` (t,3,2)
    per-corner UVs already flipped to image-space (v=0 at top, the same
    convention `fuse.build_cylindrical_textured_mesh` writes), and `albedo`
    (H,W,3) uint8 (or None if no diffuse map was found).
    """
    verts: list[list[float]] = []
    uvs: list[list[float]] = []
    faces: list[list[int]] = []
    face_uvs: list[list[int]] = []
    for line in open(obj_path):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            uvs.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("f "):
            vi, ti = [], []
            for tok in line.split()[1:]:
                p = tok.split("/")
                vi.append(int(p[0]) - 1)
                ti.append(int(p[1]) - 1 if len(p) > 1 and p[1] else -1)
            for k in range(1, len(vi) - 1):          # fan-triangulate any polygon
                faces.append([vi[0], vi[k], vi[k + 1]])
                face_uvs.append([ti[0], ti[k], ti[k + 1]])

    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    VT = np.asarray(uvs, dtype=np.float64) if uvs else np.zeros((0, 2))
    FT = np.asarray(face_uvs, dtype=np.int64)

    tri_uv = None
    if len(VT) and len(FT) and (FT >= 0).all():
        tri_uv = VT[FT].astype(np.float32)           # (t,3,2), OBJ bottom-left origin
        tri_uv[:, :, 1] = 1.0 - tri_uv[:, :, 1]      # -> image-space (v=0 top)

    albedo = _load_diffuse_texture(obj_path)
    return V, F, tri_uv, albedo


def _load_diffuse_texture(obj_path: str):
    """Read the `map_Kd` PNG referenced by the OBJ's .mtl -> (H,W,3) uint8 / None."""
    mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
    if not os.path.isfile(mtl_path):
        return None
    tex_name = None
    for line in open(mtl_path):
        if line.strip().startswith("map_Kd"):
            tex_name = line.split(None, 1)[1].strip()
            break
    if not tex_name:
        return None
    tex_path = os.path.join(os.path.dirname(obj_path), tex_name)
    if not os.path.isfile(tex_path):
        return None
    img = np.asarray(o3d.io.read_image(tex_path))
    if img.ndim == 3 and img.shape[2] >= 3:
        return np.ascontiguousarray(img[:, :, :3], dtype=np.uint8)
    return None


def _sample_albedo(tri_uv: np.ndarray, F: np.ndarray, n_verts: int,
                   albedo: np.ndarray) -> np.ndarray:
    """One representative colour per vertex (n,3) float 0..1, sampled from the
    albedo at the first triangle-corner that references each vertex."""
    H, W = albedo.shape[:2]
    vert_uv = np.zeros((n_verts, 2), dtype=np.float64)
    seen = np.zeros(n_verts, dtype=bool)
    for ti in range(len(F)):
        for corner in range(3):
            vi = F[ti, corner]
            if not seen[vi]:
                vert_uv[vi] = tri_uv[ti, corner]
                seen[vi] = True
    u = np.clip(vert_uv[:, 0] * (W - 1), 0, W - 1)
    v = np.clip(vert_uv[:, 1] * (H - 1), 0, H - 1)
    col = fuse._bilinear(albedo, u, v) / 255.0
    col[~seen] = 0.6
    return np.clip(col, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Landmark-anchored metric alignment
# --------------------------------------------------------------------------- #

def _detect_landmarks(image_path: str) -> dict | None:
    """Run the MediaPipe CLI on an image -> its JSON dict (or None)."""
    try:
        proc = subprocess.run([MP_PYTHON, LANDMARK_SCRIPT, image_path],
                              capture_output=True, text=True, timeout=120)
    except (subprocess.SubprocessError, OSError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        d = json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return None
    return d if d.get("ok") else None


def _ref_landmarks_world(raw_dir: str, poses: list[PoseCapture]):
    """Metric 3D positions of the 478 face landmarks in the ARKit world frame.

    The front view's depth is the most accurate (live anchor, head-on), so each
    landmark is taken from the FIRST listed view that sees it (front first); the
    half views only fill in landmarks the front depth can't reach. Averaging
    across views was tried and hurt — the oblique half-view depth pulls central
    landmarks off and inflates the alignment residual. Returns (L (478,3), valid)."""
    by_name = {p.name: p for p in poses}
    L = np.zeros((478, 3))
    valid = np.zeros(478, dtype=bool)
    for view in REF_VIEWS:
        pose = by_name.get(view)
        img_path = os.path.join(raw_dir, f"color_{view}.jpg")
        if pose is None or not os.path.isfile(img_path):
            continue
        lmk = _detect_landmarks(img_path)
        if lmk is None or len(lmk.get("landmarks", [])) != 478:
            continue
        pw, ok = _unproject_landmarks(pose, np.asarray(lmk["landmarks"]))
        take = ok & ~valid
        L[take] = pw[take]
        valid |= take
    return L, valid


def _unproject_landmarks(pose: PoseCapture, pts2d: np.ndarray):
    """Back-project colour-image landmark pixels through this pose's depth into
    world coords. Returns (world (n,3), valid)."""
    r, di = pose.rgb_intrinsics, pose.intrinsics
    dd = pose.depth
    dh, dw = dd.shape
    # colour pixel -> depth pixel (inverse of fuse._aligned_color's remap)
    ud = (pts2d[:, 0] - r.cx) * (di.fx / r.fx) + di.cx
    vd = (pts2d[:, 1] - r.cy) * (di.fy / r.fy) + di.cy
    iu = np.round(ud).astype(int)
    iv = np.round(vd).astype(int)
    valid = (iu >= 0) & (iu < dw) & (iv >= 0) & (iv < dh)
    iu = np.clip(iu, 0, dw - 1)
    iv = np.clip(iv, 0, dh - 1)
    Z = dd[iv, iu]
    valid &= Z > 0
    Xc = (ud - di.cx) / di.fx * Z
    Yc = (vd - di.cy) / di.fy * Z
    cam = np.stack([Xc, Yc, Z, np.ones_like(Z)], axis=1)
    world = (np.linalg.inv(pose.world_to_camera) @ cam.T).T[:, :3]
    return world, valid


def _render_textured(Vor: np.ndarray, F: np.ndarray, tri_uv: np.ndarray,
                     albedo: np.ndarray, size: int = 512, cz: float = 2.2):
    """Ray-cast a textured frontal render of a normalized-into-unit-cube mesh
    (camera on +z looking toward -z). Returns (img HxWx3 uint8, hitpos HxWx3
    world coords, hit HxW bool)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(Vor.astype(np.float32)),
        o3d.core.Tensor(F.astype(np.uint32))))
    f = 1.05 * size
    cx = cy = size / 2.0
    us, vs = np.meshgrid(np.arange(size), np.arange(size))
    dx = (us - cx) / f
    dy = -(vs - cy) / f                 # image y down -> world y up
    dz = -np.ones_like(dx)
    dirs = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    ori = np.tile([0.0, 0.0, cz], (len(dirs), 1))
    ans = scene.cast_rays(o3d.core.Tensor(np.hstack([ori, dirs]).astype(np.float32)))
    t_hit = ans["t_hit"].numpy()
    pid = ans["primitive_ids"].numpy()
    bary = ans["primitive_uvs"].numpy()
    hit = np.isfinite(t_hit)
    hitpos = ori + dirs * np.where(hit, t_hit, 0.0)[:, None]

    img = np.full((len(dirs), 3), 255, np.uint8)
    if hit.any():
        b1, b2 = bary[hit, 0], bary[hit, 1]
        b0 = 1.0 - b1 - b2
        tu = tri_uv[pid[hit]]                                  # (n,3,2)
        uv = b0[:, None] * tu[:, 0] + b1[:, None] * tu[:, 1] + b2[:, None] * tu[:, 2]
        Ha, Wa = albedo.shape[:2]
        col = fuse._bilinear(albedo, np.clip(uv[:, 0] * (Wa - 1), 0, Wa - 1),
                             np.clip(uv[:, 1] * (Ha - 1), 0, Ha - 1))
        img[hit] = np.clip(col, 0, 255).astype(np.uint8)
    return (img.reshape(size, size, 3),
            hitpos.reshape(size, size, 3),
            hit.reshape(size, size))


def _oc_landmarks(V, F, tri_uv, albedo, work_dir):
    """3D landmarks on the OBJ in a normalized, roughly-oriented frame.

    The OBJ frame is arbitrary, so we try the PCA axes (4th/8 sign+face combos)
    as the facing direction and keep the render MediaPipe detects the largest
    face in. Returns (ocL (478,3), valid (478,), N4 4x4 mapping original OBJ
    verts -> the rendered frame, render_img) or None."""
    center = (V.max(0) + V.min(0)) / 2.0
    ext = float((V.max(0) - V.min(0)).max())
    Vn = (V - center) / ext
    cov = np.cov((Vn - Vn.mean(0)).T)
    evals, evecs = np.linalg.eigh(cov)
    axes = evecs[:, np.argsort(evals)[::-1]]          # columns, descending

    best = None
    render_path = os.path.join(work_dir, "oc_render.png")
    for face_axis in (1, 2):                           # axis 0 is the tall (up) axis
        for fsign in (1.0, -1.0):
            for usign in (1.0, -1.0):
                zc = axes[:, face_axis] * fsign
                zc /= np.linalg.norm(zc)
                up = axes[:, 0] * usign
                yc = up - np.dot(up, zc) * zc
                yc /= np.linalg.norm(yc)
                xc = np.cross(yc, zc)
                Rwc = np.stack([xc, yc, zc], axis=1)  # rows->new axes
                Vor = Vn @ Rwc
                img, hitpos, hit = _render_textured(Vor, F, tri_uv, albedo)
                o3d.io.write_image(render_path, o3d.geometry.Image(
                    np.ascontiguousarray(img)))
                lmk = _detect_landmarks(render_path)
                if lmk is None or len(lmk.get("landmarks", [])) != 478:
                    continue
                pts = np.asarray(lmk["landmarks"])
                area = float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]))
                if best is None or area > best[0]:
                    best = (area, Rwc, center, ext, hitpos, hit, pts, img)

    if best is None:
        return None
    _, Rwc, center, ext, hitpos, hit, pts, img = best
    iu = np.clip(np.round(pts[:, 0]).astype(int), 0, hit.shape[1] - 1)
    iv = np.clip(np.round(pts[:, 1]).astype(int), 0, hit.shape[0] - 1)
    ocL = hitpos[iv, iu]
    ocOK = hit[iv, iu]

    # N4: original OBJ vert -> rendered (normalized+oriented) frame.
    N4 = np.eye(4)
    N4[:3, :3] = Rwc.T / ext
    N4[:3, 3] = -(Rwc.T @ center) / ext
    return ocL, ocOK, N4, img


def _umeyama(src: np.ndarray, dst: np.ndarray):
    """Closed-form similarity (scale+R+t) mapping src->dst. Returns (T 4x4, scale)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (s ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(D) @ S) / var)
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = mu_d - scale * R @ mu_s
    return T, scale


def _solve_similarity(oc: np.ndarray, ref: np.ndarray):
    """Robust Umeyama with one round of inlier rejection. Returns (T, scale,
    rms_mm, n_inliers)."""
    T, _ = _umeyama(oc, ref)
    res = np.linalg.norm((T[:3, :3] @ oc.T).T + T[:3, 3] - ref, axis=1)
    keep = res <= np.percentile(res, 85)
    T, scale = _umeyama(oc[keep], ref[keep])
    res = np.linalg.norm((T[:3, :3] @ oc[keep].T).T + T[:3, 3] - ref[keep], axis=1)
    return T, scale, float(res.mean()), int(keep.sum())


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _stage_images(raw_dir: str, work_dir: str) -> int:
    """Copy the colour photos (keyframes + orbit) into a flat folder for OC."""
    n = 0
    for src in sorted(glob.glob(os.path.join(raw_dir, "color_*.jpg"))):
        shutil.copy2(src, os.path.join(work_dir, os.path.basename(src)))
        n += 1
    return n


def _run_ocrecon(image_dir: str, out_obj: str) -> dict:
    """Run the CLI; returns its final-line JSON ({output,images_used,seconds})."""
    proc = subprocess.run(
        [OCRECON_BIN, image_dir, out_obj, "--detail", OC_DETAIL,
         "--feature-sensitivity", "high"],
        capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.isfile(out_obj):
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"ocrecon failed (exit {proc.returncode}): {tail}")
    last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return {}


def _make_textured_tensor(V_world: np.ndarray, F: np.ndarray,
                          tri_uv: np.ndarray, albedo: np.ndarray
                          ) -> "o3d.t.geometry.TriangleMesh":
    """Tensor mesh in world frame carrying the OC UV atlas + albedo."""
    t = o3d.t.geometry.TriangleMesh()
    t.vertex.positions = o3d.core.Tensor(V_world.astype(np.float32))
    t.triangle.indices = o3d.core.Tensor(F.astype(np.int64))
    t.triangle["texture_uvs"] = o3d.core.Tensor(tri_uv.astype(np.float32))
    t.material.set_default_properties()
    t.material.material_name = "defaultLit"
    t.material.texture_maps["albedo"] = o3d.t.geometry.Image(
        o3d.core.Tensor(np.ascontiguousarray(albedo)))
    return t


def reconstruct_metric(raw_dir: str, poses: list[PoseCapture],
                       color_frames: list[ColorFrame], out_dir: str,
                       attempts: int = OC_ATTEMPTS) -> OCResult:
    """Reconstruct via Object Capture, best of `attempts` runs.

    Each attempt is a full fresh reconstruction+alignment (PhotogrammetrySession
    is non-deterministic). We keep the lowest-rms result that clears every metric
    guard; if none do, we re-raise the last failure so the caller falls back to
    TSDF. `stats` gains `oc_attempts` (run) and `oc_attempts_passed`."""
    best: OCResult | None = None
    last_err: Exception | None = None
    n_passed = 0
    for i in range(max(1, attempts)):
        try:
            res = _reconstruct_metric_once(raw_dir, poses, color_frames, out_dir)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[photogrammetry] attempt {i + 1}/{attempts} rejected: {e}")
            continue
        n_passed += 1
        if best is None or res.stats["align_rms_mm"] < best.stats["align_rms_mm"]:
            best = res
        print(f"[photogrammetry] attempt {i + 1}/{attempts} ok: "
              f"rms={res.stats['align_rms_mm']}mm ipd={res.stats['align_ipd_mm']}mm")
    if best is None:
        raise last_err if last_err else RuntimeError("no Object Capture attempt succeeded")
    best.stats["oc_attempts"] = attempts
    best.stats["oc_attempts_passed"] = n_passed
    return best


def _reconstruct_metric_once(raw_dir: str, poses: list[PoseCapture],
                             color_frames: list[ColorFrame], out_dir: str) -> OCResult:
    """One Object Capture reconstruction + metric alignment. Raises on any failure
    or guard violation."""
    if not poses:
        raise RuntimeError("no depth keyframes for metric anchoring")
    if not landmark_tooling_available():
        raise RuntimeError("MediaPipe landmark tooling unavailable")

    work = tempfile.mkdtemp(prefix="oc_", dir=out_dir)
    try:
        n_img = _stage_images(raw_dir, work)
        if n_img < 8:
            raise RuntimeError(f"too few photos for photogrammetry ({n_img})")
        out_obj = os.path.join(work, "model.obj")
        run_info = _run_ocrecon(work, out_obj)

        V, F, tri_uv, albedo = _load_obj_with_texture(out_obj)
        if len(V) == 0 or len(F) == 0:
            raise RuntimeError("ocrecon produced an empty mesh")
        if albedo is None or tri_uv is None:
            raise RuntimeError("ocrecon mesh has no texture for landmark alignment")

        # 1) landmarks on the OBJ (its own frame) and on the metric depth views.
        oc = _oc_landmarks(V, F, tri_uv, albedo, work)
        if oc is None:
            raise RuntimeError("no face detected in any OBJ render")
        ocL, ocOK, N4, _ = oc
        refL, refOK = _ref_landmarks_world(raw_dir, poses)

        # 2) solve the metric similarity from the shared landmarks.
        m = ocOK & refOK
        if int(m.sum()) < MIN_CORRESPONDENCES:
            raise RuntimeError(f"too few landmark correspondences ({int(m.sum())})")
        T_u, scale_u, rms, n_inl = _solve_similarity(ocL[m], refL[m])
        if rms > MAX_ALIGN_RMS_MM:
            raise RuntimeError(f"poor landmark alignment (rms={rms:.1f}mm)")

        # IPD sanity (independent metric check on the recovered scale).
        worldL = (T_u[:3, :3] @ ocL.T).T + T_u[:3, 3]
        ipd = float(np.linalg.norm(worldL[LEFT_IRIS_LM] - worldL[RIGHT_IRIS_LM]))
        if not (IPD_MIN_MM <= ipd <= IPD_MAX_MM):
            raise RuntimeError(f"implausible inter-pupillary distance {ipd:.1f}mm")

        # 3) full transform: original OBJ vert -> metric world frame.
        M = T_u @ N4
        V_world = (M[:3, :3] @ V.T).T + M[:3, 3]

        # head-height plausibility (around the anchor origin).
        head = V_world[np.linalg.norm(V_world, axis=1) <= 135.0]
        head_h = float(np.ptp(head[:, 1])) if len(head) else 0.0
        if not (HEAD_HEIGHT_MIN_MM <= head_h <= HEAD_HEIGHT_MAX_MM):
            raise RuntimeError(f"implausible head height {head_h:.0f}mm")

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(V_world), o3d.utility.Vector3iVector(F))
        mesh.compute_vertex_normals()
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            _sample_albedo(tri_uv, F, len(V_world), albedo))

        textured = _make_textured_tensor(V_world, F, tri_uv, albedo)

        metric_scale = float(np.linalg.norm(M[:3, 0]))   # mm per OBJ unit
        stats = {
            "reconstruction": "object_capture",
            "oc_detail": OC_DETAIL,
            "oc_images_used": run_info.get("images_used", n_img),
            "oc_seconds": run_info.get("seconds"),
            "metric_scale_mm_per_unit": round(metric_scale, 4),
            "align_rms_mm": round(rms, 3),
            "align_correspondences": n_inl,
            "align_ipd_mm": round(ipd, 2),
            "align_method": "landmark_umeyama",
        }
        return OCResult(mesh=mesh, textured=textured, stats=stats)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _keep_main_components_tri(V: np.ndarray, F: np.ndarray,
                              min_frac: float = 0.05,
                              max_offset_mm: float = 70.0) -> np.ndarray:
    """Triangle keep-mask dropping disconnected floaters, mirroring
    `processing.keep_main_components` but returned as a mask so the caller can
    apply it to the parallel per-triangle UV array. Keeps the largest connected
    component plus any sizeable component centred within `max_offset_mm` of it."""
    comp = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    labels, counts, _ = comp.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if len(counts) == 0:
        return np.ones(len(F), dtype=bool)
    biggest = int(counts.argmax())
    face_center = V[np.unique(F[labels == biggest])].mean(axis=0)
    size_ok = counts >= max(1, int(counts.max() * min_frac))
    keep = np.zeros(len(counts), dtype=bool)
    keep[biggest] = True
    for ci in np.nonzero(size_ok)[0]:
        if ci == biggest:
            continue
        c = V[np.unique(F[labels == ci])].mean(axis=0)
        if np.linalg.norm(c - face_center) <= max_offset_mm:
            keep[ci] = True
    return keep[labels]


def write_normalized_textured_glb(textured: "o3d.t.geometry.TriangleMesh",
                                  world_to_norm: np.ndarray, out_path: str,
                                  crop_center: np.ndarray, crop_radius_mm: float,
                                  smooth_iters: int = 0) -> bool:
    """Clean the world-frame textured mesh for display, move it into the
    normalized frame, and write the GLB (UVs/albedo preserved). Best-effort.

    OC's raw atlas mesh has a ragged periphery and disconnected hair/neck
    fragments that read as a gray blob halo around the face. We (1) sphere-crop to
    a face-focused radius, (2) drop disconnected floaters, and (3) optionally apply
    a few Taubin iterations. UVs are stored per triangle-corner, so dropping
    triangles and moving vertex positions leaves the atlas mapping intact."""
    try:
        V = textured.vertex.positions.numpy().astype(np.float64)
        F = textured.triangle.indices.numpy().astype(np.int64)
        tri_uv = textured.triangle["texture_uvs"].numpy()

        # 1) sphere crop — clips hair/neck periphery and the ragged outline.
        keep_v = np.linalg.norm(V - crop_center, axis=1) <= crop_radius_mm
        tri_keep = keep_v[F].all(axis=1)
        if not tri_keep.any():
            return False
        F_keep = F[tri_keep]
        uv_keep = tri_uv[tri_keep]

        # 2) drop disconnected floaters (gray hair/neck fragments OC leaves around
        #    the face).
        comp_keep = _keep_main_components_tri(V, F_keep)
        F_keep = F_keep[comp_keep]
        uv_keep = uv_keep[comp_keep]
        if len(F_keep) == 0:
            return False

        # compact the vertex set to the surviving triangles.
        used = np.unique(F_keep)
        remap = -np.ones(len(V), dtype=np.int64)
        remap[used] = np.arange(len(used))
        F2 = remap[F_keep]
        V2 = V[used]
        uv2 = uv_keep

        # 3) optional light Taubin (positions only; triangles/UVs unchanged).
        if smooth_iters:
            sm = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(V2), o3d.utility.Vector3iVector(F2))
            sm = sm.filter_smooth_taubin(number_of_iterations=smooth_iters)
            V2 = np.asarray(sm.vertices)

        V2 = (world_to_norm[:3, :3] @ V2.T).T + world_to_norm[:3, 3]

        albedo = textured.material.texture_maps["albedo"]
        out = _make_textured_tensor(V2, F2, uv2, albedo.as_tensor().numpy())
        o3d.t.io.write_triangle_mesh(out_path, out)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[photogrammetry] textured glb skipped: {e}")
        return False
