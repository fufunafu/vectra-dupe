"""Photo-projection texture baking (torch-free process).

Takes the fused mesh (VGGT world frame) and projects the ORIGINAL full-res
clinical photos onto a UV atlas: decimate -> xatlas unwrap -> per-texel
weighted blend over all views that see the texel (weight = facing^3 x head-mask,
visibility via per-view ray-cast depth maps). Output is a textured GLB — the
step that lifts renders from vertex-color splats to photographic skin.
"""
from __future__ import annotations
import os

import cv2
import numpy as np
import open3d as o3d

ATLAS_RES = 4096
TARGET_TRIS = 200_000
FACING_POWER = 3.0
DEPTH_REL_TOL = float(os.environ.get("VECTRA_BAKE_DEPTH_TOL", "0.015"))
# visibility: |z - depth| < tol * median depth; raise for bumpier meshes whose
# self-occlusion at noise scale otherwise rejects valid texels (gray speckle)
DEPTH_MAP_SCALE = 0.25         # ray-cast depth maps at 1/4 of photo resolution
MIN_WEIGHT = 0.05


def bake(mesh: o3d.geometry.TriangleMesh, names: list[str],
         intrinsic_vggt: np.ndarray, extrinsic: np.ndarray,
         vggt_hw: tuple[int, int], photo_paths: dict[str, str],
         mask_dir: str | None, out_glb: str,
         canonical: np.ndarray | None = None,
         n_render_samples: int = 1_500_000):
    """Bake a UV texture from the original photos onto `mesh` (world frame).

    names/intrinsic_vggt/extrinsic: per-view, at VGGT resolution `vggt_hw`.
    photo_paths: view name -> original full-res photo path.
    canonical: optional 3x3 basis applied to positions before writing (the
    display orientation finish_vggt uses), texture unaffected.
    """
    # --- decimate + unwrap
    m = o3d.geometry.TriangleMesh(mesh)
    if len(m.triangles) > TARGET_TRIS:
        m = m.simplify_quadric_decimation(target_number_of_triangles=TARGET_TRIS)
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    V = np.asarray(m.vertices)
    F = np.asarray(m.triangles)
    VN = np.asarray(m.vertex_normals)

    import xatlas
    vmap, F_uv, UV = xatlas.parametrize(V.astype(np.float32), F.astype(np.uint32))
    V2, N2 = V[vmap], VN[vmap]            # vertex-split copies per xatlas
    print(f"[texture] unwrapped {len(V2)} verts / {len(F_uv)} tris", flush=True)

    # --- rasterize the atlas: per-texel 3D position + normal
    pos, nrm, texel_mask = _rasterize_atlas(V2, N2, F_uv.astype(np.int64), UV)
    tex_pts = pos[texel_mask]
    tex_nrm = nrm[texel_mask]
    print(f"[texture] {len(tex_pts)} texels to bake", flush=True)

    # --- visibility scene
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(V.astype(np.float32)),
        o3d.core.Tensor(F.astype(np.uint32))))

    Hv, Wv = vggt_hw
    acc = np.zeros((len(tex_pts), 3), np.float64)
    wacc = np.zeros(len(tex_pts), np.float64)
    # loose accumulators ignore the visibility test — fallback for crease/grazing
    # texels that the 1/4-res depth maps mark occluded in every view
    acc_loose = np.zeros((len(tex_pts), 3), np.float64)
    wacc_loose = np.zeros(len(tex_pts), np.float64)
    z_all = []
    for s, name in enumerate(names):
        path = photo_paths.get(name)
        if path is None or not os.path.isfile(path):
            continue
        photo = cv2.imread(path)
        if photo is None:
            continue
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
        Hp, Wp = photo.shape[:2]
        K = intrinsic_vggt[s].copy()      # VGGT res -> original photo res
        K[0] *= Wp / Wv
        K[1] *= Hp / Hv
        R, t = extrinsic[s, :, :3], extrinsic[s, :, 3]

        cam = tex_pts @ R.T + t
        z = cam[:, 2]
        ok = z > 1e-9
        u = np.where(ok, cam[:, 0] / np.where(ok, z, 1) * K[0, 0] + K[0, 2], -1)
        v = np.where(ok, cam[:, 1] / np.where(ok, z, 1) * K[1, 1] + K[1, 2], -1)
        inb = ok & (u >= 0) & (u < Wp - 1) & (v >= 0) & (v < Hp - 1)
        if not inb.any():
            continue
        z_all.append(np.median(z[inb]))

        # facing weight
        cam_c = -R.T @ t
        vd = cam_c - tex_pts
        vd /= np.linalg.norm(vd, axis=1, keepdims=True) + 1e-12
        facing = np.einsum("ij,ij->i", tex_nrm, vd)
        w = np.clip(facing, 0, 1) ** FACING_POWER
        w[~inb] = 0.0

        w_loose = w.copy()

        # visibility via ray-cast depth map at reduced res
        wd, hd = int(Wp * DEPTH_MAP_SCALE), int(Hp * DEPTH_MAP_SCALE)
        Kd = K.copy()
        Kd[:2] *= DEPTH_MAP_SCALE
        T = np.vstack([extrinsic[s], [0, 0, 0, 1]])
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(Kd.astype(np.float64)),
            o3d.core.Tensor(T.astype(np.float64)), wd, hd)
        dmap = scene.cast_rays(rays)["t_hit"].numpy()   # (hd, wd) distance
        ud = np.clip((u * DEPTH_MAP_SCALE).astype(int), 0, wd - 1)
        vdx = np.clip((v * DEPTH_MAP_SCALE).astype(int), 0, hd - 1)
        dist = np.linalg.norm(tex_pts - cam_c, axis=1)
        tol = DEPTH_REL_TOL * np.median(z[inb])
        visible = np.isfinite(dmap[vdx, ud]) & (np.abs(dmap[vdx, ud] - dist) < tol)
        w[~visible] = 0.0

        # head mask (eroded, work res) — no clothing/backdrop bleed
        if mask_dir:
            mp = os.path.join(mask_dir, name + ".png")
            if os.path.isfile(mp):
                mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if mk is not None:
                    mk = cv2.resize(mk, (wd, hd), interpolation=cv2.INTER_NEAREST)
                    off = mk[vdx, ud] < 128
                    w[off] = 0.0
                    w_loose[off] = 0.0

        sel = (w > 0) | (w_loose > 0)
        if not sel.any():
            continue
        col = _bilinear(photo, u[sel], v[sel])
        acc[sel] += w[sel, None] * col
        wacc[sel] += w[sel]
        acc_loose[sel] += w_loose[sel, None] * col
        wacc_loose[sel] += w_loose[sel]

    baked = np.zeros((len(tex_pts), 3), np.float64)
    good = wacc > MIN_WEIGHT
    baked[good] = acc[good] / wacc[good, None]
    fallback = ~good & (wacc_loose > MIN_WEIGHT)
    baked[fallback] = acc_loose[fallback] / wacc_loose[fallback, None]
    good |= fallback
    print(f"[texture] baked {int(good.sum())}/{len(tex_pts)} texels "
          f"({int(fallback.sum())} via loose pass) from {len(z_all)} views",
          flush=True)

    atlas = np.zeros((ATLAS_RES, ATLAS_RES, 3), np.uint8)
    flat = np.zeros((ATLAS_RES * ATLAS_RES,), bool)
    idx = np.flatnonzero(texel_mask.reshape(-1))
    atlas.reshape(-1, 3)[idx[good]] = np.clip(baked[good], 0, 255).astype(np.uint8)
    flat[idx[good]] = True
    atlas, filled = _dilate_atlas(atlas, flat.reshape(ATLAS_RES, ATLAS_RES))
    hole = (texel_mask & ~filled).astype(np.uint8)
    if 0 < hole.sum() < 500_000:      # residual unbaked islands -> inpaint
        atlas = cv2.inpaint(atlas, hole, 3, cv2.INPAINT_TELEA)
        filled |= hole.astype(bool)
    atlas = _fill_gutter_nearest(atlas, filled)

    # --- build the textured mesh (world frame), sample dense colored points for
    # the splat renderer, then export the viewer GLB in canonical orientation
    import trimesh
    from PIL import Image
    # SimpleMaterial: trimesh can sample colors from it AND converts it to a PBR
    # baseColorTexture on GLB export.
    vis = trimesh.visual.TextureVisuals(
        uv=np.column_stack([UV[:, 0], UV[:, 1]]),
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.fromarray(atlas[::-1])))            # glTF v origin bottom
    tm = trimesh.Trimesh(vertices=V2, faces=F_uv, visual=vis, process=False)

    pts, fidx, cols = trimesh.sample.sample_surface(
        tm, n_render_samples, sample_color=True)
    nrms = tm.face_normals[fidx]
    samples = (np.asarray(pts), np.asarray(nrms),
               np.asarray(cols, np.float64)[:, :3] / 255.0)

    if canonical is not None:
        tm.vertices = (tm.vertices - tm.vertices.mean(0)) @ canonical
    tm.export(out_glb)
    print(f"[texture] wrote {out_glb}", flush=True)
    return out_glb, samples


def _rasterize_atlas(V2, N2, F_uv, UV, res: int = ATLAS_RES):
    """Barycentric-rasterize triangles into UV space. Returns per-texel
    (position (res,res,3), normal (res,res,3), covered (res,res) bool)."""
    pos = np.zeros((res, res, 3), np.float32)
    nrm = np.zeros((res, res, 3), np.float32)
    cov = np.zeros((res, res), bool)
    uv_px = UV * (res - 1)
    tris_uv = uv_px[F_uv]                # (t,3,2)
    tris_v = V2[F_uv].astype(np.float32)
    tris_n = N2[F_uv].astype(np.float32)
    for i in range(len(F_uv)):
        t2, t3, tn = tris_uv[i], tris_v[i], tris_n[i]
        lo = np.floor(t2.min(0)).astype(int)
        hi = np.ceil(t2.max(0)).astype(int) + 1
        if (hi - lo).min() <= 0:
            continue
        xs, ys = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]))
        p = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
        d = np.cross(t2[1] - t2[0], t2[2] - t2[0])
        if abs(d) < 1e-12:
            continue
        b1 = np.cross(p - t2[0], t2[2] - t2[0]) / d
        b2 = np.cross(t2[1] - t2[0], p - t2[0]) / d
        b0 = 1.0 - b1 - b2
        inside = (b0 >= -1e-4) & (b1 >= -1e-4) & (b2 >= -1e-4)
        if not inside.any():
            continue
        b = np.stack([b0, b1, b2], 1)[inside]
        px = p[inside].astype(int)
        pos[px[:, 1], px[:, 0]] = b @ t3
        n = b @ tn
        nrm[px[:, 1], px[:, 0]] = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        cov[px[:, 1], px[:, 0]] = True
    return pos, nrm, cov


def _dilate_atlas(atlas: np.ndarray, filled: np.ndarray, iters: int = 8):
    """Grow baked colors into empty texels so bilinear sampling at UV seams
    never reads black."""
    a = atlas.copy()
    f = filled.copy()
    k = np.ones((3, 3), np.uint8)
    for _ in range(iters):
        grown = cv2.dilate(f.astype(np.uint8), k).astype(bool)
        ring = grown & ~f
        if not ring.any():
            break
        blur = cv2.blur(a, (3, 3))
        wsum = cv2.blur(f.astype(np.float32), (3, 3))
        ys, xs = np.nonzero(ring)
        ok = wsum[ys, xs] > 1e-6
        a[ys[ok], xs[ok]] = (blur[ys[ok], xs[ok]] / wsum[ys[ok], xs[ok], None])
        f = grown
    return a, f


def _fill_gutter_nearest(atlas: np.ndarray, filled: np.ndarray) -> np.ndarray:
    """Replace every unfilled texel with its nearest filled texel's color, so any
    UV lookup (bilinear at chart seams, point samples in the gutter) reads
    plausible skin instead of black."""
    if filled.all():
        return atlas
    inv = (~filled).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        inv, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(filled)                # zero pixels of `inv`
    lut = np.zeros((labels.max() + 1, 3), np.uint8)
    lut[labels[ys, xs]] = atlas[ys, xs]
    out = lut[labels]
    out[filled] = atlas[filled]
    return out


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    c00 = img[y0, x0].astype(np.float64)
    c10 = img[y0, x1].astype(np.float64)
    c01 = img[y1, x0].astype(np.float64)
    c11 = img[y1, x1].astype(np.float64)
    return (c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy)
            + c01 * (1 - fx) * fy + c11 * fx * fy)
