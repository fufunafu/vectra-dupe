"""Stage 5 - render the reconstructed head from canonical views (like VECTRA's
exports). Pure-numpy z-buffer splat renderer (no OpenGL); the mesh is oriented into
a canonical face frame using the recovered camera poses."""
from __future__ import annotations
import numpy as np
import open3d as o3d

# VECTRA-style canonical azimuths (deg around the vertical axis), 0 = front.
VIEW_AZIMUTHS = [-90, -55, -25, 0, 25, 55, 90]
ELEVATION = -3.0


def canonical_basis(extrinsics: list[np.ndarray]) -> np.ndarray:
    """Columns = [right, up, front] in world coords, from how the photos were shot.
    front points toward the cameras (the face front); up is the cameras' shared up."""
    ups, fwds = [], []
    for T in extrinsics:
        R = T[:3, :3]
        ups.append(-R[1, :])        # world up  = R^T @ (0,-1,0)
        fwds.append(-R[2, :])       # face normal toward cams = -view_dir
    up = np.mean(ups, 0); up /= np.linalg.norm(up)
    front = np.mean(fwds, 0); front -= up * (front @ up); front /= np.linalg.norm(front)
    right = np.cross(up, front); right /= np.linalg.norm(right)
    front = np.cross(right, up)
    return np.stack([right, up, front], 1)   # 3x3, columns are the axes


def _look_at(eye, target, up):
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    R = np.stack([s, u, -f], 0)
    return R, -R @ eye   # world->cam R, t


def _render_one(V, N, C, az, el, W=760, H=900, fov=38.0, gain=1.35):
    a, e = np.radians(az), np.radians(el)
    d = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    extent = np.percentile(np.linalg.norm(V, axis=1), 95) * 2
    eye = d * extent * 1.7
    R, t = _look_at(eye, np.zeros(3), np.array([0., 1., 0.]))

    Vc = V @ R.T + t                         # camera space (looks down -z)
    z = -Vc[:, 2]
    # backface culling: keep only points whose outward normal faces the camera, so
    # back-of-head points never bleed through the sparse front surface
    facing = (N @ d) > 0.05          # camera sits in the +d direction from origin
    valid = (z > 1e-4) & facing
    f = (H / 2) / np.tan(np.radians(fov) / 2)
    u = (Vc[:, 0] / np.clip(z, 1e-4, None)) * f + W / 2
    v = -(Vc[:, 1] / np.clip(z, 1e-4, None)) * f + H / 2
    px, py = np.round(u).astype(int), np.round(v).astype(int)

    # Lambertian shade: light at the camera (in the +d direction), so surfaces facing
    # the camera (N.d > 0) are lit. `gain` compensates VGGT's dim internal images
    # (1.35); photo-textured colors need none (1.0).
    shade = 0.45 + 0.55 * np.clip(N @ d, 0, 1)
    col = np.clip(C * gain * shade[:, None], 0, 1)

    # dark vertical-gradient background like VECTRA
    bg = np.linspace(0.18, 0.05, H)[:, None] * np.ones((1, W))
    img = np.repeat(bg[..., None], 3, axis=2)
    zbuf = np.full((H, W), np.inf)

    rad = 3                                   # splat radius (closes gaps between points)
    for dx in range(-rad, rad + 1):
        for dy in range(-rad, rad + 1):
            if dx * dx + dy * dy > rad * rad:
                continue
            xs, ys, zs, cs, ok = px + dx, py + dy, z, col, valid
            m = ok & (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            xs, ys, zs, cs = xs[m], ys[m], zs[m], cs[m]
            idx = ys * W + xs
            o = np.argsort(-zs)
            zb = zbuf.ravel(); ib = img.reshape(-1, 3)
            zb[idx[o]] = np.minimum(zb[idx[o]], zs[o])
            closer = zs[o] <= zb[idx[o]] + 1e-9
            ib[idx[o[closer]]] = cs[o[closer]]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def render_arrays(V: np.ndarray, N: np.ndarray, C: np.ndarray,
                  extrinsics: list[np.ndarray], out_dir: str,
                  gain: float = 1.35) -> list[str]:
    """Render points/normals/colors (world frame) from the canonical views."""
    import os, cv2
    B = canonical_basis(extrinsics)
    centroid = V.mean(0)
    Vc = (V - centroid) @ B
    Nc = N @ B
    paths = []
    for i, az in enumerate(VIEW_AZIMUTHS):
        img = _render_one(Vc, Nc, C, az, ELEVATION, gain=gain)
        p = os.path.join(out_dir, f"render_{i}_az{az:+d}.png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        paths.append(p)
    print(f"[render] wrote {len(paths)} canonical views", flush=True)
    return paths


def render(mesh_ply: str, extrinsics: list[np.ndarray], out_dir: str) -> list[str]:
    mesh = o3d.io.read_triangle_mesh(mesh_ply)
    mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    C = (np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors()
         else np.full((len(V), 3), 0.7))
    return render_arrays(V, N, C, extrinsics, out_dir)


def render_point_cloud(points: np.ndarray, colors: np.ndarray,
                       extrinsics: list[np.ndarray], out_dir: str,
                       skin_only: bool = True, lum_thresh: float = 0.17) -> list[str]:
    """Robust path: render the dense colored point cloud directly (no meshing).
    skin_only drops near-black points (hair is intrinsically noisy in photogrammetry
    and overlaps the face); the bright skin surface reconstructs cleanly."""
    if skin_only:
        lum = colors @ np.array([0.299, 0.587, 0.114])   # colors are RGB in [0,1]
        keep = lum > lum_thresh
        points, colors = points[keep], colors[keep]
        print(f"[render] skin filter kept {keep.sum()}/{len(keep)} points", flush=True)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    ext = np.percentile(points, 98, 0) - np.percentile(points, 2, 0)
    voxel = float(np.linalg.norm(ext) / 600.0)
    pc = pc.voxel_down_sample(voxel)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    # keep the largest spatial cluster (the head), drop floating hair/speckle blobs
    labels = np.asarray(pc.cluster_dbscan(eps=voxel * 4, min_points=10))
    if labels.max() >= 0:
        biggest = np.bincount(labels[labels >= 0]).argmax()
        pc = pc.select_by_index(np.where(labels == biggest)[0])
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
    V = np.asarray(pc.points)
    N = np.asarray(pc.normals)
    C = np.asarray(pc.colors)
    # cheap consistent orientation: normals point outward from the cloud centroid
    out = V - V.mean(0)
    flip = np.sum(N * out, axis=1) < 0
    N[flip] *= -1
    print(f"[render] point cloud: {len(V)} splat points", flush=True)
    return render_arrays(V, N, C, extrinsics, out_dir)
