"""Global pose self-calibration — the "global consistency step".

VGGT predicts every view's pose independently; nothing ever forces the 25
views to agree, so adjacent views disagree by a few millimetres and TSDF
faithfully fuses BOTH copies of anything seen by few views (the doubled ear,
the vertical crease seams). This module makes the views agree after the fact:

  1. per-view point clouds from the VGGT world-point maps
  2. pairwise point-to-plane ICP between every overlapping pair of views
  3. pose-graph optimization (Open3D multiway registration) -> one globally
     consistent correction per view — a miniature bundle adjustment over
     rigid poses, CPU-only
  4. corrections applied to the world points AND composed into the extrinsics,
     re-persisted to vggt_perview.npz / vggt_raw.npz so every downstream
     consumer (TSDF, texture bake, renders, metric) sees calibrated data
  5. a tightened cross-view filter afterwards — post-alignment, honest
     geometry agrees easily, so residual ghosts can be culled hard

Torch-free; runs in the finish stage. Idempotent (a `calibrated` flag in the
npz short-circuits reruns).
"""
from __future__ import annotations

import os

import numpy as np
import open3d as o3d

from .render_views import canonical_basis

VOXEL_DIV = 175          # ICP working voxel = scene diag / this (~2x TSDF voxel)
PAIR_MAX_YAW_DEG = 40.0  # register view pairs closer than this in camera yaw
MIN_FITNESS = 0.25       # discard ICP edges with less inlier support
MAX_TRANS_FRAC = 0.06    # guard: max correction, fraction of scene diag (~20mm)
MAX_ROT_DEG = 8.0
FILTER_REL_TOL = 0.005   # post-calibration cross-view agreement (0.5% depth)
FILTER_MIN_CONFIRM = 2   # neighbors that must confirm a point to keep it


def ensure_calibrated(out_dir: str) -> bool:
    """Calibrate the per-view maps under `out_dir` in place (once)."""
    pv_path = os.path.join(out_dir, "vggt_perview.npz")
    if not os.path.isfile(pv_path):
        return False
    pv = dict(np.load(pv_path, allow_pickle=True))
    if "valid" not in pv:
        return False                     # pre-valid-mask run; nothing to do
    if bool(np.asarray(pv.get("calibrated", False))):
        print("[calibrate] already calibrated", flush=True)
        return True

    wp = pv["world_points"].astype(np.float64)
    valid = pv["valid"].astype(bool)
    extr = pv["extrinsic"].astype(np.float64)          # (S,3,4) world->cam
    S = len(wp)

    pts_all = wp[valid]
    diag = float(np.linalg.norm(
        np.percentile(pts_all, 98, 0) - np.percentile(pts_all, 2, 0)))
    voxel = diag / VOXEL_DIV

    clouds = _per_view_clouds(wp, valid, voxel)
    order, yaws = _yaw_order(extr)
    corrections = _optimize(clouds, order, yaws, voxel, diag)

    # apply: points move by P; the camera must follow (E' = E @ P^-1)
    n_moved = 0
    mags = []
    for s in range(S):
        P = corrections[s]
        if np.allclose(P, np.eye(4), atol=1e-9):
            continue
        R, t = P[:3, :3], P[:3, 3]
        flat = wp[s].reshape(-1, 3)
        wp[s] = (flat @ R.T + t).reshape(wp[s].shape)
        E = np.vstack([extr[s], [0, 0, 0, 1]]) @ np.linalg.inv(P)
        extr[s] = E[:3]
        n_moved += 1
        mags.append(float(np.linalg.norm(t) / diag))
    if mags:
        print(f"[calibrate] corrected {n_moved}/{S} views, median shift "
              f"{np.median(mags) * 100:.2f}% of head size "
              f"(~{np.median(mags) * diag:.4f} units)", flush=True)
    else:
        print("[calibrate] no corrections applied", flush=True)

    valid &= _cross_view_filter(wp, valid, extr, pv["intrinsic"].astype(np.float64))

    pv["world_points"] = wp.astype(np.float32)
    pv["valid"] = valid
    pv["extrinsic"] = extr.astype(np.float64)
    pv["calibrated"] = np.asarray(True)
    np.savez(pv_path, **pv)

    # rebuild the fused cloud consumed by renders/fallback meshing
    raw_path = os.path.join(out_dir, "vggt_raw.npz")
    imgs = pv["images"].astype(np.float64) / 255.0
    np.savez(raw_path, points=wp[valid], colors=imgs[valid],
             extrinsic=pv["extrinsic"], intrinsic=pv["intrinsic"])
    print(f"[calibrate] persisted calibrated maps "
          f"({int(valid.sum())} points survive the tight filter)", flush=True)
    return True


def _per_view_clouds(wp, valid, voxel):
    clouds = []
    for s in range(len(wp)):
        pc = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(wp[s][valid[s]]))
        pc = pc.voxel_down_sample(voxel)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
        n = np.asarray(pc.normals)
        out = np.asarray(pc.points) - np.asarray(pc.points).mean(0)
        n[np.sum(n * out, 1) < 0] *= -1
        pc.normals = o3d.utility.Vector3dVector(n)
        clouds.append(pc)
    return clouds


def _yaw_order(extr):
    """Views sorted by camera yaw around the head, plus the yaw of each."""
    extr44 = [np.vstack([e, [0, 0, 0, 1]]) for e in extr]
    B = canonical_basis(extr44)
    yaws = []
    for e in extr44:
        c = -e[:3, :3].T @ e[:3, 3]                     # camera center, world
        f = B.T @ c                                     # face frame
        yaws.append(float(np.degrees(np.arctan2(f[0], f[2]))))
    return list(np.argsort(yaws)), yaws


def _pairwise_icp(src, dst, voxel):
    """Coarse-to-fine point-to-plane ICP. Returns (T, info, fitness)."""
    T = np.eye(4)
    for max_corr in (voxel * 4, voxel * 1.5):
        reg = o3d.pipelines.registration.registration_icp(
            src, dst, max_corr, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        T = reg.transformation
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src, dst, voxel * 1.5, T)
    return T, info, reg.fitness


def _optimize(clouds, order, yaws, voxel, diag):
    """Multiway registration (Open3D pose-graph). Returns a 4x4 correction per
    view (identity where the guard rejects or no edges support a view)."""
    n = len(order)
    graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.eye(4)
    graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_edges = 0
    for a in range(n):
        for b in range(a + 1, n):
            i, j = order[a], order[b]
            adjacent = b == a + 1
            if not adjacent and abs(yaws[i] - yaws[j]) > PAIR_MAX_YAW_DEG:
                continue
            T, info, fitness = _pairwise_icp(clouds[i], clouds[j], voxel)
            if fitness < MIN_FITNESS:
                if adjacent:                 # chain must stay connected
                    T, info = np.eye(4), np.eye(6)
                else:
                    continue
            if adjacent:
                odometry = T @ odometry
                graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(
                    np.linalg.inv(odometry)))
                graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                    a, b, T, info, uncertain=False))
            else:
                graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                    a, b, T, info, uncertain=True))
            n_edges += 1
    print(f"[calibrate] pose graph: {n} views, {n_edges} edges", flush=True)

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel * 1.5,
        edge_prune_threshold=0.25, reference_node=0)
    o3d.pipelines.registration.global_optimization(
        graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    # Gauge fix: the graph is anchored at order[0] — an extreme lateral view —
    # so the solution carries a common offset shared by every node (observed
    # ~8% of head size, tripping the guard on nearly all views). The signal is
    # DIFFERENTIAL misalignment, so re-express every pose relative to the
    # front-most view's, which then stays exactly fixed.
    a_front = int(np.argmin([abs(yaws[order[a]]) for a in range(n)]))
    P_ref_inv = np.linalg.inv(np.asarray(graph.nodes[a_front].pose))

    corrections = [np.eye(4)] * len(clouds)
    max_trans = MAX_TRANS_FRAC * diag
    for a in range(n):
        P = P_ref_inv @ np.asarray(graph.nodes[a].pose)
        i = order[a]
        t = float(np.linalg.norm(P[:3, 3]))
        rot = float(np.degrees(np.arccos(np.clip(
            (np.trace(P[:3, :3]) - 1) / 2, -1, 1))))
        if t > max_trans or rot > MAX_ROT_DEG:
            print(f"[calibrate] view {i}: correction rejected "
                  f"({t / diag:.3f} diag, {rot:.1f} deg)", flush=True)
            continue
        corrections[i] = P
    return corrections


def _cross_view_filter(wp, valid, extr, intrinsic,
                       neighbors: int = 2,
                       rel_tol: float = FILTER_REL_TOL,
                       min_confirm: int = FILTER_MIN_CONFIRM):
    """Post-calibration agreement filter (strict variant of the recon stage's
    filter): keep a point only if >= min_confirm neighboring views (or every
    neighbor that sees it, when fewer) put their surface within rel_tol of it."""
    S, H, W = valid.shape
    depths = np.full((S, H, W), np.nan, np.float32)
    for s in range(S):
        R, t = extr[s, :, :3], extr[s, :, 3]
        z = wp[s].reshape(-1, 3) @ R[2] + t[2]
        depths[s] = np.where(valid[s].reshape(-1), z, np.nan).reshape(H, W)
    tol = rel_tol * np.nanmedian(depths)
    keep = np.zeros((S, H, W), bool)
    for s in range(S):
        pts = wp[s][valid[s]]
        if not len(pts):
            continue
        confirm = np.zeros(len(pts), int)
        seen = np.zeros(len(pts), int)
        for nb in range(max(0, s - neighbors), min(S, s + neighbors + 1)):
            if nb == s:
                continue
            R, t = extr[nb, :, :3], extr[nb, :, 3]
            cam = pts @ R.T + t
            z = cam[:, 2]
            ok = z > 1e-9
            u = np.where(ok, cam[:, 0] / np.where(ok, z, 1)
                         * intrinsic[nb, 0, 0] + intrinsic[nb, 0, 2], -1)
            v = np.where(ok, cam[:, 1] / np.where(ok, z, 1)
                         * intrinsic[nb, 1, 1] + intrinsic[nb, 1, 2], -1)
            ui, vi = np.round(u).astype(int), np.round(v).astype(int)
            inb = ok & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            dn = np.full(len(pts), np.nan, np.float32)
            dn[inb] = depths[nb, vi[inb], ui[inb]]
            vis = np.isfinite(dn)
            seen += vis
            confirm += vis & (np.abs(dn - z) < tol)
        need = np.minimum(min_confirm, np.maximum(seen, 1))
        keep[s][valid[s]] = confirm >= need
    print(f"[calibrate] strict filter kept {int(keep.sum())}/"
          f"{int(valid.sum())} points", flush=True)
    return keep
