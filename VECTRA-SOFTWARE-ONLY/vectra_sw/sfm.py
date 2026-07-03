"""Stage 2 - classical Structure-from-Motion with COLMAP (pycolmap).

Recovers camera intrinsics, per-image poses (world->camera), and a sparse 3D point
cloud. This is the photogrammetry core, the same principle VECTRA's software uses.
"""
from __future__ import annotations
import os, shutil, glob
import cv2
import numpy as np
import pycolmap

SENSOR_WIDTH_MM = 22.3   # Canon EOS Rebel T2i (550D) APS-C sensor width


def focal_px_from_exif(orig_photo: str, work_width: int, focal_mm: float = 85.0) -> float:
    """Pixel focal length for the working resolution, from EXIF focal + sensor width."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        ex = {TAGS.get(k, k): v for k, v in (Image.open(orig_photo)._getexif() or {}).items()}
        focal_mm = float(ex.get("FocalLength", focal_mm))
    except Exception:
        pass
    return focal_mm * work_width / SENSOR_WIDTH_MM


def _sift_options():
    """Hard-surface recipe: many features, affine-shape + domain-size-pooling, low
    peak threshold -> far more, more repeatable keypoints on specular low-texture skin."""
    o = pycolmap.SiftExtractionOptions()
    o.max_num_features = 16384
    o.estimate_affine_shape = True
    o.domain_size_pooling = True
    o.peak_threshold = 0.004          # default 0.0066 -> more keypoints
    o.edge_threshold = 16.0           # default 10 -> keep edge-like skin features
    o.max_num_orientations = 3
    return o


def _mapper_options():
    """Relax registration thresholds so weakly-matched face views still register."""
    opt = pycolmap.IncrementalPipelineOptions()
    opt.min_num_matches = 8           # default 15
    opt.mapper.init_min_num_inliers = 30      # default 100
    opt.mapper.abs_pose_min_num_inliers = 12  # default 30
    opt.mapper.abs_pose_min_inlier_ratio = 0.1  # default 0.25
    opt.mapper.filter_max_reproj_error = 6.0
    opt.mapper.init_max_error = 6.0
    opt.min_model_size = 5
    # intrinsics are known from EXIF; keep them fixed (low parallax can't self-calibrate)
    opt.ba_refine_focal_length = False
    opt.ba_refine_principal_point = False
    opt.ba_refine_extra_params = False
    return opt


def run(work_dir: str, out_dir: str, mask_dir: str | None = None,
        focal_px: float | None = None) -> dict:
    db_path = os.path.join(out_dir, "colmap.db")
    sparse_dir = os.path.join(out_dir, "sparse")
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(sparse_dir):
        shutil.rmtree(sparse_dir)
    os.makedirs(sparse_dir, exist_ok=True)

    reader = pycolmap.ImageReaderOptions()
    if mask_dir:
        reader.mask_path = mask_dir   # features restricted to the rigid face only

    camera_mode = pycolmap.CameraMode.AUTO
    if focal_px:
        # one shared, calibrated pinhole camera (all photos: same lens, same body)
        sample = sorted(glob.glob(os.path.join(work_dir, "*.JPG")) +
                        glob.glob(os.path.join(work_dir, "*.jpg")))[0]
        h, w = cv2.imread(sample).shape[:2]
        reader.camera_model = "SIMPLE_PINHOLE"
        reader.camera_params = f"{focal_px},{w/2},{h/2}"
        camera_mode = pycolmap.CameraMode.SINGLE
        print(f"[sfm] fixed camera: SIMPLE_PINHOLE f={focal_px:.1f}px @ {w}x{h}")

    fe = pycolmap.FeatureExtractionOptions()
    fe.sift = _sift_options()

    print("[sfm] extracting SIFT features (affine-shape + DSP, masked to face) ...")
    pycolmap.extract_features(db_path, work_dir, reader_options=reader,
                              extraction_options=fe, camera_mode=camera_mode)

    print("[sfm] exhaustive matching ...")
    mo = pycolmap.FeatureMatchingOptions()
    mo.sift.max_ratio = 0.9           # default 0.8 -> keep more putative matches
    pycolmap.match_exhaustive(db_path, matching_options=mo)

    print("[sfm] incremental mapping ...")
    recs = pycolmap.incremental_mapping(db_path, work_dir, sparse_dir,
                                        options=_mapper_options())
    if not recs:
        raise RuntimeError("SfM failed: no reconstruction produced")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    print(f"[sfm] registered {rec.num_reg_images()}/{len(rec.images)} images, "
          f"{rec.num_points3D()} sparse points, "
          f"mean reproj err {rec.compute_mean_reprojection_error():.3f}px")

    rec.write(sparse_dir)
    rec.export_PLY(os.path.join(out_dir, "sparse.ply"))
    return {"reconstruction_dir": sparse_dir, "db": db_path}


def load(sparse_dir: str):
    return pycolmap.Reconstruction(sparse_dir)


def camera_views(rec) -> list[dict]:
    """Per registered image: name, K (3x3), extrinsic world->cam (4x4), W, H, and
    the sparse observations as (pixel_uv array, camera-space depth array)."""
    views = []
    for img in rec.images.values():
        if not img.has_pose:
            continue
        cam = rec.cameras[img.camera_id]
        K = np.asarray(cam.calibration_matrix(), float)
        T = np.eye(4)
        T[:3, :4] = np.asarray(img.cam_from_world().matrix(), float)  # [R|t], world->cam
        R, t = T[:3, :3], T[:3, 3]

        uv, depth = [], []
        for p2d in img.points2D:
            if not p2d.has_point3D():
                continue
            Xw = np.asarray(rec.points3D[p2d.point3D_id].xyz, float)
            Xc = R @ Xw + t
            if Xc[2] <= 0:
                continue
            uv.append(np.asarray(p2d.xy, float))
            depth.append(Xc[2])
        views.append({
            "name": img.name,
            "K": K, "extrinsic": T,
            "width": int(cam.width), "height": int(cam.height),
            "uv": np.asarray(uv).reshape(-1, 2),
            "depth": np.asarray(depth).reshape(-1),
        })
    return views


if __name__ == "__main__":
    import sys
    run(sys.argv[1], sys.argv[2])
