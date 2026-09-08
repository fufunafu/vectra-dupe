"""Opt-in photo-surface comparison, separate from the depth measurement path.

Landmarks are redetected on a close-up textured render, not reused from a small
face in a whole-scene render. Inner canthi establish common scale; upper-face
landmarks and forehead surfaces establish rigid alignment. No deformation or
bias subtraction is allowed. A directly measured intercanthal distance is
required before reporting any exploratory volume in cc.
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from matplotlib.path import Path as Polygon

from . import analyze, photogrammetry as pg, quality

LANDMARK_VERSION = 1
INNER_CANTHI = (133, 362)
STABLE_LANDMARKS = np.array([33, 133, 362, 263, 168, 6, 9, 151, 10,
                             107, 336, 105, 334, 66, 296])
FACE_OVAL = np.array([10,338,297,332,284,251,389,356,454,323,361,288,
                     397,365,379,378,400,377,152,148,176,149,150,136,
                     172,58,132,93,234,127,162,21,54,103,67,109])


def intercanthal_distance(points):
    distance = float(np.linalg.norm(points[INNER_CANTHI[0]] - points[INNER_CANTHI[1]]))
    if not np.isfinite(distance) or distance <= 0:
        raise quality.QualityError("Inner eye corners are missing or coincident.")
    return distance


def rigid_fit(source, target):
    """Rigid landmark fit, with an actual RMS residual and no fitted scale."""
    source, target = np.asarray(source), np.asarray(target)
    if (source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3
            or len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all()
            or np.linalg.matrix_rank(source - source.mean(0)) < 2):
        raise quality.QualityError("Stable facial landmarks cannot determine a rigid alignment.")
    s, t = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((target-t).T @ (source-s))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, t-rotation@s
    residual = source@rotation.T + transform[:3, 3] - target
    return transform, float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def closeup_landmarks(asset_path, work):
    """Return uncropped geometry and indexed 3D landmarks; cache by asset hash."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    fingerprint = quality.mesh_fingerprint(str(asset_path))
    with np.load(asset_path, allow_pickle=False) as data:
        v, f, uv, albedo, seed = [data[key] for key in
                                  ("vertices", "triangles", "texture_uvs", "albedo", "face_landmarks")]
    if (seed.ndim != 2 or seed.shape[1] != 3 or len(seed) < 100 or not np.isfinite(seed).all()
            or v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1] != 3 or not len(f)
            or f.min() < 0 or f.max() >= len(v)):
        raise quality.QualityError("Photo source lacks valid geometry and facial bounds.")
    cache = work / "landmarks.npz"
    points = None
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            if (str(saved["source_sha256"]) == fingerprint
                    and int(saved["version"]) == LANDMARK_VERSION):
                points = saved["points"]
    if points is None:
        center = (seed.min(0) + seed.max(0)) / 2
        extent = float(np.ptp(seed, axis=0).max()) * 1.15
        if extent <= 0:
            raise quality.QualityError("Photo face bounds are invalid.")
        normalized = (v-center)/extent
        size, camera_z = 1536, 2.2
        image, _, _ = pg._render_textured(normalized, f, uv, albedo, size=size, cz=camera_z)
        render_path = work / "landmark_closeup.png"
        if not o3d.io.write_image(str(render_path), o3d.geometry.Image(np.ascontiguousarray(image))):
            raise RuntimeError("Could not save the close-up landmark render")
        detected = pg._detect_landmarks(str(render_path))
        if not detected or len(detected.get("landmarks", [])) != 478:
            raise quality.QualityError("Inner eye corners could not be located on the photo model.")
        pixels = np.asarray(detected["landmarks"])
        directions = np.c_[(pixels[:,0]-size/2)/(1.05*size),
                            -(pixels[:,1]-size/2)/(1.05*size), -np.ones(len(pixels))]
        directions /= np.linalg.norm(directions, axis=1)[:,None]
        origins = np.tile([0, 0, camera_z], (len(pixels),1))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(normalized.astype(np.float32)), o3d.core.Tensor(f.astype(np.uint32))))
        distances = scene.cast_rays(o3d.core.Tensor(
            np.c_[origins,directions].astype(np.float32)))["t_hit"].numpy()
        points = (origins + directions*distances[:,None])*extent + center
        if not np.isfinite(points).all():
            raise quality.QualityError("A close-up landmark has no corresponding photo surface.")
        np.savez_compressed(cache, points=points, source_sha256=fingerprint, version=LANDMARK_VERSION)
    if points.shape != (478, 3) or not np.isfinite(points).all():
        raise quality.QualityError("Invalid cached photo landmarks.")
    intercanthal_distance(points)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    center = (points.min(0)+points.max(0))/2
    radius = float(np.linalg.norm(points-center,axis=1).max())+10
    mesh.remove_vertices_by_mask(np.linalg.norm(v-center,axis=1)>radius)
    mesh.compute_vertex_normals()
    return mesh, points, fingerprint


def forehead_mask(vertices, landmarks, unit):
    eye = landmarks[list(INNER_CANTHI)].mean(0)
    return ((vertices[:,1] > eye[1]+10*unit) & (vertices[:,1] < eye[1]+55*unit)
            & (np.abs(vertices[:,0]-eye[0]) < 55*unit) & (vertices[:,2] > eye[2]-15*unit))


def surface_alignment(before, after, landmarks, unit):
    """Bidirectional point-to-surface error, not tessellation-dependent vertex error."""
    reports = []
    for source, target in ((before,after),(after,before)):
        vertices = np.asarray(source.vertices)
        selected = vertices[forehead_mask(vertices,landmarks,unit)]
        if len(selected) < 30:
            raise quality.QualityError("Not enough unchanged forehead surface for alignment checks.")
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(target))
        d = scene.compute_distance(o3d.core.Tensor(selected.astype(np.float32))).numpy()/unit
        reports.append({"samples":len(d), "rms_reference_mm":float(np.sqrt(np.mean(d*d))),
                        "within_2_reference_mm":float(np.mean(d<2))})
    return {"directions":reports, "eligible":all(
        r["rms_reference_mm"] <= quality.MAX_ALIGNMENT_RMSE_MM
        and r["within_2_reference_mm"] >= quality.MIN_ALIGNMENT_FITNESS for r in reports)}


def align_surfaces(before, after, before_points, after_points, measured_icd_mm=None):
    """Preserve originals; scale only by inner canthi, then align the upper face."""
    bd, ad = intercanthal_distance(before_points), intercanthal_distance(after_points)
    if measured_icd_mm is not None and (not np.isfinite(measured_icd_mm) or measured_icd_mm <= 0):
        raise quality.QualityError("Measured intercanthal distance must be a positive number in mm.")
    target = bd if measured_icd_mm is None else measured_icd_mm
    unit = target/bd
    b, a = o3d.geometry.TriangleMesh(before), o3d.geometry.TriangleMesh(after)
    b.scale(unit, center=np.zeros(3)); a.scale(target/ad, center=np.zeros(3))
    bl, al = before_points*unit, after_points*(target/ad)
    transform, rms = rigid_fit(al[STABLE_LANDMARKS], bl[STABLE_LANDMARKS])
    if rms/unit > 2:
        raise quality.QualityError("Upper-face landmarks disagree after inner-eye scale normalization.")
    a.transform(transform)
    initial = surface_alignment(b,a,bl,unit)
    clouds=[]
    for mesh in (b,a):
        vertices=np.asarray(mesh.vertices);mask=forehead_mask(vertices,bl,unit)
        cloud=o3d.geometry.PointCloud()
        cloud.points=o3d.utility.Vector3dVector(vertices[mask])
        cloud.normals=o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals)[mask])
        clouds.append(cloud)
    refinement=o3d.pipelines.registration.registration_icp(
        clouds[1],clouds[0],3*unit,np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)).transformation
    probe=np.c_[np.asarray(clouds[1].points),np.ones(len(clouds[1].points))]
    motion=float(np.max(np.linalg.norm((probe@refinement.T-probe)[:,:3],axis=1)))/unit
    rotation=float(np.degrees(np.arccos(np.clip((np.trace(refinement[:3,:3])-1)/2,-1,1))))
    accepted=False
    if motion<=3 and rotation<=3:
        candidate=o3d.geometry.TriangleMesh(a).transform(refinement)
        revised=surface_alignment(b,candidate,bl,unit)
        if sum(r['rms_reference_mm'] for r in revised['directions']) < sum(r['rms_reference_mm'] for r in initial['directions']):
            a=candidate;transform=refinement@transform;initial=revised;accepted=True
    if not initial['eligible']:
        raise quality.QualityError("Unchanged upper-face surfaces do not align sufficiently for photo comparison.")
    calibration={"measured_intercanthal_mm":measured_icd_mm,
                 "source":"user_measurement" if measured_icd_mm is not None else "uncalibrated_reference",
                 "before_detected_distance":bd,"after_detected_distance":ad,
                 "before_scale":unit,"after_scale":target/ad,
                 "relative_after_scale":bd/ad}
    checks={"alignment":initial,"landmark_rms_reference_mm":rms/unit,
            "forehead_refinement_applied":accepted,"refinement_max_motion_reference_mm":motion}
    return b,a,bl,transform,calibration,checks


def save_display_texture(asset_path, display, scale, out_dir):
    """Transfer the before photo atlas to the exact heatmap surface, preserving seams.

    Every subdivided triangle lies within one source triangle. Its center selects
    that triangle, then barycentric coordinates transfer the three corner UVs.
    This changes presentation only, not geometry or measured surface distances.
    """
    with np.load(asset_path, allow_pickle=False) as data:
        vertices = data['vertices'] * scale
        faces, uv, albedo = data['triangles'], data['texture_uvs'], data['albedo']
    if (uv.shape != (len(faces), 3, 2) or not np.isfinite(uv).all()
            or albedo.ndim != 3 or albedo.shape[2] != 3 or albedo.dtype != np.uint8):
        raise ValueError('Photo atlas or UV coordinates are invalid')
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(faces.astype(np.uint32))))
    target = np.asarray(display.vertices)[np.asarray(display.triangles)]
    centers = target.mean(axis=1)
    hit = scene.compute_closest_points(o3d.core.Tensor(centers.astype(np.float32)))
    source_ids = hit['primitive_ids'].numpy()
    source = vertices[faces[source_ids]]
    a, b = source[:, 1] - source[:, 0], source[:, 2] - source[:, 0]
    p = target - source[:, None, 0]
    aa, ab, bb = (a*a).sum(1), (a*b).sum(1), (b*b).sum(1)
    pa, pb = (p*a[:, None]).sum(2), (p*b[:, None]).sum(2)
    determinant = aa*bb - ab*ab
    if np.any(determinant <= 1e-16):
        raise ValueError('Degenerate source triangles cannot map the face texture')
    u = (bb[:, None]*pa - ab[:, None]*pb) / determinant[:, None]
    v = (aa[:, None]*pb - ab[:, None]*pa) / determinant[:, None]
    weights = np.stack([1-u-v, u, v], axis=2)
    reconstructed = np.einsum('tck,tkd->tcd', weights, source)
    if (np.max(np.linalg.norm(reconstructed-target, axis=2)) > 1e-4
            or weights.min() < -1e-3 or weights.max() > 1.001):
        raise ValueError('Heatmap triangles do not match the before photo surface')
    mapped_uv = np.einsum('tck,tkd->tcd', weights, uv[source_ids]).astype('<f4')
    out = Path(out_dir)
    uv_path, image_path = out/'heatmap.uv.bin', out/'heatmap.texture.png'
    mapped_uv.tofile(uv_path)
    if not o3d.io.write_image(str(image_path), o3d.geometry.Image(np.ascontiguousarray(albedo))):
        raise RuntimeError('Could not save the heatmap face texture')
    return {'uv_file': uv_path.name, 'image_file': image_path.name,
            'uv_sha256': quality.mesh_fingerprint(str(uv_path)),
            'image_sha256': quality.mesh_fingerprint(str(image_path)),
            'corner_count': int(mapped_uv.shape[0]*3), 'source': 'before'}


def compare_assets(before_path, after_path, out_dir, measured_icd_mm=None):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    before,bl,bhash=closeup_landmarks(before_path,out/'before')
    after,al,ahash=closeup_landmarks(after_path,out/'after')
    before,after,landmarks,transform,calibration,checks=align_surfaces(before,after,bl,al,measured_icd_mm)
    # Midpoint subdivision adds samples without smoothing or changing the surface.
    display=before.subdivide_midpoint(number_of_iterations=2)
    field=analyze.signed_distance_field(display,after)
    polygon=Polygon(landmarks[FACE_OVAL,:2])
    inside=polygon.contains_points(field.vertices[:,:2])
    field.distances[~inside]=np.nan
    warnings=["Photo-based research comparison, not clinically validated.",
              "Uses photo geometry, not the depth measurement mesh. Depth quality warnings are not repaired.",
              "Visible differences can include expression, reconstruction, and landmark-placement errors."]
    if measured_icd_mm is None:
        warnings.append("Absolute scale is unverified. Enter a directly measured intercanthal distance for exploratory cc estimates.")
    summary={"source":"photo","experimental":True,"diagnostic_only":True,
             "regions":[],"net_significant_volume_ml":None,"noise_floor_ml":None,
             "calibration":calibration,"transform":transform.tolist(),
             "quality_checks":{"version":quality.QUALITY_VERSION,**checks,"warnings":warnings},
             "input_asset_sha256":{"before":bhash,"after":ahash}}
    # Do not use synthetic noise-floor thresholds or bias subtraction on photos.
    regions = projected_region_volumes(before,after,landmarks)
    summary["regional_coverage"] = [{"label":r["label"],"coverage":r["coverage"]} for r in regions]
    if any(r["coverage"] < 1 for r in regions):
        warnings.append("Some lower-face patches have incomplete or ambiguous surface coverage; their full-patch volumes are withheld.")
    if measured_icd_mm is not None:
        summary["regional_volumes"] = regions
        summary["diagnostic_only"] = not all(r["volume_ml"] is not None for r in summary["regional_volumes"])
    if [bhash,ahash] != [quality.mesh_fingerprint(str(p)) for p in (before_path,after_path)]:
        raise quality.QualityError("A photo source changed during comparison. Retry after processing finishes.")
    texture = None
    try:
        texture = save_display_texture(before_path, display, calibration['before_scale'], out)
    except (OSError, ValueError, RuntimeError) as exc:
        warnings.append(f'Face texture overlay unavailable: {exc}')
    analyze.save_colored_mesh(display,field,str(out/'heatmap.ply'), texture=texture)
    analyze.save_heatmap_png(field,str(out/'heatmap.png'),
        "PHOTO COMPARISON: inner-eye scale, upper-face alignment\nResearch only; not a validated treatment-volume measurement",
        colorbar_label="Surface difference (mm)" if measured_icd_mm else "Surface difference (approximate reference mm)")
    (out/'result.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary


def projected_region_volumes(before,after,landmarks):
    """Signed volumes over fixed frontal lower-face patches, with no hole filling.

    Integrate z_after-z_before over projected xy area. These are lower-face
    patch differences, not the amount of gauze, filler, or treatment administered.
    """
    from scipy.ndimage import binary_erosion
    step=0.75
    oval=landmarks[FACE_OVAL,:2]
    x,y=np.meshgrid(np.arange(oval[:,0].min(),oval[:,0].max(),step),
                     np.arange(oval[:,1].min(),oval[:,1].max(),step))
    domain=Polygon(oval).contains_points(np.c_[x.ravel(),y.ravel()]).reshape(x.shape)
    domain=binary_erosion(domain,iterations=4)
    eye=landmarks[list(INNER_CANTHI)].mean(0)
    domain &= y < eye[1]-15
    surfaces=[]
    for mesh in (before,after):
        scene=o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        front=float(np.asarray(mesh.vertices)[:,2].max()+10)
        rays=np.c_[x.ravel(),y.ravel(),np.full(x.size,front),np.zeros(x.size),np.zeros(x.size),-np.ones(x.size)]
        hits=scene.list_intersections(o3d.core.Tensor(rays.astype(np.float32)))
        ids,dist=hits['ray_ids'].numpy(),hits['t_hit'].numpy()
        mesh.compute_triangle_normals()
        facing=np.asarray(mesh.triangle_normals)[hits['primitive_ids'].numpy(),2]
        keep=np.isfinite(dist)&(dist>0)&(front-dist>eye[2]-40)
        ids,dist,facing=ids[keep],dist[keep],facing[keep]
        order=np.lexsort((dist,ids));ids,dist,facing=ids[order],dist[order],facing[order]
        z=np.full(x.size,np.nan)
        if len(ids):
            distinct=np.r_[True,(ids[1:]!=ids[:-1])|(np.diff(dist)>0.5)]
            ids,dist,facing=ids[distinct],dist[distinct],facing[distinct]
            # A closed surface normally has an entry AND an exit. Do not call
            # its back-facing exit a duplicate layer. Multiple front entries
            # before the first exit still flag overlapping sheets.
            exits=np.full(x.size,np.inf)
            np.minimum.at(exits,ids[facing<0],dist[facing<0])
            entries=(facing>0)&(dist<exits[ids])
            counts=np.bincount(ids[entries],minlength=x.size)
            single=entries&(counts[ids]==1)
            z[ids[single]]=front-dist[single]
        surfaces.append(z.reshape(x.shape))
    delta=surfaces[1]-surfaces[0]
    regions=[]
    for name,side in (("Subject's right lower face",x<eye[0]),("Subject's left lower face",x>=eye[0])):
        roi=domain&side;valid=roi&np.isfinite(delta)
        count=int(roi.sum());coverage=float(valid.sum()/count) if count else 0.
        # Full coverage is required to call this a patch volume. Missing data is
        # never silently treated as zero, even if it is only a small fraction.
        volume=float(delta[valid].sum()*step**2/1000) if count and valid.sum()==count else None
        regions.append({"label":name,"volume_ml":round(volume,3) if volume is not None else None,
                        "coverage":coverage,"projected_area_mm2":count*step**2,
                        "warning":"Not a validated treatment volume." if volume is not None
                        else "Missing or layered surface in this patch; volume withheld."})
    return regions
