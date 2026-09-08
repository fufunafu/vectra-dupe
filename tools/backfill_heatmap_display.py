"""Add sensitivity samples to a saved photo result without changing its analysis.

Uses the saved scale and transform, never a new alignment. Verifies both source
hashes and exact displayed geometry before publishing the new sidecar.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from matplotlib.path import Path as Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vectra3d import analyze, photo_compare as pc, quality


def backfill(result_dir):
    out = Path(result_dir).resolve()
    summary = json.loads((out / 'result.json').read_text())
    if summary.get('source') != 'photo':
        raise ValueError('Only saved photo comparisons are supported')
    before_id, after_id, suffix = out.name.split('__')
    if suffix != 'photo':
        raise ValueError('Unexpected comparison directory')
    session_root = out.parent.parent / 'sessions'
    meshes, points = [], []
    for role, sid in [('before', before_id), ('after', after_id)]:
        asset = session_root / sid / 'display_uncropped.npz'
        if quality.mesh_fingerprint(str(asset)) != summary['input_asset_sha256'][role]:
            raise ValueError(f'{role} photo asset changed; rerun comparison')
        mesh, landmarks, _ = pc.closeup_landmarks(asset, out / role)
        scale = summary['calibration'][f'{role}_scale']
        mesh.scale(scale, center=np.zeros(3))
        meshes.append(mesh)
        points.append(landmarks * scale)
    meshes[1].transform(np.asarray(summary['transform']))
    display = meshes[0].subdivide_midpoint(number_of_iterations=2)
    path = out / 'heatmap.ply'
    saved = o3d.io.read_triangle_mesh(str(path))
    np.testing.assert_array_equal(saved.triangles, display.triangles)
    np.testing.assert_allclose(saved.vertices, display.vertices, atol=1e-8, rtol=0)
    field = analyze.signed_distance_field(display, meshes[1])
    inside = Polygon(points[0][pc.FACE_OVAL, :2]).contains_points(field.vertices[:, :2])
    field.distances[~inside] = np.nan
    texture = pc.save_display_texture(session_root / before_id / 'display_uncropped.npz',
                                     display, summary['calibration']['before_scale'], out)
    analyze.save_heatmap_data(field, str(path), texture=texture)
    print(f'Added sensitivity samples: {path.with_suffix(".display.json")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    backfill(parser.parse_args().result_dir)
