import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from vectra3d import analyze, photo_compare


class HeatmapDisplayTests(unittest.TestCase):
    def test_export_keeps_unclipped_distances_and_missing_samples(self):
        mesh = o3d.geometry.TriangleMesh.create_box()
        distances = np.array([-8., -2.5, -1., 0., 1., 2.5, 8., np.nan])
        original = distances.copy()
        field = analyze.DistanceField(np.asarray(mesh.vertices), distances, np.ones(8))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'heatmap.ply'
            analyze.save_colored_mesh(mesh, field, str(path))
            data = json.loads(path.with_suffix('.display.json').read_text())
            self.assertEqual(data['mesh_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(data['distances'], [-8., -2.5, -1., 0., 1., 2.5, 8., None])
            self.assertEqual(len(data['palette']), 256)
            self.assertEqual(data['default_range'], 2.5)
            reloaded = o3d.io.read_triangle_mesh(str(path))
            np.testing.assert_array_equal(reloaded.vertices, mesh.vertices)
        np.testing.assert_array_equal(field.distances, original)
        self.assertFalse(mesh.has_vertex_colors())

    def test_photo_uv_transfer_preserves_seams_and_original_pixels(self):
        vertices = np.array([[0., 0, 0], [1., 0, 0], [1., 1, 0], [0., 1, 0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        uv = vertices[faces, :2].copy()
        uv[1] = uv[1] * 0.5 + 0.5
        albedo = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                       o3d.utility.Vector3iVector(faces))
        display = mesh.subdivide_midpoint(2).scale(2.5, center=np.zeros(3))
        with tempfile.TemporaryDirectory() as folder:
            asset = Path(folder) / 'photo.npz'
            np.savez(asset, vertices=vertices, triangles=faces, texture_uvs=uv, albedo=albedo)
            metadata = photo_compare.save_display_texture(asset, display, 2.5, folder)
            actual = np.fromfile(Path(folder)/metadata['uv_file'], dtype='<f4').reshape(-1, 3, 2)
            corners = np.asarray(display.vertices)[np.asarray(display.triangles)] / 2.5
            centers = corners.mean(1)
            expected = corners[:, :, :2].copy()
            second = centers[:, 1] > centers[:, 0]
            expected[second] = expected[second] * 0.5 + 0.5
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            np.testing.assert_array_equal(np.asarray(o3d.io.read_image(str(Path(folder)/metadata['image_file']))), albedo)
            self.assertEqual(metadata['corner_count'], len(corners)*3)
            self.assertEqual(metadata['source'], 'before')


if __name__ == '__main__':
    unittest.main()
