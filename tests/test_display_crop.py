"""Regression checks for chin preservation in display crops."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
import processing


class DisplayCropTests(unittest.TestCase):
    def test_offset_face_is_preserved_and_only_recentered(self):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=90)
        mesh.translate([0, -35, 90])
        points = np.asarray(mesh.vertices).copy()
        self.assertTrue((np.linalg.norm(points, axis=1) > 110).any())
        with patch.dict(os.environ, {}, clear=True):
            center, radius, source = processing.display_crop(
                SimpleNamespace(face_landmarks=points))
        self.assertEqual(source, "facial_landmarks")
        np.testing.assert_allclose(center, [0, -35, 90])
        self.assertGreaterEqual(radius, 90 + processing.DISPLAY_FACE_MARGIN_MM)
        cropped, transform = processing.normalize_to_front_frame(
            mesh, np.eye(4), radius_mm=radius, crop_center=center)
        self.assertEqual(len(cropped.triangles), len(mesh.triangles))
        np.testing.assert_allclose(np.asarray(cropped.vertices),
                                   points + transform[:3, 3], atol=1e-6)
        np.testing.assert_array_equal(np.asarray(mesh.vertices), points)

    def test_old_small_override_cannot_cut_through_face(self):
        points = np.asarray(o3d.geometry.TriangleMesh.create_sphere(radius=130).vertices)
        with patch.dict(os.environ, {"VECTRA_DISPLAY_CROP_MM": "110"}), \
                patch.object(processing, "DISPLAY_CROP_RADIUS_MM", 110):
            center, radius, _ = processing.display_crop(
                SimpleNamespace(face_landmarks=points))
        self.assertTrue((np.linalg.norm(points - center, axis=1) < radius).all())
        self.assertGreaterEqual(radius, 155)

    def test_missing_or_invalid_landmarks_use_conservative_fallback(self):
        for points in (None, np.zeros((10, 3)), np.full((100, 3), np.nan)):
            with self.subTest(points_type=type(points)), \
                    patch.object(processing, "DISPLAY_CROP_RADIUS_MM", 110):
                center, radius, source = processing.display_crop(
                    SimpleNamespace(face_landmarks=points))
            self.assertEqual(source, "head_fallback")
            np.testing.assert_array_equal(center, np.zeros(3))
            self.assertGreaterEqual(radius, 160)


if __name__ == "__main__":
    unittest.main()
