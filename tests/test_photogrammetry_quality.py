import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from vectra3d import photogrammetry as pg


class LandmarkSearchTests(unittest.TestCase):
    def test_all_up_axes_form_24_unique_proper_rotations(self):
        rotations = [r for up in range(3) for r in pg._landmark_view_rotations(np.eye(3), up)]
        self.assertEqual(len(rotations), 24)
        self.assertEqual(len({tuple(r.ravel()) for r in rotations}), 24)
        for rotation in rotations:
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3))
            self.assertAlmostEqual(np.linalg.det(rotation), 1)

    def test_whole_scene_detection_failure_uses_high_resolution_tiled_search(self):
        vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-2, 2) for z in (-3, 3)])
        calls = []

        def detect(path, *, search=False):
            calls.append(search)
            return {"landmarks": [[1, 1]] * 478} if search else None

        rendered = (np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3)),
                    np.ones((4, 4), bool))
        diagnostics = {}
        with patch.object(pg, "_detect_landmarks", side_effect=detect), \
                patch.object(pg, "_render_textured", return_value=rendered) as render, \
                patch.object(pg.o3d.io, "write_image", return_value=True):
            result = pg._oc_landmarks(vertices, None, None, None, "/unused",
                                      diagnostics=diagnostics)
        self.assertIsNotNone(result)
        self.assertEqual(int(result[1].sum()), 478)
        self.assertEqual(calls, [False] * 8 + [True] * 8)
        self.assertEqual(diagnostics["landmark_render_size"], 1024)
        self.assertEqual(render.call_args.kwargs["size"], 1024)

    def test_detector_crash_is_not_misreported_as_no_face(self):
        failure = subprocess.CompletedProcess([], 1, "", "model could not initialize")
        with patch.object(pg.subprocess, "run", return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "model could not initialize"):
                pg._detect_landmarks("unused.png")
        no_face = subprocess.CompletedProcess([], 0, '{"ok": false}', "")
        with patch.object(pg.subprocess, "run", return_value=no_face):
            self.assertIsNone(pg._detect_landmarks("unused.png"))


if __name__ == "__main__":
    unittest.main()
