"""Run in VECTRA-SOFTWARE-ONLY/.venv-mp; checks crop-to-original pixel mapping."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

script = Path(__file__).resolve().parents[1] / "VECTRA-SOFTWARE-ONLY/tools/landmark_detect.py"
spec = importlib.util.spec_from_file_location("landmark_detect", script)
detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector)


class CropTests(unittest.TestCase):
    def test_windows_stay_in_bounds_and_cover_edges(self):
        windows = list(detector.search_windows(1024, 768))
        self.assertEqual(windows[0], (0, 0, 1024, 768))
        for x, y, width, height in windows:
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + width, 1024)
            self.assertLessEqual(y + height, 768)

    def test_landmarks_and_iris_centres_return_in_original_image_pixels(self):
        class FakeDetector:
            calls = 0

            def detect(self, image):
                self.calls += 1
                # Full image, first tile, then tile offset 128 pixels to right.
                points = [SimpleNamespace(x=0.5, y=0.5)] * 478
                return SimpleNamespace(face_landmarks=[points] if self.calls == 3 else [])

        output = io.StringIO()
        with patch.object(detector.cv2, "imread", return_value=np.zeros((512, 512, 3), np.uint8)), \
                patch.object(detector.vision.FaceLandmarker, "create_from_options",
                             return_value=contextlib.nullcontext(FakeDetector())), \
                contextlib.redirect_stdout(output):
            detector.main("unused", search=True)
        result = json.loads(output.getvalue())
        self.assertEqual(result["landmarks"][0], [256, 128])
        self.assertEqual(result["left_iris"], [256, 128])
        self.assertEqual((result["width"], result["height"]), (512, 512))


if __name__ == "__main__":
    unittest.main()
