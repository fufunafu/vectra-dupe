"""Standalone facial-landmark detector (run with the isolated .venv-mp interpreter,
which has mediapipe 0.10 + numpy 2). Outputs pixel-space landmarks as JSON to stdout.

    .venv-mp/bin/python tools/landmark_detect.py <image_path>

Uses the MediaPipe Tasks FaceLandmarker (model: tools/face_landmarker.task), which
returns 478 landmarks including iris centers (468 = left iris, 473 = right iris).
The CLI searches overlapping crops if whole-image detection fails. Pass
--whole-image-only for the fast first pass used by the reconstruction pipeline.
"""
import sys, json, os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def search_windows(width, height):
    """Overlapping crops for a small face inside a full-scene render."""
    yield 0, 0, width, height
    for fraction in (0.5, 0.25):
        side = max(64, round(min(width, height) * fraction))
        xs = sorted(set(list(range(0, max(1, width - side + 1), max(1, side // 2)))
                        + [max(0, width - side)]))
        ys = sorted(set(list(range(0, max(1, height - side + 1), max(1, side // 2)))
                        + [max(0, height - side)]))
        for y in ys:
            for x in xs:
                yield x, y, min(side, width), min(side, height)


def main(path, search=False):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read landmark input: {path}")
    h, w = img.shape[:2]
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL),
        num_faces=1, output_face_blendshapes=False)
    with vision.FaceLandmarker.create_from_options(opts) as lmk:
        windows = search_windows(w, h) if search else [(0, 0, w, h)]
        pts = None
        for x, y, cw, ch in windows:
            rgb = cv2.cvtColor(img[y:y + ch, x:x + cw], cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = lmk.detect(mp_img)
            if not res.face_landmarks:
                continue
            lm = res.face_landmarks[0]
            # Return pixels in the ORIGINAL image, never crop coordinates.
            candidate = [[p.x * cw + x, p.y * ch + y] for p in lm]
            # Avoid accepting a crop that cuts through the detected face.
            if search and any(not (x <= px < x + cw and y <= py < y + ch)
                              for px, py in candidate):
                continue
            pts = candidate
            break
    if pts is None:
        print(json.dumps({"ok": False}))
        return

    def center(idxs):
        return [sum(pts[i][0] for i in idxs) / len(idxs),
                sum(pts[i][1] for i in idxs) / len(idxs)]

    print(json.dumps({"ok": True, "width": w, "height": h,
                      "left_iris": center(LEFT_IRIS),
                      "right_iris": center(RIGHT_IRIS),
                      "landmarks": pts}))


if __name__ == "__main__":
    # Legacy workers call this CLI without flags. Give them the same small-face
    # fallback without requiring cancellation of an in-flight reconstruction.
    main(sys.argv[1], search="--whole-image-only" not in sys.argv[2:])
