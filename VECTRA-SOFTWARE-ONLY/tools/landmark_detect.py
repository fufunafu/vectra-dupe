"""Standalone facial-landmark detector (run with the isolated .venv-mp interpreter,
which has mediapipe 0.10 + numpy 2). Outputs pixel-space landmarks as JSON to stdout.

    .venv-mp/bin/python tools/landmark_detect.py <image_path>

Uses the MediaPipe Tasks FaceLandmarker (model: tools/face_landmarker.task), which
returns 478 landmarks including iris centers (468 = left iris, 473 = right iris).
"""
import sys, json, os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def main(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL),
        num_faces=1, output_face_blendshapes=False)
    with vision.FaceLandmarker.create_from_options(opts) as lmk:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = lmk.detect(mp_img)
    if not res.face_landmarks:
        print(json.dumps({"ok": False}))
        return
    lm = res.face_landmarks[0]
    pts = [[p.x * w, p.y * h] for p in lm]

    def center(idxs):
        return [sum(pts[i][0] for i in idxs) / len(idxs),
                sum(pts[i][1] for i in idxs) / len(idxs)]

    print(json.dumps({"ok": True, "width": w, "height": h,
                      "left_iris": center(LEFT_IRIS),
                      "right_iris": center(RIGHT_IRIS),
                      "landmarks": pts}))


if __name__ == "__main__":
    main(sys.argv[1])
