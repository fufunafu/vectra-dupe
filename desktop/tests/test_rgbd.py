"""Synthetic capture integrity and calibrated detail checks, never facial accuracy."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from PIL import Image

from facemap_desktop.session import Session, validate, CaptureError
from facemap_desktop.rgbd import FORMAT, assess, read_depth, gate_stereo, alignment_pairs
from facemap_desktop.reconstruct import matrix_k, extrinsic, stereo_depth


from facemap_desktop.rgbd_fixture import make_rgbd_plane


class RGBDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        cls.capture = make_rgbd_plane(cls.root/'capture')
        cls.original = json.loads((cls.capture/'session.json').read_text())
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def setUp(self): (self.capture/'session.json').write_text(json.dumps(self.original))
    def tearDown(self): (self.capture/'session.json').write_text(json.dumps(self.original))

    def test_valid_capture_and_bounded_connected_pair_graph(self):
        session = validate(self.capture)
        self.assertEqual(assess(session.metadata)['status'], 'passed')
        pairs = alignment_pairs(session)
        self.assertLess(len(pairs), len(session.poses)*10)
        self.assertTrue(all((i, i+1) in pairs for i in range(len(session.poses)-1)))

    def test_confidence_gates_without_changing_source(self):
        session = validate(self.capture); entry = session.poses[0]
        path = self.capture/entry['confidence_file']; original = path.read_bytes()
        depth_path = self.capture/entry['depth_file']; before = hashlib.sha256(depth_path.read_bytes()).hexdigest()
        try:
            values = np.frombuffer(original, dtype=np.uint8).copy(); values[0] = 0; values.tofile(path)
            measured = read_depth(session, entry)
            self.assertEqual(measured[0, 0], 0); self.assertGreater(measured[0, 1], 0)
            self.assertEqual(before, hashlib.sha256(depth_path.read_bytes()).hexdigest())
            values[0] = 3; values.tofile(path)
            with self.assertRaisesRegex(CaptureError, 'confidence'): validate(self.capture)
        finally: path.write_bytes(original)

    def test_calibration_timestamp_and_tracking_damage_rejected(self):
        for mutate in (lambda p: p['intrinsics'].update(fx=123), lambda p: p.update(frame_timestamp_seconds=50),
                       lambda p: p.update(tracking_state='limited'), lambda p: p['rgb_world_to_camera'][0].__setitem__(3, 100)):
            meta = copy.deepcopy(self.original); mutate(meta['poses'][0])
            (self.capture/'session.json').write_text(json.dumps(meta))
            with self.assertRaises(CaptureError): validate(self.capture)

    def test_repeated_views_and_missing_tail_are_blocked(self):
        meta = copy.deepcopy(self.original)
        for pose in meta['poses']: pose['world_to_camera'] = copy.deepcopy(meta['poses'][0]['world_to_camera'])
        self.assertTrue(any('distinct' in i for i in assess(meta)['issues']))
        meta = copy.deepcopy(self.original); meta['rgbd']['duration_seconds'] = 30
        self.assertTrue(any('Too much' in i for i in assess(meta)['issues']))

    def test_stereo_detail_recovers_plane_and_rejects_inconsistent_depth(self):
        session = validate(self.capture); left, right = session.poses[:2]
        depth, _, k, e, _ = stereo_depth(session, left, right, 640)
        gated = gate_stereo(session, depth, k, e, [left, right])
        self.assertGreater(np.count_nonzero(gated), 1000)
        self.assertEqual(np.count_nonzero(gate_stereo(session, depth+60, k, e, [left, right])), 0)
        y, x = np.indices(gated.shape); good = gated > 0
        points = np.stack(((x-k[0, 2])*gated/k[0, 0], (y-k[1, 2])*gated/k[1, 1], gated), axis=-1)
        world = (points-e[:3, 3]) @ e[:3, :3]
        self.assertLess(np.median(np.abs(world[..., 2][good])), 3)

    def test_small_pose_drift_uses_bounded_sparse_recovery(self):
        from facemap_desktop.alignment import align_depth_session
        session = validate(self.capture)
        metadata = copy.deepcopy(session.metadata)
        for pose in metadata['poses'][10:]:
            e = np.array(pose['world_to_camera']); correction = np.eye(4); correction[0, 3] = 8
            pose['world_to_camera'] = (e @ np.linalg.inv(correction)).tolist()
            pose['rgb_world_to_camera'] = copy.deepcopy(pose['world_to_camera'])
        diagnostics = {}
        recovered = align_depth_session(Session(session.directory, metadata), diagnostics, lambda *_: None)
        self.assertEqual(diagnostics['status'], 'recovered')
        self.assertLess(diagnostics['aligned_median_error_mm'], 2)
        for actual, expected in zip(recovered.poses, session.poses):
            self.assertLess(np.linalg.norm(np.array(actual['world_to_camera'])[:3, 3]-np.array(expected['world_to_camera'])[:3, 3]), 2)

    def test_missing_and_duplicate_camera_timeline_rejected(self):
        path = self.capture/'camera-frames.jsonl'; original = path.read_bytes()
        try:
            path.write_text('{}\n')
            with self.assertRaises(CaptureError): validate(self.capture)
            lines = original.decode().splitlines(); lines[1] = lines[0]; path.write_text('\n'.join(lines)+'\n')
            with self.assertRaises(CaptureError): validate(self.capture)
        finally: path.write_bytes(original)

if __name__ == '__main__': unittest.main()
