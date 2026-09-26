import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation

from facemap_desktop.alignment import align_depth_session, _features
from facemap_desktop.fixtures import make_plane
from facemap_desktop.reconstruct import reconstruct, _clean_mesh, PRESETS
from facemap_desktop.session import CaptureError, validate


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='facemap-alignment-test-')
        self.root = Path(self.temp.name)
        self.capture = make_plane(self.root / 'capture')
        self.manifest = self.capture / 'session.json'

    def tearDown(self):
        self.temp.cleanup()

    def with_depth_photos(self):
        meta = json.loads(self.manifest.read_text())
        for pose, photo in zip(meta['poses'], meta['color_frames']):
            name = pose['name'] + '.jpg'
            shutil.copyfile(self.capture / photo['color_file'], self.capture / name)
            pose.update({k: copy.deepcopy(v) for k, v in photo.items() if k.startswith('rgb_')})
            pose['color_file'] = name
            pose['rgb_world_to_camera'] = copy.deepcopy(photo['world_to_camera'])
        self.manifest.write_text(json.dumps(meta))
        return meta

    def test_known_large_pose_drift_recovered_without_changing_capture(self):
        meta = self.with_depth_photos()
        expected = [np.array(p['world_to_camera']) for p in meta['poses']]
        for index, (angles, shift) in enumerate([([0, 0, 0], [0, 0, 0]),
                                               ([0, 0, 3], [30, -20, 70]),
                                               ([0, -4, 0], [-40, 25, -90])]):
            error = np.eye(4)
            error[:3, :3] = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
            error[:3, 3] = shift
            entry = meta['poses'][index]
            entry['world_to_camera'] = (expected[index] @ np.linalg.inv(error)).tolist()
            entry['rgb_world_to_camera'] = copy.deepcopy(entry['world_to_camera'])
        self.manifest.write_text(json.dumps(meta))
        original = self.manifest.read_bytes()
        session = validate(self.capture)
        original_metadata = copy.deepcopy(session.metadata)
        report = {}
        aligned = align_depth_session(session, report, lambda *_: None)
        self.assertEqual(report['status'], 'recovered')
        self.assertGreater(report['recorded_median_error_mm'], 30)
        self.assertLess(report['aligned_median_error_mm'], 1)
        self.assertEqual(report['aligned_sweep_photos'], 3)
        for actual, wanted in zip(aligned.poses, expected):
            matrix = np.array(actual['world_to_camera'])
            self.assertLess(np.linalg.norm(matrix[:3, 3]-wanted[:3, 3]), 1)
            self.assertLess(Rotation.from_matrix(matrix[:3, :3] @ wanted[:3, :3].T).magnitude(), .01)
            np.testing.assert_allclose(matrix, actual['rgb_world_to_camera'])
        for actual, wanted in zip(aligned.metadata['color_frames'], expected):
            self.assertLess(np.linalg.norm(np.array(actual['world_to_camera'])[:3, 3]-wanted[:3, 3]), 2)
        self.assertEqual(self.manifest.read_bytes(), original)
        self.assertEqual(session.metadata, original_metadata)

    def test_already_aligned_views_keep_recorded_poses(self):
        self.with_depth_photos()
        session = validate(self.capture)
        report = {}
        aligned = align_depth_session(session, report, lambda *_: None)
        self.assertEqual(report['status'], 'recorded')
        self.assertEqual(aligned.metadata, session.metadata)

    def test_depth_photos_at_separate_camera_positions_use_their_own_pose(self):
        meta = self.with_depth_photos()
        first, second = meta['poses'][:2]
        shutil.copyfile(self.capture / second['color_file'], self.capture / first['color_file'])
        first['rgb_world_to_camera'] = second['world_to_camera']
        self.manifest.write_text(json.dumps(meta))
        session = validate(self.capture)
        features = _features(session, first, with_depth=True)
        self.assertGreater(len(features.points), 100)
        self.assertLess(np.max(np.abs(features.points[:, 2])), .01)

    def test_unverifiable_scan_fails_without_exporting_a_model(self):
        meta = self.with_depth_photos()
        for entry in meta['poses']:
            Image.new('RGB', (320, 240), (140, 140, 140)).save(self.capture / entry['color_file'])
        output = self.root / 'result'
        with self.assertRaisesRegex(CaptureError, 'could not be aligned reliably'):
            reconstruct(self.capture, output, preset='quick')
        report = json.loads((output / 'report.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['depth_alignment']['status'], 'rejected')
        self.assertFalse((output / 'model.glb').exists())
        self.assertFalse((output / 'surface-mm.ply').exists())
        self.assertTrue(self.manifest.exists())

    def test_separated_depth_without_photos_is_rejected(self):
        meta = json.loads(self.manifest.read_text())
        for index, pose in enumerate(meta['poses']):
            pose['world_to_camera'][2][3] += index * 90
        self.manifest.write_text(json.dumps(meta))
        report = {}
        with self.assertRaisesRegex(CaptureError, 'do not overlap consistently'):
            align_depth_session(validate(self.capture), report, lambda *_: None)
        self.assertEqual(report['status'], 'rejected')

    def test_two_disconnected_large_surfaces_are_not_reported_as_success(self):
        a = o3d.geometry.TriangleMesh.create_sphere(radius=35)
        b = o3d.geometry.TriangleMesh.create_sphere(radius=35).translate((100, 0, 0))
        with self.assertRaisesRegex(CaptureError, 'one consistent surface'):
            _clean_mesh(a + b, PRESETS['quick'])


if __name__ == '__main__':
    unittest.main()
