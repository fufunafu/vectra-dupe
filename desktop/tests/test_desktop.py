import copy
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import open3d as o3d
import trimesh

from facemap_desktop.fixtures import make_plane
from facemap_desktop.session import CaptureError, open_session, validate
from facemap_desktop.reconstruct import reconstruct, stereo_depth, stereo_pairs
from facemap_desktop.viewer import serve_viewer


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='facemap test ')
        self.root = Path(self.temp.name)
        self.capture = make_plane(self.root / 'capture')

    def tearDown(self):
        self.temp.cleanup()

    def make_zip(self, extra=None, prefix='export folder/'):
        path = self.root / 'scan.zip'
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for file in self.capture.iterdir():
                archive.write(file, prefix + file.name)
            if extra:
                for name, data in extra:
                    archive.writestr(name, data)
        return path

    def test_zip_import_and_source_preservation(self):
        path = self.make_zip()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with open_session(path) as session:
            private = session.directory
            self.assertEqual(len(session.poses), 3)
            self.assertEqual(len(session.photos), 3)
        self.assertFalse(private.exists())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_root_zip_supported(self):
        with open_session(self.make_zip(prefix='')) as session:
            self.assertTrue(session.metadata['is_demo'])

    def test_zip_path_traversal_rejected(self):
        for name in ('../escape', '/absolute', 'C:/escape', 'folder\\escape'):
            with self.subTest(name=name), self.assertRaises(CaptureError):
                with open_session(self.make_zip([(name, b'data')])):
                    pass

    def test_zip_duplicate_case_rejected(self):
        with self.assertRaises(CaptureError):
            with open_session(self.make_zip([('export folder/PHOTO_0.JPG', b'data')])):
                pass

    def test_zip_multiple_scans_rejected(self):
        with self.assertRaises(CaptureError):
            with open_session(self.make_zip([('other/session.json', b'{}')])):
                pass

    def test_zip_symlink_rejected(self):
        info = zipfile.ZipInfo('export folder/link')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(CaptureError):
            with open_session(self.make_zip([(info, 'outside')])):
                pass

    def test_truncated_depth_rejected(self):
        (self.capture / 'depth_0.bin').write_bytes(b'123')
        with self.assertRaises(CaptureError):
            validate(self.capture)

    def test_invalid_depth_rejected(self):
        depth = np.full(320 * 240, 500, dtype='<f4')
        depth[11] = np.nan
        depth.tofile(self.capture / 'depth_0.bin')
        with self.assertRaises(CaptureError):
            validate(self.capture)

    def test_wrong_image_size_rejected(self):
        path = self.capture / 'session.json'
        meta = json.loads(path.read_text())
        meta['color_frames'][0]['rgb_width'] = 321
        path.write_text(json.dumps(meta))
        with self.assertRaises(CaptureError):
            validate(self.capture)

    def test_nonrigid_camera_rejected(self):
        path = self.capture / 'session.json'
        meta = json.loads(path.read_text())
        meta['poses'][0]['world_to_camera'][0][0] = 2
        path.write_text(json.dumps(meta))
        with self.assertRaises(CaptureError):
            validate(self.capture)

    def test_photo_stereo_recovers_known_plane_horizontal_and_vertical(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                session = validate(make_plane(self.root / str(vertical), vertical=vertical, with_depth=False))
                depth, _, _, _, stats = stereo_depth(session, *session.photos[:2], image_size=640)
                valid = depth[depth > 0]
                self.assertGreater(len(valid), 10000)
                self.assertLess(abs(float(np.median(valid)) - 500), 3)
                self.assertEqual(stats['vertical_stereo'], vertical)

    def test_reversed_camera_order_has_positive_correct_depth(self):
        session = validate(self.capture)
        depth, *_ = stereo_depth(session, session.photos[1], session.photos[0], 640)
        self.assertLess(abs(float(np.median(depth[depth > 0])) - 500), 3)

    def test_blank_photos_rejected(self):
        session = validate(make_plane(self.root / 'blank', with_depth=False, blank=True))
        with self.assertRaises(CaptureError):
            stereo_depth(session, *session.photos[:2], image_size=640)

    def test_duplicate_camera_positions_rejected(self):
        photos = [copy.deepcopy(validate(self.capture).photos[0]) for _ in range(4)]
        with self.assertRaises(CaptureError):
            stereo_pairs(photos, 12)

    def test_complete_depth_export_units_and_existing_output_guard(self):
        output = self.root / 'result'
        report = reconstruct(self.make_zip(), output, preset='quick')
        self.assertEqual(report['status'], 'complete')
        self.assertFalse(report['measurement_validated'])
        mesh = o3d.io.read_triangle_mesh(str(output / 'surface-mm.ply'))
        vertices = np.asarray(mesh.vertices)
        self.assertLess(np.max(np.abs(vertices[:, 2])), 3)
        scene = trimesh.load(output / 'model.glb', force='scene')
        self.assertTrue(np.allclose(scene.extents, np.ptp(vertices, axis=0) * .001, atol=1e-5))
        self.assertFalse(list(output.glob('facemap-import-*')))
        with self.assertRaises(FileExistsError):
            reconstruct(self.capture, output, preset='quick')
        self.assertEqual(json.loads((output / 'report.json').read_text())['status'], 'complete')

    def test_complete_photo_only_reconstruction(self):
        capture = make_plane(self.root / 'photos', with_depth=False)
        output = self.root / 'photo-result'
        report = reconstruct(capture, output, preset='quick')
        self.assertEqual(report['mode'], 'photos')
        self.assertGreaterEqual(report['integrated_views'], 2)
        self.assertTrue((output / 'model.glb').is_file())

    def test_failed_reconstruction_has_no_model_and_keeps_source(self):
        capture = make_plane(self.root / 'blank', with_depth=False, blank=True)
        output = self.root / 'failed'
        with self.assertRaises(CaptureError):
            reconstruct(capture, output, preset='quick')
        self.assertEqual(json.loads((output / 'report.json').read_text())['status'], 'failed')
        self.assertFalse((output / 'model.glb').exists())
        self.assertFalse(list(output.glob('facemap-import-*')))
        self.assertTrue((capture / 'session.json').is_file())

    def test_viewer_serves_only_allowlisted_result_assets(self):
        output = self.root / 'view-result'
        reconstruct(self.capture, output, preset='quick')
        (output / 'private.txt').write_text('not a public asset')
        server, url = serve_viewer(output, open_browser=False)
        try:
            self.assertEqual(urlopen(url).status, 200)
            self.assertEqual(urlopen(url + 'model.glb').read(4), b'glTF')
            for suffix in ('private.txt', '../private.txt', '%2e%2e/private.txt'):
                with self.subTest(suffix=suffix), self.assertRaises(HTTPError):
                    urlopen(url + suffix)
            with self.assertRaises(HTTPError):
                urlopen(Request(url, headers={'Host': 'attacker.example'}))
            with self.assertRaises(HTTPError):
                urlopen(Request(url, data=b'anything', method='POST'))
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
