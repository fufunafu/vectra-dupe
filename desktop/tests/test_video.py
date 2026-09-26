import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
from PIL import Image
import trimesh

from facemap_desktop.video_decode import probe, _upright
from facemap_desktop.video_fixture import make_video
from facemap_desktop.video_input import prepare_video
from facemap_desktop.video_manifest import validate, assemble, FORMAT, PART_BYTES
from facemap_desktop.reconstruct import reconstruct
from facemap_desktop.session import CaptureError


class VideoManifestTests(unittest.TestCase):
    def test_video_parts_are_bounded_complete_ordered_and_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); raw = root / 'raw'; raw.mkdir()
            parts = [b'a' * PART_BYTES, b'the original final bytes']
            manifest = {'format': FORMAT, 'capture_kind': 'video', 'state': 'saved', 'video': {
                'file': 'capture.mov', 'bytes': sum(map(len, parts)), 'parts': [
                    {'name': f'video_{i:04d}.bin', 'bytes': len(part)} for i, part in enumerate(parts)]}}
            (raw / 'session.json').write_text(json.dumps(manifest))
            for i, part in enumerate(parts): (raw / f'video_{i:04d}.bin').write_bytes(part)
            self.assertEqual(validate(raw), manifest)
            self.assertEqual(assemble(raw, root).read_bytes(), b''.join(parts))
            self.assertEqual((raw / 'video_0001.bin').read_bytes(), parts[1])
            with self.assertRaises(FileExistsError): assemble(raw, root)
            manifest['video']['parts'].reverse()
            (raw / 'session.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'out of order'): validate(raw)
            manifest['video']['parts'].reverse()
            (raw / 'session.json').write_text(json.dumps(manifest))
            (raw / 'video_0001.bin').write_bytes(b'short')
            with self.assertRaisesRegex(ValueError, 'truncated'): validate(raw)

    def test_path_escape_and_oversize_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp)
            for file, size in [('../capture.mov', 1), ('capture.mov', 2 * 1024**3 + 1), ('capture.mov', True)]:
                manifest = {'format': FORMAT, 'state': 'saved', 'capture_kind': 'video',
                            'video': {'file': file, 'bytes': size, 'parts': []}}
                (raw / 'session.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError): validate(raw)


class VideoPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='facemap-video-pipeline-')
        cls.root = Path(cls.temp.name)
        cls.source = make_video(cls.root / 'generated-ellipsoid.mov')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_actual_video_to_textured_model_without_supplied_camera_data(self):
        import pycolmap
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        output = self.root / 'result'
        mapping = pycolmap.incremental_mapping
        attempts = 0

        def fail_first_initialization(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            return {} if attempts == 1 else mapping(*args, **kwargs)

        # Exercise a real alternate-pair reconstruction after a failed initial
        # alignment, without lowering any of the normal output requirements.
        with patch.object(pycolmap, 'incremental_mapping', side_effect=fail_first_initialization):
            report = reconstruct(self.source, output, preset='quick')
        self.assertGreaterEqual(report['camera_recovery']['initialization_attempts'], 2)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['mode'], 'video')
        self.assertFalse(report['metric_scale_available'])
        self.assertFalse(report['measurement_validated'])
        self.assertEqual(report['glb_units'], 'arbitrary')
        self.assertGreaterEqual(report['camera_recovery']['registered_frames'], 12)
        self.assertGreaterEqual(report['integrated_views'], 3)
        self.assertGreater(report['triangles'], 1000)
        self.assertGreater(report['texture_coverage'], .7)
        surface = trimesh.load(output / 'surface.ply', process=False)
        model = trimesh.load(output / 'model.glb', force='scene')
        self.assertTrue(np.isfinite(surface.vertices).all())
        self.assertGreater(surface.extents.min() / surface.extents.max(), .15)
        self.assertTrue(np.allclose(surface.extents, model.extents, atol=1e-5))
        self.assertFalse((output / 'surface-mm.ply').exists())
        self.assertFalse(list(output.glob('facemap-import-*')))
        self.assertFalse(list(output.rglob('session.json')))
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), digest)
        with self.assertRaises(FileExistsError): reconstruct(self.source, output)
        self.assertEqual(json.loads((output / 'report.json').read_text())['status'], 'complete')

    def test_low_resolution_and_depth_mode_rejected_without_fake_model(self):
        low = self.root / 'low.mp4'
        with av.open(str(low), 'w') as output:
            stream = output.add_stream('libx264', rate=30)
            stream.width = stream.height = 64
            stream.pix_fmt = 'yuv420p'
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.full((64,64,3),128,np.uint8),format='rgb24')): output.mux(packet)
            for packet in stream.encode(): output.mux(packet)
        with self.assertRaisesRegex(ValueError, 'native 4K'): reconstruct(low, self.root / 'low-result')
        self.assertEqual(json.loads((self.root / 'low-result' / 'report.json').read_text())['status'], 'failed')
        self.assertFalse((self.root / 'low-result' / 'model.glb').exists())
        with self.assertRaisesRegex(CaptureError, 'no LiDAR'): reconstruct(self.source, self.root / 'depth-result', mode='depth')

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is only needed for independent decoder comparison')
    def test_phone_display_rotation_and_decoded_pixels_match_ffmpeg(self):
        rotated = self.root / 'rotated.mov'
        help_text = subprocess.run(['ffmpeg','-hide_banner','-h','full'], capture_output=True,
                                   text=True, check=True, timeout=15).stdout
        if '-display_rotation' in help_text:
            rotation_args = ['-display_rotation','90','-i',str(self.source),'-c','copy']
        else:
            # Older FFmpeg releases set the container display matrix through
            # stream metadata. Recent releases require the input override.
            rotation_args = ['-i',str(self.source),'-c','copy','-metadata:s:v:0','rotate=90']
        subprocess.run(['ffmpeg','-v','error','-n',*rotation_args,str(rotated)], check=True, timeout=30)
        meta, _ = probe(rotated)
        self.assertEqual((meta['display_width'],meta['display_height']), (2160,3840))
        reference = self.root / 'rotated.png'
        subprocess.run(['ffmpeg','-v','error','-n','-i',str(rotated),'-frames:v','1','-pix_fmt','rgb24',str(reference)], check=True, timeout=30)
        with av.open(str(rotated)) as video:
            actual = _upright(next(video.decode(video=0)))
        with Image.open(reference) as expected:
            self.assertEqual(actual.shape, np.asarray(expected).shape)
            # Bundled and system FFmpeg use different swscale versions. Their
            # integer YUV-to-RGB conversion differs by at most three code values;
            # a wrong orientation or frame would fail by a much larger margin.
            self.assertLessEqual(np.abs(actual.astype(float)-np.asarray(expected).astype(float)).max(), 3.)
