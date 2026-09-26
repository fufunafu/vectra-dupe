import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from facemap_desktop.readiness import assess
from facemap_desktop.session import IncompleteCapture, validate
from facemap_desktop.reconstruct import reconstruct

FIXTURE = Path(__file__).resolve().parents[2] / 'tests/fixtures/readiness-complete.json'


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(FIXTURE.read_text())

    def test_complete_scan_passes_but_many_repeated_views_do_not(self):
        self.assertEqual(assess(self.manifest)['status'], 'passed')
        first = self.manifest['color_frames'][0]['world_to_camera']
        for photo in self.manifest['color_frames']:
            photo['world_to_camera'] = first
        result = assess(self.manifest)
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['usable_distinct_photos'], 1)

    def test_missing_regions_are_reported_despite_sufficient_photo_count(self):
        gaps = {0, 1, 2, 4, 5, 6, 10}
        self.manifest['color_frames'] = [p for i, p in enumerate(self.manifest['color_frames']) if i not in gaps]
        result = assess(self.manifest)
        self.assertGreaterEqual(result['usable_distinct_photos'], 12)
        self.assertEqual(result['missing_regions'], ['lower left', 'lower right', 'level front'])

    def test_sharpness_stability_and_unknown_metadata_block(self):
        for mutation in ('blur', 'motion', 'missing_quality', 'guided'):
            m = copy.deepcopy(self.manifest)
            if mutation == 'blur':
                for photo in m['color_frames']: photo['photo_quality']['sharpness'] = 1
            elif mutation == 'motion': m['poses'][0]['quality']['maximum_translation_mm'] = 3
            elif mutation == 'missing_quality': m['color_frames'][0].pop('photo_quality')
            else: m['poses'].pop()
            with self.subTest(mutation=mutation):
                self.assertEqual(assess(m)['status'], 'blocked')

    def test_photo_only_requires_guided_views_and_twenty_distinct_photos(self):
        m = self.manifest
        m['capture_kind'] = 'photo_only'
        m['color_frames'] += [dict(p, name='key_'+p['name']) for p in m['poses']]
        m['poses'] = []
        self.assertEqual(assess(m)['status'], 'passed')
        m['color_frames'] = m['color_frames'][2:]
        self.assertEqual(assess(m)['status'], 'blocked')

    def test_legacy_incomplete_warning_is_not_erased_by_new_checks(self):
        self.manifest['warnings'] = ['Missing coverage: lower left.\n\nSaved anyway.']
        self.assertIn('saved incomplete', ' '.join(assess(self.manifest)['issues']))

    def test_import_rejects_before_alignment_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'capture'; source.mkdir()
            m = self.manifest
            # All files are present and readable, but one required view is absent.
            m['poses'] = m['poses'][:-1]
            for p in m['poses']:
                np.full((4, 4), 300, dtype='<f4').tofile(source / p['depth_file'])
            for p in m['poses'] + m['color_frames']:
                Image.new('RGB', (8, 8), (120, 130, 140)).save(source / p['color_file'])
            manifest = source / 'session.json'; manifest.write_text(json.dumps(m))
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            with self.assertRaises(IncompleteCapture): validate(source)
            output = Path(temporary) / 'result'
            with patch('facemap_desktop.alignment.align_depth_session') as alignment:
                with self.assertRaises(IncompleteCapture): reconstruct(source, output)
                alignment.assert_not_called()
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['capture_readiness']['status'], 'blocked')
            self.assertFalse((output / 'model.glb').exists())
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()})

    def test_demo_remains_explicitly_synthetic(self):
        self.manifest.update(is_demo=True, poses=[], color_frames=[])
        self.assertEqual(assess(self.manifest)['status'], 'demo')
