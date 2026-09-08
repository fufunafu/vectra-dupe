"""Safety screens must reject corrupt scans without inventing a zero volume."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))
from vectra3d import analyze, compare, quality
import processing
import app as api
import store


def plane(z=50, half_width=65):
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector([[-half_width, -60, z], [half_width, -60, z],
                                   [half_width, 60, z], [-half_width, 60, z]]),
        o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]))


class GeometryTests(unittest.TestCase):
    def test_experimental_reports_failures_without_volume_or_bias_correction(self):
        field = analyze.DistanceField(np.zeros((4, 3)), np.array([1., 2., 3., np.nan]), np.ones(4))
        with patch.object(compare.register, 'register_with_exclusion', return_value=np.eye(4)), \
                patch.object(analyze, 'signed_distance_field', return_value=field), \
                patch.object(analyze, 'subtract_bias_field') as bias, \
                patch.object(compare, 'detect_change_regions') as detect:
            result = compare.compare_experimental(plane(), plane(z=80) + plane(z=83))
        self.assertEqual(result.regions, [])
        self.assertTrue(result.quality_checks['diagnostic_only'])
        self.assertFalse(result.quality_checks['after']['eligible'])
        self.assertFalse(result.quality_checks['alignment']['eligible'])
        self.assertTrue(result.quality_checks['warnings'])
        bias.assert_not_called()
        detect.assert_not_called()
        with self.assertRaises(quality.QualityError):
            compare.compare_experimental(plane(), o3d.geometry.TriangleMesh())

    def test_single_surface_passes_and_shared_triangle_edges_are_not_layers(self):
        result = quality.inspect_mesh(plane())
        self.assertTrue(result["eligible"], result)
        self.assertEqual(result["front_coverage"], 1)
        self.assertEqual(result["layered_fraction"], 0)

    def test_overlapping_sheets_are_rejected(self):
        result = quality.inspect_mesh(plane() + plane(z=53))
        self.assertFalse(result["eligible"])
        self.assertEqual(result["layered_fraction"], 1)
        with self.assertRaises(quality.QualityError):
            compare.compare_sessions(plane(), plane() + plane(z=53))

    def test_small_coverage_empty_and_nonfinite_meshes_are_rejected(self):
        bad = plane()
        np.asarray(bad.vertices)[0, 0] = np.nan
        for mesh in (plane(half_width=20), o3d.geometry.TriangleMesh(), bad):
            with self.subTest(vertices=len(mesh.vertices)):
                self.assertFalse(quality.inspect_mesh(mesh)["eligible"])

    def test_alignment_is_checked_before_bias_removal(self):
        quality.require_alignment(plane(), plane())
        with self.assertRaises(quality.QualityError):
            quality.require_alignment(plane(), plane(z=80))

    def test_nan_coverage_is_area_weighted_and_cannot_be_reported_as_zero(self):
        field = analyze.DistanceField(np.zeros((3, 3)), np.array([0., 0., np.nan]),
                                       np.array([1., 1., 20.]))
        with self.assertRaises(quality.QualityError):
            quality.require_distance_coverage(field, np.ones(3, bool))
        self.assertEqual(quality.require_distance_coverage(
            field, np.array([True, True, False])), 1)


class SessionTests(unittest.TestCase):
    def test_photo_comparison_is_explicit_and_rejects_active_sources(self):
        from vectra3d import photo_compare
        with tempfile.TemporaryDirectory() as data, patch.object(store,'DATA_DIR',data), \
                TestClient(api.app) as client, \
                patch.object(photo_compare,'compare_assets',return_value={'source':'photo','diagnostic_only':True}) as run:
            pid=store.create_patient('Photo test')['id']
            sids=[store.create_session(pid,label)['id'] for label in ('before','after')]
            for sid in sids:
                directory=Path(store.session_dir(pid,sid))
                (directory/'display_uncropped.npz').write_bytes(b'fixture: parser is mocked')
                (directory/'stats.json').write_text(json.dumps({'align_method':'landmark_umeyama'}))
                store.update_session_meta(pid,sid,processed=True,status='done')
            body={'before':sids[0],'after':sids[1],'source':'photo'}
            url=f'/api/patients/{pid}/compare'
            self.assertEqual(client.post(url,json=body).status_code,400)
            run.assert_not_called()
            response=client.post(url,json={**body,'experimental':True})
            self.assertEqual(response.status_code,200,response.text)
            self.assertTrue(response.json()['id'].endswith('__photo'))
            self.assertIsNone(run.call_args.args[3])
            for value in (0,-1,100):
                self.assertEqual(client.post(url,json={**body,'experimental':True,'intercanthal_mm':value}).status_code,422)
            run.reset_mock()
            store.update_session_meta(pid,sids[1],status='processing')
            self.assertEqual(client.post(url,json={**body,'experimental':True}).status_code,422)
            run.assert_not_called()

    def test_experimental_api_keeps_normal_results_and_input_meshes_unchanged(self):
        with tempfile.TemporaryDirectory() as data, patch.object(store, 'DATA_DIR', data), \
                TestClient(api.app) as client:
            pid = store.create_patient('Experimental test')['id']
            sids = [store.create_session(pid, label)['id'] for label in ('before', 'after')]
            paths = []
            for sid, mesh in zip(sids, (plane(), plane() + plane(z=53))):
                directory = Path(store.session_dir(pid, sid))
                path = directory / 'mesh.ply'
                o3d.io.write_triangle_mesh(str(path), mesh)
                report = quality.inspect_mesh(mesh)
                report['mesh_sha256'] = quality.mesh_fingerprint(str(path))
                (directory / 'stats.json').write_text(json.dumps({'measurement_quality': report}))
                store.update_session_meta(pid, sid, processed=True, status='done')
                paths.append(path)
            original = [p.read_bytes() for p in paths]
            url = f'/api/patients/{pid}/compare'
            body = dict(zip(('before', 'after'), sids))
            self.assertEqual(client.post(url, json=body).status_code, 422)
            normal = Path(store.compare_dir(pid, *sids))
            normal.mkdir(parents=True)
            (normal / 'result.json').write_text('{}')
            field = analyze.DistanceField(np.asarray(plane().vertices), np.ones(4), np.ones(4))
            with patch.object(compare.register, 'register_with_exclusion', return_value=np.eye(4)), \
                    patch.object(analyze, 'signed_distance_field', return_value=field):
                response = client.post(url, json={**body, 'experimental': True})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertTrue(result['diagnostic_only'])
            self.assertIsNone(result['net_significant_volume_ml'])
            self.assertEqual(result['regions'], [])
            self.assertTrue(result['quality_checks']['warnings'])
            self.assertTrue(result['id'].endswith('__experimental'))
            self.assertEqual((normal / 'result.json').read_text(), '{}')
            self.assertEqual([p.read_bytes() for p in paths], original)
            saved = [r for r in store.list_compares(pid) if r.get('experimental')][0]
            self.assertEqual(saved['after'], sids[1])
            for status in ('processing', 'queued', 'failed'):
                store.update_session_meta(pid, sids[1], status=status)
                self.assertEqual(client.post(url, json={**body, 'experimental': True}).status_code, 422)
            store.update_session_meta(pid, sids[1], status='done')
            o3d.io.write_triangle_mesh(str(paths[1]), plane(z=70))
            self.assertEqual(client.post(url, json={**body, 'experimental': True}).status_code, 422)

    def test_api_rejects_copies_of_the_same_capture(self):
        with tempfile.TemporaryDirectory() as data, patch.object(store, "DATA_DIR", data), \
                TestClient(api.app) as client:
            pid = store.create_patient("Copy test")["id"]
            original = store.create_session(pid, "original")["id"]
            copy = store.create_session(pid, "copy")["id"]
            store.update_session_meta(pid, copy, source_session_id=original)
            response = client.post(f'/api/patients/{pid}/compare',
                                   json={"before": original, "after": copy})
            self.assertEqual(response.status_code, 400)
            self.assertIn("same capture", response.json()["detail"])
            response = client.post(f'/api/patients/{pid}/compare',
                                   json={"before": original, "after": copy, "experimental": True})
            self.assertEqual(response.status_code, 400)

    def test_active_failed_and_tampered_results_fail_closed(self):
        with tempfile.TemporaryDirectory() as data:
            directory = Path(data)
            mesh_path = directory / "mesh.ply"
            o3d.io.write_triangle_mesh(str(mesh_path), plane())
            report = quality.inspect_mesh(plane())
            report["mesh_sha256"] = quality.mesh_fingerprint(str(mesh_path))
            stats = {"measurement_grade": True, "measurement_quality": report}
            (directory / "stats.json").write_text(json.dumps(stats))
            meta = {"processed": True, "status": "done"}
            processing.require_session_measurement(data, meta)
            for status in ("queued", "processing", "failed"):
                with self.assertRaises(quality.QualityError):
                    processing.require_session_measurement(data, {**meta, "status": status})
            o3d.io.write_triangle_mesh(str(mesh_path), plane(z=55))
            with self.assertRaisesRegex(quality.QualityError, "changed"):
                processing.require_session_measurement(data, meta)
            (directory / "stats.json").write_text(json.dumps({"measurement_grade": True}))
            # Legacy geometry is checked on read, not blindly accepted or
            # forced through another expensive reconstruction.
            legacy_before = (directory / "stats.json").read_bytes()
            processing.require_session_measurement(data, meta)
            self.assertEqual((directory / "stats.json").read_bytes(), legacy_before)
            o3d.io.write_triangle_mesh(str(mesh_path), plane() + plane(z=53))
            with self.assertRaisesRegex(quality.QualityError, "Overlapping"):
                processing.require_session_measurement(data, meta)

    def test_api_blocks_bad_scans_before_creating_comparison_artifacts(self):
        with tempfile.TemporaryDirectory() as data, patch.object(store, "DATA_DIR", data), \
                patch.object(api.processing, "compare_sessions_on_disk") as run, \
                TestClient(api.app) as client:
            pid = client.post('/api/patients', json={"name": "Quality test"}).json()["id"]
            sids = [store.create_session(pid, label)["id"] for label in ("before", "after")]
            for sid in sids:
                store.update_session_meta(pid, sid, processed=True, status="done")
                path = Path(store.session_dir(pid, sid))
                (path / "stats.json").write_text(json.dumps({"measurement_quality": {
                    "version": quality.QUALITY_VERSION, "eligible": False,
                    "reasons": ["Overlapping surface layers."]}}))
            response = client.post(f'/api/patients/{pid}/compare',
                                   json={"before": sids[0], "after": sids[1]})
            self.assertEqual(response.status_code, 422)
            self.assertIn("Overlapping", response.json()["detail"])
            run.assert_not_called()
            self.assertFalse((Path(data) / pid / "compares").exists())


if __name__ == '__main__':
    unittest.main()
