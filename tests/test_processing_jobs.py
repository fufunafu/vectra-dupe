"""Regression checks for retries and queued API requests, without reconstruction.

Requires the installed Starlette version's test client dependency (httpx2).
Run: .venv/bin/python -m unittest discover -s tests -p 'test_processing_jobs.py' -v
"""

import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

from fastapi.testclient import TestClient
from vectra3d import photogrammetry as pg
import app as api
import store
from jobs import ProcessingQueue


class RetryTests(unittest.TestCase):
    def paths(self):
        return (
            (pg.reconstruct_metric, "_reconstruct_metric_once", ("raw", [], [], "out")),
            (pg.reconstruct_photo_only, "_reconstruct_photo_only_once", ("raw", [], "out")),
        )

    def result(self):
        return SimpleNamespace(stats={"align_rms_mm": 2.0, "align_ipd_mm": 62.0})

    def test_first_passing_result_stops_full_reconstructions(self):
        for run, target, args in self.paths():
            with self.subTest(path=target):
                expected = self.result()
                with patch.object(pg, target, return_value=expected) as once:
                    result = run(*args, attempts=5)
                self.assertIs(result, expected)
                self.assertEqual(once.call_count, 1)
                self.assertEqual(result.stats["oc_attempts"], 1)
                self.assertEqual(result.stats["oc_attempts_passed"], 1)

    def test_rejected_result_retries_then_stops_and_records_failure(self):
        for run, target, args in self.paths():
            with self.subTest(path=target):
                with patch.object(pg, target, side_effect=[
                    RuntimeError("poor landmark alignment"), self.result()
                ]) as once:
                    result = run(*args, attempts=5)
                self.assertEqual(once.call_count, 2)
                self.assertEqual(result.stats["oc_attempts"], 2)
                self.assertEqual(result.stats["oc_attempts_passed"], 1)
                self.assertIn("poor landmark alignment",
                              result.stats["oc_attempt_log"][0]["error"])

    def test_all_rejected_results_exhaust_limit_and_raise(self):
        for run, target, args in self.paths():
            with self.subTest(path=target):
                with patch.object(pg, target, side_effect=RuntimeError("bad scale")) as once:
                    with self.assertRaisesRegex(RuntimeError, "all 3.*failed"):
                        run(*args, attempts=3)
                self.assertEqual(once.call_count, 3)


class QueueTests(unittest.TestCase):
    def test_fifo_duplicates_and_failure_do_not_block_next_job(self):
        queue = ProcessingQueue()
        entered, release = Event(), Event()
        order = []

        def first():
            order.append("first")
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test did not release worker")
            raise ValueError("failed scan")

        try:
            a = queue.submit(("patient", "a"), first, on_queued=lambda: None)
            self.assertTrue(entered.wait(10))
            duplicate = queue.submit(("patient", "a"), lambda: self.fail("duplicate ran"),
                                     on_queued=lambda: self.fail("duplicate requeued"))
            b = queue.submit(("patient", "b"), lambda: order.append("second"),
                             on_queued=lambda: None)
            c = queue.submit(("patient", "c"), lambda: order.append("third"),
                             on_queued=lambda: None)
            self.assertIs(a, duplicate)
            self.assertEqual(order, ["first"])
            release.set()
            with self.assertRaisesRegex(ValueError, "failed scan"):
                a.result(timeout=10)
            b.result(timeout=10)
            c.result(timeout=10)
            self.assertEqual(order, ["first", "second", "third"])
            # Explicit reprocessing works after the earlier job has finished.
            again = queue.submit(("patient", "a"), lambda: "new result",
                                 on_queued=lambda: None)
            self.assertEqual(again.result(timeout=10), "new result")
        finally:
            release.set()
            queue.shutdown()


class APITests(unittest.TestCase):
    def session(self, client, pid, label):
        response = client.post(f"/api/patients/{pid}/sessions", json={"label": label})
        self.assertEqual(response.status_code, 200)
        sid = response.json()["id"]
        response = client.post(f"/api/patients/{pid}/sessions/{sid}/upload",
                               files={"files": ("session.json", b"{}")})
        self.assertEqual(response.status_code, 200)
        return f"/api/patients/{pid}/sessions/{sid}"

    def test_api_queues_deduplicates_and_waits_for_the_existing_job(self):
        entered, release = Event(), Event()
        calls = []

        def process(raw, output, texture_mode):
            calls.append(raw)
            if len(calls) == 1:
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("test did not release worker")
            return {"patient_id": "", "vertices": 42}

        with tempfile.TemporaryDirectory() as data, patch.object(store, "DATA_DIR", data), \
                patch.object(api.processing, "process_session", side_effect=process), \
                TestClient(api.app) as client:
            try:
                pid = client.post("/api/patients", json={"name": "Test"}).json()["id"]
                a = self.session(client, pid, "first")
                b = self.session(client, pid, "second")
                self.assertEqual(client.post(a + "/process").status_code, 200)
                self.assertTrue(entered.wait(10))
                self.assertEqual(client.get(a).json()["status"], "processing")
                self.assertEqual(client.post(b + "/process").json()["status"], "queued")
                self.assertEqual(client.post(a + "/process").json()["status"], "processing")
                self.assertEqual(client.post(b + "/process").json()["status"], "queued")
                self.assertEqual(len(calls), 1)
                # Join a queued job through wait=true while the first is blocked.
                joined_queue = Event()
                submit = api.app.state.processing_queue.submit

                def join(*args, **kwargs):
                    future = submit(*args, **kwargs)
                    joined_queue.set()
                    return future

                with patch.object(api.app.state.processing_queue, "submit", side_effect=join), \
                        ThreadPoolExecutor(max_workers=1) as waiter:
                    joined = waiter.submit(client.post, b + "/process?wait=true")
                    try:
                        self.assertTrue(joined_queue.wait(10))
                        self.assertFalse(joined.done())
                    finally:
                        release.set()
                    response = joined.result(timeout=10)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "done")
                self.assertEqual(len(calls), 2)
                self.assertIsInstance(response.json()["processing_seconds"], (int, float))
            finally:
                release.set()

    def test_failure_is_visible_and_next_request_can_process(self):
        with tempfile.TemporaryDirectory() as data, patch.object(store, "DATA_DIR", data), \
                patch.object(api.processing, "process_session", side_effect=[
                    RuntimeError("reconstruction failed"), {"vertices": 42}
                ]), TestClient(api.app) as client:
            pid = client.post("/api/patients", json={"name": "Test"}).json()["id"]
            url = self.session(client, pid, "scan")
            failure = client.post(url + "/process?wait=true")
            self.assertEqual(failure.status_code, 500)
            self.assertEqual(client.get(url).json()["status"], "failed")
            self.assertIn("reconstruction failed", failure.json()["detail"])
            success = client.post(url + "/process?wait=true")
            self.assertEqual(success.status_code, 200)
            self.assertEqual(success.json()["status"], "done")


if __name__ == "__main__":
    unittest.main()
