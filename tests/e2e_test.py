"""End-to-end test: synthetic capture sessions through the live HTTP API.

Boots the server on a scratch data dir, generates three sessions (before,
after-with-known-bump, repeat-of-before), uploads them exactly the way the
iOS app will, processes, compares, and checks the measured volumes.

Run:  .venv/bin/python tests/e2e_test.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import open3d as o3d
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vectra3d import capture, io_session, synthetic  # noqa: E402

PORT = 8077
BASE = f"http://127.0.0.1:{PORT}"
PYTHON = os.path.join(ROOT, ".venv", "bin", "python")


def generate_sessions(work: str) -> float:
    """Write before/after/repeat session dirs; return ground-truth volume mm3."""
    rng = np.random.default_rng(42)
    sensor = capture.Sensor(rng)
    head = synthetic.make_head()
    bump = synthetic.cheek_bump_spec(head, amplitude_mm=2.0, sigma_mm=8.0)
    head_after = synthetic.apply_bump(head, bump)
    gt = synthetic.closed_mesh_volume(head_after) - synthetic.closed_mesh_volume(head)

    def write(name, mesh, transform=None):
        m = o3d.geometry.TriangleMesh(mesh)
        if transform is not None:
            m.transform(transform)
        poses = capture.capture_session(m, sensor, rng)
        color_frames = capture.capture_color_frames(m)
        io_session.write_session(os.path.join(work, name), poses, label=name,
                                 color_frames=color_frames)

    write("before", head)
    write("after", head_after, synthetic.random_session_transform(rng))
    write("repeat", head, synthetic.random_session_transform(rng))
    return gt


def upload_session(pid: str, label: str, session_dir: str) -> str:
    sid = requests.post(f"{BASE}/api/patients/{pid}/sessions",
                        json={"label": label}).json()["id"]
    files = [("files", (name, open(os.path.join(session_dir, name), "rb")))
             for name in sorted(os.listdir(session_dir))]
    r = requests.post(f"{BASE}/api/patients/{pid}/sessions/{sid}/upload", files=files)
    r.raise_for_status()
    r = requests.post(f"{BASE}/api/patients/{pid}/sessions/{sid}/process",
                      params={"wait": "true"}, timeout=300)
    r.raise_for_status()
    print(f"  processed {label}: {r.json()['stats']['vertices']} vertices")
    return sid


def main():
    work = tempfile.mkdtemp(prefix="vectra-e2e-")
    data_dir = os.path.join(work, "data")
    print("Generating synthetic sessions ...")
    gt = generate_sessions(work)
    print(f"  ground-truth bump: {gt / 1000:.3f} mL")

    # Force the TSDF path: the synthetic renders below aren't a valid Object
    # Capture input, and this test validates the depth-fusion measurement path.
    env = {**os.environ, "VECTRA_DATA_DIR": data_dir, "VECTRA_DISABLE_OC": "1"}
    server = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "app:app", "--port", str(PORT)],
        cwd=os.path.join(ROOT, "server"), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                requests.get(f"{BASE}/api/patients", timeout=1)
                break
            except requests.ConnectionError:
                time.sleep(0.2)
        else:
            raise RuntimeError("server did not start")

        print("Uploading + processing via HTTP ...")
        pid = requests.post(f"{BASE}/api/patients",
                            json={"name": "E2E Test"}).json()["id"]
        sids = {label: upload_session(pid, label, os.path.join(work, label))
                for label in ("before", "after", "repeat")}

        print("Comparing before -> after ...")
        bump_result = requests.post(
            f"{BASE}/api/patients/{pid}/compare",
            json={"before": sids["before"], "after": sids["after"]},
            timeout=600).json()
        print("Comparing before -> repeat (null) ...")
        null_result = requests.post(
            f"{BASE}/api/patients/{pid}/compare",
            json={"before": sids["before"], "after": sids["repeat"]},
            timeout=600).json()

        failures = []
        significant = [r for r in bump_result["regions"] if r["significant"]]
        if len(significant) < 1:
            failures.append("bump compare found no significant region")
        else:
            v = significant[0]["volume_ml"]
            print(f"  measured {v:+.3f} mL (truth {gt / 1000:.3f})")
            if abs(v - gt / 1000) > 0.45:
                failures.append(f"volume {v} too far from truth {gt / 1000:.3f}")
        null_sig = [r for r in null_result["regions"] if r["significant"]]
        if null_sig:
            failures.append(f"null compare reported significant regions: {null_sig}")

        cmp_id = bump_result["id"]
        for rel in (f"{pid}/sessions/{sids['before']}/mesh.ply",
                    f"{pid}/compares/{cmp_id}/heatmap.ply",
                    f"{pid}/compares/{cmp_id}/heatmap.png",
                    f"{pid}/compares/{cmp_id}/result.json"):
            if requests.get(f"{BASE}/files/{rel}").status_code != 200:
                failures.append(f"artifact missing: {rel}")

        # Normalized orientation: foremost vertex (+z) should be the nose,
        # i.e. near the vertical midline of the face.
        mesh = o3d.io.read_triangle_mesh(
            os.path.join(data_dir, pid, "sessions", sids["before"], "mesh.ply"))
        v = np.asarray(mesh.vertices)
        nose = v[np.argmax(v[:, 2])]
        if abs(nose[0]) > 25:
            failures.append(f"orientation suspect: foremost vertex at x={nose[0]:.1f}")

        if failures:
            print("\nE2E FAILED:")
            for f in failures:
                print("  -", f)
            sys.exit(1)
        print("\nE2E PASSED")
        print(f"  bump regions: {bump_result['regions']}")
        print(f"  null regions: {null_result['regions']}")
    finally:
        server.terminate()
        server.wait(timeout=10)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
