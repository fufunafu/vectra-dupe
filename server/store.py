"""Filesystem-backed storage for patients, sessions, and comparisons.

Layout under DATA_DIR:
    patients.json
    <patient_id>/sessions/<session_id>/raw/...      uploaded capture files
    <patient_id>/sessions/<session_id>/mesh.ply     processed surface
    <patient_id>/sessions/<session_id>/stats.json
    <patient_id>/sessions/<session_id>/meta.json
    <patient_id>/compares/<before>__<after>/        result.json, heatmap.ply/png
"""

import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone

DATA_DIR = os.environ.get(
    "VECTRA_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "item"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _patients_file() -> str:
    return os.path.join(DATA_DIR, "patients.json")


def list_patients() -> dict:
    if not os.path.exists(_patients_file()):
        return {}
    with open(_patients_file()) as f:
        return json.load(f)


def create_patient(name: str) -> dict:
    patients = list_patients()
    pid = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
    patients[pid] = {"id": pid, "name": name, "created_at": _now()}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(_patients_file(), "w") as f:
        json.dump(patients, f, indent=2)
    return patients[pid]


def patient_dir(pid: str) -> str:
    path = os.path.join(DATA_DIR, pid)
    if pid not in list_patients():
        raise KeyError(f"unknown patient {pid}")
    return path


def create_session(pid: str, label: str) -> dict:
    sid = f"{_slug(label)}-{uuid.uuid4().hex[:6]}"
    sdir = os.path.join(patient_dir(pid), "sessions", sid)
    os.makedirs(os.path.join(sdir, "raw"), exist_ok=True)
    meta = {"id": sid, "label": label, "created_at": _now(),
            "processed": False, "status": "new"}
    with open(os.path.join(sdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def session_dir(pid: str, sid: str) -> str:
    sdir = os.path.join(patient_dir(pid), "sessions", sid)
    if not os.path.isdir(sdir):
        raise KeyError(f"unknown session {sid}")
    return sdir


def get_session_meta(pid: str, sid: str) -> dict:
    sdir = session_dir(pid, sid)  # raises KeyError if unknown
    with open(os.path.join(sdir, "meta.json")) as f:
        return json.load(f)


def update_session_meta(pid: str, sid: str, **updates) -> dict:
    sdir = session_dir(pid, sid)
    with open(os.path.join(sdir, "meta.json")) as f:
        meta = json.load(f)
    meta.update(updates)
    with open(os.path.join(sdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def list_sessions(pid: str) -> list[dict]:
    root = os.path.join(patient_dir(pid), "sessions")
    sessions = []
    if os.path.isdir(root):
        for sid in sorted(os.listdir(root)):
            meta_path = os.path.join(root, sid, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    sessions.append(json.load(f))
    return sessions


def delete_session(pid: str, sid: str) -> None:
    """Remove a session directory and any comparison that referenced it."""
    sdir = session_dir(pid, sid)  # raises KeyError if unknown
    shutil.rmtree(sdir, ignore_errors=True)
    compares_root = os.path.join(patient_dir(pid), "compares")
    if os.path.isdir(compares_root):
        for name in os.listdir(compares_root):
            before, _, after = name.partition("__")
            if sid in (before, after):
                shutil.rmtree(os.path.join(compares_root, name), ignore_errors=True)


def compare_dir(pid: str, before_sid: str, after_sid: str) -> str:
    return os.path.join(patient_dir(pid), "compares", f"{before_sid}__{after_sid}")


def list_compares(pid: str) -> list[dict]:
    root = os.path.join(patient_dir(pid), "compares")
    results = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            result_path = os.path.join(root, name, "result.json")
            if os.path.exists(result_path):
                before_sid, _, after_sid = name.partition("__")
                with open(result_path) as f:
                    results.append({"before": before_sid, "after": after_sid,
                                    "id": name, **json.load(f)})
    return results
