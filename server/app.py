"""Vectra-dupe processing server.

Run:  ../.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8008
(from the server/ directory; the iOS app and web viewer talk to this)
"""

import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import processing
import store
from jobs import ProcessingQueue


@asynccontextmanager
async def lifespan(app):
    app.state.processing_queue = ProcessingQueue()
    try:
        yield
    finally:
        app.state.processing_queue.shutdown()


app = FastAPI(title="vectra-dupe", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class PatientIn(BaseModel):
    name: str


class SessionIn(BaseModel):
    label: str


class CompareIn(BaseModel):
    before: str
    after: str


@app.get("/api/patients")
def get_patients():
    return list(store.list_patients().values())


@app.post("/api/patients")
def post_patient(body: PatientIn):
    return store.create_patient(body.name)


@app.get("/api/patients/{pid}/sessions")
def get_sessions(pid: str):
    try:
        return store.list_sessions(pid)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/api/patients/{pid}/sessions")
def post_session(pid: str, body: SessionIn):
    try:
        return store.create_session(pid, body.label)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/api/patients/{pid}/sessions/{sid}/upload")
def upload_files(pid: str, sid: str, files: list[UploadFile]):
    try:
        raw_dir = os.path.join(store.session_dir(pid, sid), "raw")
    except KeyError as e:
        raise HTTPException(404, str(e))
    saved = []
    for f in files:
        name = os.path.basename(f.filename or "")
        if not name:
            continue
        with open(os.path.join(raw_dir, name), "wb") as out:
            out.write(f.file.read())
        saved.append(name)
    return {"saved": saved}


@app.delete("/api/patients/{pid}/sessions/{sid}")
def delete_session(pid: str, sid: str):
    try:
        store.delete_session(pid, sid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"deleted": sid}


@app.get("/api/patients/{pid}/sessions/{sid}")
def get_session(pid: str, sid: str):
    try:
        return store.get_session_meta(pid, sid)
    except KeyError as e:
        raise HTTPException(404, str(e))


def _process_core(pid: str, sid: str, raw_dir: str, sdir: str, mode: str) -> dict:
    """Run the pipeline and record success in the session meta. Raises on error."""
    stats = processing.process_session(raw_dir, sdir, texture_mode=mode)
    return store.update_session_meta(
        pid, sid, processed=True, status="done", stats=stats,
        patient_id=stats.get("patient_id", ""), error=None)


def _run_processing(pid: str, sid: str, raw_dir: str, sdir: str, mode: str) -> dict:
    """Run one queued job, persisting its status, duration, and any failure."""
    started = time.monotonic()
    try:
        store.update_session_meta(pid, sid, status="processing", error=None)
        _process_core(pid, sid, raw_dir, sdir, mode)
    except Exception as e:
        return store.update_session_meta(
            pid, sid, status="failed", error=str(e),
            processing_seconds=round(time.monotonic() - started, 2))
    return store.update_session_meta(
        pid, sid, processing_seconds=round(time.monotonic() - started, 2))


@app.post("/api/patients/{pid}/sessions/{sid}/process")
def process_session(pid: str, sid: str, mode: str = "both", wait: bool = False):
    """Kick off processing. By default it runs in the background and returns
    immediately with status="queued" or "processing"; clients poll the session
    until status is "done"/"failed" (a dense capture's projection can take
    minutes, longer than any sane HTTP timeout). Pass ?wait=true to block and
    return the finished meta (used by the e2e test)."""
    if mode not in ("vertex", "cylindrical", "both"):
        raise HTTPException(400, f"unknown texture mode: {mode}")
    try:
        sdir = store.session_dir(pid, sid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    raw_dir = os.path.join(sdir, "raw")
    if not os.path.exists(os.path.join(raw_dir, "session.json")):
        raise HTTPException(400, "session.json not uploaded yet")
    future = app.state.processing_queue.submit(
        (pid, sid),
        lambda: _run_processing(pid, sid, raw_dir, sdir, mode),
        on_queued=lambda: store.update_session_meta(
            pid, sid, status="queued", error=None, processing_seconds=None))
    if wait:
        result = future.result()
        if result["status"] == "failed":
            raise HTTPException(500, f"processing failed: {result['error']}")
        return result
    return store.get_session_meta(pid, sid)


@app.post("/api/patients/{pid}/compare")
def compare(pid: str, body: CompareIn):
    try:
        before_mesh = os.path.join(store.session_dir(pid, body.before), "mesh.ply")
        after_mesh = os.path.join(store.session_dir(pid, body.after), "mesh.ply")
    except KeyError as e:
        raise HTTPException(404, str(e))
    for path, sid in ((before_mesh, body.before), (after_mesh, body.after)):
        if not os.path.exists(path):
            # A photo-only (rear, no-LiDAR) session processes fine but never
            # writes a measurement mesh — tell the caller why, not just "wait".
            detail = f"session {sid} is not processed yet"
            stats_path = os.path.join(os.path.dirname(path), "stats.json")
            try:
                with open(stats_path) as f:
                    if json.load(f).get("measurement_grade") is False:
                        detail = (f"session {sid} is photo-only (no depth) — "
                                  "it has no measurement mesh to compare")
            except (OSError, json.JSONDecodeError):
                pass
            raise HTTPException(400, detail)
    out_dir = store.compare_dir(pid, body.before, body.after)
    try:
        summary = processing.compare_sessions_on_disk(before_mesh, after_mesh, out_dir)
    except Exception as e:
        raise HTTPException(500, f"comparison failed: {e}")
    return {"before": body.before, "after": body.after,
            "id": f"{body.before}__{body.after}", **summary}


@app.get("/api/patients/{pid}/compares")
def get_compares(pid: str):
    try:
        return store.list_compares(pid)
    except KeyError as e:
        raise HTTPException(404, str(e))


os.makedirs(store.DATA_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=store.DATA_DIR), name="files")
app.mount("/", StaticFiles(
    directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"),
    html=True), name="static")
