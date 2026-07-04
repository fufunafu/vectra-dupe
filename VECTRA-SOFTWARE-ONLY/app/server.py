"""Desktop app backend: FastAPI server that drives the reconstruction pipeline and
serves the 3D result + measurements to the web UI. Launched inside a native window
by run_app.py.

The reconstruction runs as three sequential SUBPROCESSES (recon -> mesh -> metric)
so PyTorch's OpenMP never shares a process with Open3D/COLMAP (which deadlocks).
"""
from __future__ import annotations
import os, sys, re, json, threading, subprocess, uuid

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUTPUTS = os.path.join(ROOT, "outputs")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PYEXE = sys.executable

app = FastAPI()
JOBS: dict[str, dict] = {}

# log marker -> (progress %, friendly message)
PROGRESS = [
    ("[preprocess] kept", 8, "Cleaned & masked photos"),
    ("[vggt] loading", 15, "Loading the AI model"),
    ("[vggt] preprocessing", 22, "Preparing views"),
    ("[vggt] running forward", 30, "AI reconstructing 3D (this is the slow part)"),
    ("[vggt] forward done", 58, "3D geometry recovered"),
    ("[recon] done", 66, "Saved point cloud"),
    ("[calibrate] pose graph", 70, "Calibrating camera poses"),
    ("[mesh] tsdf", 80, "Built surface mesh"),
    ("[mesh] alpha-shape", 80, "Built surface mesh"),
    ("[render] wrote", 90, "Rendered views"),
    ("[compare] wrote", 92, "Built comparison"),
    ("[metric] frontal", 95, "Detecting facial landmarks"),
    ("[metric] measurements", 99, "Computed measurements"),
]


def out_dir_for(visit: str) -> str:
    return os.path.join(OUTPUTS, visit.replace(" ", "_"))


def _run_stage(cmd: list[str], job: dict):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        line = line.rstrip()
        for marker, pct, msg in PROGRESS:
            if marker in line:
                job["progress"] = max(job["progress"], pct)
                job["message"] = msg
        job["log"].append(line)
        job["log"] = job["log"][-400:]
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"stage failed ({' '.join(cmd[2:4])}) rc={p.returncode}")


def _job_worker(job_id: str, visit: str):
    job = JOBS[job_id]
    visit_dir = os.path.join(DATA, visit)
    out_dir = out_dir_for(visit)
    try:
        job["status"], job["message"] = "running", "Starting…"
        _run_stage([PYEXE, "-m", "vectra_sw.recon_stage", visit_dir, out_dir], job)
        _run_stage([PYEXE, "-m", "vectra_sw.finish_vggt", visit_dir, out_dir], job)
        _run_stage([PYEXE, "-m", "vectra_sw.metric", out_dir], job)
        job["status"], job["progress"], job["message"] = "done", 100, "Complete"
    except Exception as e:        # noqa: BLE001
        job["status"], job["message"], job["error"] = "error", str(e), str(e)


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC, "index.html")) as f:
        # no-store: WKWebView (the pywebview window) otherwise keeps serving a
        # stale cached page across app relaunches
        return HTMLResponse(f.read(), headers={"Cache-Control": "no-store"})


@app.get("/api/visits")
def visits():
    items = []
    if os.path.isdir(DATA):
        for name in sorted(os.listdir(DATA)):
            d = os.path.join(DATA, name)
            if not os.path.isdir(d):
                continue
            n_photos = len([f for f in os.listdir(d) if f.upper().endswith(".JPG")])
            if n_photos == 0:
                continue
            od = out_dir_for(name)
            items.append({
                "name": name, "photos": n_photos,
                "has_mesh": os.path.exists(os.path.join(od, "mesh.ply")),
                "has_measures": os.path.exists(os.path.join(od, "measurements.json")),
            })
    return items


@app.post("/api/reconstruct")
async def reconstruct(payload: dict):
    visit = payload["visit"]
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"id": job_id, "visit": visit, "status": "queued",
                    "progress": 0, "message": "Queued", "log": [], "error": None}
    threading.Thread(target=_job_worker, args=(job_id, visit), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    return JOBS.get(job_id, {"status": "unknown"})


@app.get("/api/result/{visit}/measurements")
def measurements(visit: str):
    p = os.path.join(out_dir_for(visit), "measurements.json")
    if not os.path.exists(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    with open(p) as f:
        return json.load(f)


@app.get("/api/result/{visit}/mesh.ply")
def mesh(visit: str):
    p = os.path.join(out_dir_for(visit), "mesh.ply")
    if not os.path.exists(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="application/octet-stream")


@app.get("/api/result/{visit}/mesh_textured.glb")
def mesh_textured(visit: str):
    p = os.path.join(out_dir_for(visit), "mesh_textured.glb")
    if not os.path.exists(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="model/gltf-binary")


@app.get("/api/result/{visit}/comparison.png")
def comparison(visit: str):
    p = os.path.join(out_dir_for(visit), "comparison.png")
    if not os.path.exists(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="image/png")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
