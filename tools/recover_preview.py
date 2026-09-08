#!/usr/bin/env python3
"""Recover only a COPY's display from its retained Object Capture diagnostic.

Never operates on an original session, never changes raw inputs or mesh.ply,
and never promotes a prettier display to measurement eligibility.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

from vectra3d import fuse, io_session, photogrammetry
import processing
import store
from tools.reprocess_copy import fingerprint


def recover(pid, sid, *, reconstruct=False):
    meta = store.get_session_meta(pid, sid)
    source_id = meta.get("source_session_id")
    if not source_id or source_id == sid:
        raise ValueError("Recovery requires a separate session made by reprocess_copy.py")
    if meta.get("status") in ("processing", "queued"):
        raise ValueError("Copy is still processing")
    target = Path(store.session_dir(pid, sid))
    source = Path(store.session_dir(pid, source_id))
    source_before, raw_before = fingerprint(source), fingerprint(source / "raw")
    if fingerprint(target / "raw") != raw_before:
        raise ValueError("Copy's raw files no longer match the source capture")
    measurement_path = target / "mesh.ply"
    measurement_before = processing.quality.mesh_fingerprint(str(measurement_path))
    with (target / "stats.json").open() as stream:
        stats = json.load(stream)
    obj = target / "oc_debug" / "model.obj"
    if not obj.is_file() and not reconstruct:
        raise ValueError("No retained Object Capture diagnostic model")
    started = time.monotonic()
    store.update_session_meta(pid, sid, status="processing", error=None)
    try:
        poses, colors, _ = io_session.load_session(str(target / "raw"))
        extrinsics = fuse.view_extrinsics(poses)
        diagnostics = {}
        oc = photogrammetry._reconstruct_metric_once(
            str(target / "raw"), poses, colors, str(target), diag=diagnostics,
            extrinsics=extrinsics, cached_obj_path=str(obj) if obj.is_file() else None)
        vertex_ok, textured_ok = processing._write_oc_display_meshes(oc, str(target), "both")
        if not textured_ok:
            raise RuntimeError("Recovered model could not be exported with its texture")
        if processing.quality.mesh_fingerprint(str(measurement_path)) != measurement_before:
            raise RuntimeError("Measurement mesh changed during preview recovery")
        if fingerprint(source) != source_before or fingerprint(source / "raw") != raw_before:
            raise RuntimeError("Source changed concurrently; inspect before accepting recovery")
        stats.update({key: value for key, value in oc.stats.items() if key != "reconstruction"})
        if not stats.get("measurement_quality"):
            import open3d as o3d
            report = processing.quality.inspect_mesh(processing.crop_to_face(
                o3d.io.read_triangle_mesh(str(measurement_path))))
            report["mesh_sha256"] = measurement_before
            stats.update(measurement_quality=report, measurement_grade=report["eligible"],
                         has_measurement_mesh=True,
                         measurement_warning="Engineering checks only; real-patient accuracy unvalidated.")
        stats.pop("oc_error", None)
        stats.update(display_source="object_capture", textured=vertex_ok,
                     has_textured_glb=textured_ok,
                     display_quality={"status": "ready", "reason": ""},
                     preview_recovery=diagnostics)
        with (target / "stats.json").open("w") as stream:
            json.dump(stats, stream, indent=2)
        return store.update_session_meta(
            pid, sid, processed=True, status="done", stats=stats, error=None,
            source_files_unchanged=True, measurement_mesh_unchanged=True,
            processing_seconds=round(time.monotonic() - started, 2))
    except Exception as exc:
        store.update_session_meta(pid, sid, status="failed", error=str(exc),
                                  processing_seconds=round(time.monotonic() - started, 2))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patient_id")
    parser.add_argument("copy_session_id")
    args = parser.parse_args()
    print(json.dumps(recover(args.patient_id, args.copy_session_id), indent=2), flush=True)
