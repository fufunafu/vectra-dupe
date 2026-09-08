#!/usr/bin/env python3
"""Reconstruct a copy as a new session, keeping the source session intact.

Run while the local server's processing queue is idle:
  .venv/bin/python tools/reprocess_copy.py PATIENT_ID SESSION_ID --label LABEL

This standalone process uses the current pipeline code. It does not share the
API's in-memory queue. Existing session files are never processing targets.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

import processing
import store


def fingerprint(directory):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.iterdir() if p.is_file()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patient_id")
    parser.add_argument("session_id")
    parser.add_argument("--label", required=True)
    parser.add_argument("--preview-only", action="store_true",
                        help="Copy the existing measurement mesh and rebuild only the display")
    parser.add_argument("--retained-dir", type=Path,
                        help="Reuse diagnostic assets retained from this source capture")
    args = parser.parse_args()
    source = Path(store.session_dir(args.patient_id, args.session_id))
    if store.get_session_meta(args.patient_id, args.session_id).get("status") in ("queued", "processing"):
        parser.error("source session is still processing")
    if not (source / "raw" / "session.json").is_file():
        parser.error("source session has no raw capture")
    original = fingerprint(source)
    original_raw = fingerprint(source / "raw")
    if args.retained_dir:
        if not args.preview_only:
            parser.error("--retained-dir requires --preview-only")
        provenance = json.loads((args.retained_dir / "provenance.json").read_text())
        if (provenance.get("source_session_id") != args.session_id
                or provenance.get("raw_sha256") != original_raw):
            parser.error("retained reconstruction does not match this source capture")
    created = store.create_session(args.patient_id, args.label)
    sid = created["id"]
    target = Path(store.session_dir(args.patient_id, sid))
    print(json.dumps({"new_session": sid, "directory": str(target),
                      "source_session": args.session_id}), flush=True)
    started = time.monotonic()
    try:
        shutil.copytree(source / "raw", target / "raw", dirs_exist_ok=True)
        if args.preview_only:
            for name in ("mesh.ply", "stats.json"):
                shutil.copy2(source / name, target / name)
            if args.retained_dir:
                debug = target / "oc_debug"
                debug.mkdir()
                for item in args.retained_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, debug / item.name)
            store.update_session_meta(args.patient_id, sid, source_session_id=args.session_id)
            from tools.recover_preview import recover
            result = recover(args.patient_id, sid, reconstruct=True)
            print(json.dumps(result, indent=2), flush=True)
            return 0
        store.update_session_meta(args.patient_id, sid, status="processing",
                                  source_session_id=args.session_id)
        stats = processing.process_session(str(target / "raw"), str(target))
        if stats.get("display_source") != "object_capture" or not stats.get("has_textured_glb"):
            raise RuntimeError("Textured Object Capture output unavailable; inspect new session diagnostics")
        if fingerprint(source) != original or fingerprint(source / "raw") != original_raw:
            raise RuntimeError("Source session changed during processing; inspect concurrent activity")
        result = store.update_session_meta(
            args.patient_id, sid, status="done", processed=True, error=None,
            stats=stats, patient_id=stats.get("patient_id", ""),
            processing_seconds=round(time.monotonic() - started, 2),
            source_files_unchanged=True)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except Exception as exc:
        store.update_session_meta(args.patient_id, sid, status="failed", error=str(exc),
                                  processing_seconds=round(time.monotonic() - started, 2))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
