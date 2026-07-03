#!/usr/bin/env python3
"""Reprocess every on-disk session through the hybrid pipeline.

Walks `server/data/<patient>/sessions/<session>/raw/` and runs
`processing.process_session(raw, session_dir)` for each, writing mesh.ply +
display GLBs back into the session dir (where the server serves them from). Use
this after any server-side change — the running uvicorn holds stale code, and old
sessions keep their old meshes until reprocessed.

Usage:
    PYTHONPATH=. .venv/bin/python tools/reprocess_all.py [DATA_DIR]
    VECTRA_DISABLE_OC=1 ... tools/reprocess_all.py     # force TSDF display
    VECTRA_OC_DETAIL=reduced ... tools/reprocess_all.py  # faster OC

Prints a per-session summary table: which got the Object Capture photoreal
display vs fell back to TSDF, plus the OC alignment diagnostics.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))
sys.path.insert(0, ROOT)

import processing  # noqa: E402


def find_sessions(data_dir: str) -> list[str]:
    """All raw/ dirs under data_dir, sorted."""
    out = []
    for dirpath, dirnames, _ in os.walk(data_dir):
        if os.path.basename(dirpath) == "raw":
            out.append(dirpath)
            dirnames[:] = []  # don't descend into a raw/ dir
    return sorted(out)


def main() -> int:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "server", "data")
    raw_dirs = find_sessions(data_dir)
    if not raw_dirs:
        print(f"no sessions found under {data_dir}")
        return 1

    print(f"reprocessing {len(raw_dirs)} session(s) under {data_dir}\n")
    rows = []
    for raw in raw_dirs:
        session_dir = os.path.dirname(raw)
        name = os.path.relpath(session_dir, data_dir)
        t0 = time.time()
        try:
            s = processing.process_session(raw, session_dir)
            rows.append((name, s, time.time() - t0, None))
            disp = s.get("display_source", "?")
            extra = ""
            if disp == "object_capture":
                extra = (f"  rms={s.get('align_rms_mm')}mm ipd={s.get('align_ipd_mm')}mm "
                         f"pass={s.get('oc_attempts_passed')}/{s.get('oc_attempts')}")
            elif s.get("oc_error"):
                extra = f"  oc_error: {s['oc_error']}"
            print(f"  OK   {name}  display={disp}  "
                  f"verts={s.get('vertices')} tex={s.get('has_textured_glb')}{extra}  "
                  f"({time.time() - t0:.0f}s)")
        except Exception as e:  # noqa: BLE001
            rows.append((name, None, time.time() - t0, str(e)))
            print(f"  FAIL {name}  {e}")

    ok = [r for r in rows if r[3] is None]
    oc = [r for r in ok if r[1].get("display_source") == "object_capture"]
    print(f"\n{len(ok)}/{len(rows)} processed; "
          f"{len(oc)} with Object Capture display, {len(ok) - len(oc)} TSDF display.")
    return 0 if len(ok) == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
