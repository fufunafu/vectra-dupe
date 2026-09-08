#!/usr/bin/env python3
"""Read-only tap of a running reconstruction, retaining its diagnostic assets.

The processing worker may remove its scratch output immediately after failure.
This helper copies completed models to a private temporary directory; it never
edits the running job or its input. Stop after one snapshot or a bounded timeout.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time


def retain(session: Path, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for work in session.glob("oc_*"):
            if not work.is_dir() or not (work / "oc_render.png").is_file():
                continue
            names = ["model.obj", "model.mtl", "oc_render.png"]
            names += [p.name for p in work.glob("*_diffuseColor.png")]
            if len(names) < 4 or not all((work / name).is_file() for name in names):
                continue
            target = Path(tempfile.mkdtemp(prefix="vectra-retained-"))
            try:
                for name in names:
                    shutil.copy2(work / name, target / name)
                provenance = {"source_session_id": session.name,
                              "raw_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in (session / "raw").iterdir() if p.is_file()}}
                (target / "provenance.json").write_text(json.dumps(provenance))
                return target
            except FileNotFoundError:
                # The worker finished while we were copying. A later attempt
                # may produce another complete model; originals stay untouched.
                continue
        time.sleep(1)
    raise TimeoutError("No completed intermediate reconstruction was available")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    print(retain(args.session_dir, args.timeout), flush=True)
