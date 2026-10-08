"""Run one pilot command once: skip it if it already finished in an earlier Colab session.

    !python curriculum/pilots/once.py P3 2 'python labs/... --out "/content/drive/.../x.json" 2>&1 | tail -40'

A command is done when its marker PILOT_DIR/.done/<pilot>-<n>-<hash of the command> exists; a failed one
leaves PILOT_DIR/.failed/<same key> until it succeeds. Outputs written before markers
existed count as done too: a non-empty `tee` target without a traceback, or an existing `--out` .json file.
Unfinished training runs resume by themselves when rerun (their scripts skip finished runs).
Set PILOT_DIR in the environment (the notebook's first cell does).

Planning file — not imported into the Academy.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path


def legacy_done(cmd: str) -> bool:
    m = re.search(r'\|\s*tee\s+"([^"]+)"', cmd)
    if m:
        p = Path(m.group(1))
        if p.is_file() and p.stat().st_size > 0:
            text = p.read_text(errors="replace")
            return bool(text.strip()) and "Traceback" not in text
    m = re.search(r'--out\s+"([^"]+\.json)"', cmd)
    return bool(m) and Path(m.group(1)).is_file() and Path(m.group(1)).stat().st_size > 0


def main() -> int:
    pilot, idx, cmd = sys.argv[1], sys.argv[2], sys.argv[3]
    key = f"{pilot}-{idx}-{hashlib.sha1(cmd.encode()).hexdigest()[:8]}"
    marker = Path(os.environ["PILOT_DIR"]) / ".done" / key
    failed = Path(os.environ["PILOT_DIR"]) / ".failed" / key
    marker.parent.mkdir(parents=True, exist_ok=True)
    failed.parent.mkdir(parents=True, exist_ok=True)
    short = cmd if len(cmd) < 110 else cmd[:107] + "..."
    if marker.exists() or legacy_done(cmd):
        marker.touch()
        print(f"[{pilot}-{idx}] already done, skipped: {short}")
        return 0
    print(f"[{pilot}-{idx}] running: {short}", flush=True)
    rc = subprocess.run(["bash", "-o", "pipefail", "-c", cmd]).returncode
    if rc == 0:
        marker.touch()
        failed.unlink(missing_ok=True)
        print(f"[{pilot}-{idx}] done")
    else:
        failed.touch()
        print(f"[{pilot}-{idx}] FAILED (exit {rc}); rerun the cell to retry")
    return rc


if __name__ == "__main__":
    sys.exit(main())
