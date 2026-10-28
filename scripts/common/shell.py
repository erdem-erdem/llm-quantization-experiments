"""Subprocess runner: tees output to a log, never uses shell=True."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Sequence

logger = logging.getLogger("shell")


def run(cmd: Sequence[str], *, log_path: Path | None = None, cwd: Path | None = None,
        check: bool = True, timeout: float | None = None,
        env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    logger.info("run: %s", " ".join(map(str, cmd)))
    proc = subprocess.run(
        [str(c) for c in cmd], cwd=None if cwd is None else Path(cwd),
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"\n$ {' '.join(map(str, cmd))}\nrc={proc.returncode}\n")
            f.write(proc.stdout)
            if proc.stderr:
                f.write("\n-- stderr --\n" + proc.stderr)
    if check and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-4000:]
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(map(str, cmd))}\n{tail}")
    return proc