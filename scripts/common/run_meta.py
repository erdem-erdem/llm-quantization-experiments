"""Environment facts for run_meta.json (spec §2.2)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from . import paths


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=paths.ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        h = hashlib.sha256()
        for p in sorted(paths.ROOT.rglob("*.py")):
            h.update(p.name.encode())
            h.update(p.read_bytes())
        return f"nogit:{h.hexdigest()[:16]}"


def pip_freeze_sha256() -> str:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        )
        return hashlib.sha256(out.stdout.encode()).hexdigest()
    except Exception:
        return "unavailable"


def llamacpp_commit() -> str | None:
    p = paths.ENV / "llamacpp_commit.txt"
    if not p.exists():
        return None
    line = p.read_text(encoding="utf-8").strip().splitlines()
    for ln in line:
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            return ln
    return None


def nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def cuda_versions() -> dict[str, str | None]:
    driver = nvidia_smi("driver_version")
    cuda = None
    try:
        import torch
        cuda = torch.version.cuda
    except Exception:
        pass
    return {"driver_version": driver, "cuda_version": cuda}


def gpu_minutes(t0: datetime, t1: datetime, n_gpus: int) -> float:
    return (t1 - t0).total_seconds() / 60.0 * n_gpus


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_meta(exp: str, job: str, run_key: str, cmd: list[str], t0: datetime, t1: datetime,
               returncode: int, n_gpus: int, seed: str | None = None,
               extras: dict[str, Any] | None = None) -> dict[str, Any]:
    cuda = cuda_versions()
    meta = {
        "exp": exp, "job": job, "run_key": run_key,
        "git_commit": git_commit(), "pip_freeze_sha256": pip_freeze_sha256(),
        "llamacpp_commit": llamacpp_commit(),
        "gpu_name": nvidia_smi("name"), "gpu_memory_total": nvidia_smi("memory.total"),
        "gpu_power_limit": nvidia_smi("power.max_limit"),
        "driver_version": cuda["driver_version"], "cuda_version": cuda["cuda_version"],
        "seed": seed, "cmd": cmd,
        "start": t0.isoformat(timespec="seconds"), "end": t1.isoformat(timespec="seconds"),
        "gpu_minutes": round(gpu_minutes(t0, t1, n_gpus), 2),
        "returncode": returncode, "status": "ok" if returncode == 0 else "fail",
        "extras": extras or {},
    }
    return meta