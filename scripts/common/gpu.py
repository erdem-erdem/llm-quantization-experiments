"""GPU hygiene helpers (nvidia-smi based; every function degrades to "unknown" without it).

The scheduler runs one job per GPU at a time. A job that exits can still leave GPU memory
behind (an orphaned vLLM engine process, a llama-server child), which would make the NEXT job on
that card OOM or, for E4, measure a contended GPU. `wait_free` checks the physical card and, if
needed, kills the compute processes still on it before the next job starts.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import time

FREE_MIB = 1024          # an idle H100 context is a few MiB; > 1 GiB means something is still there
WAIT_NATURAL_S = 120.0   # let the driver release memory after a normal exit
WAIT_AFTER_KILL_S = 60.0
POLL_S = 3.0

log = logging.getLogger("gpu")


def _smi(args: list[str]) -> str | None:
    try:
        p = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return p.stdout if p.returncode == 0 else None


def memory_used_mib(gpu: int) -> int | None:
    """Used memory (MiB) of physical GPU `gpu`, or None when nvidia-smi is unavailable."""
    out = _smi(["-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if out is None or not out.strip():
        return None
    try:
        return int(float(out.strip().splitlines()[0]))
    except ValueError:
        return None


def compute_pids(gpu: int) -> list[int]:
    """PIDs of compute processes currently on physical GPU `gpu`."""
    out = _smi(["-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"])
    pids = []
    for line in (out or "").splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def snapshot(gpu: int) -> tuple[int | None, frozenset[int]]:
    """(memory MiB, compute PIDs) of a card, taken before any job runs: the baseline that is
    NOT ours (desktop session, another user's process) and must never be waited on or killed."""
    return memory_used_mib(gpu), frozenset(compute_pids(gpu))


def kill_pids(pids: list[int]) -> list[int]:
    """SIGKILL the given PIDs (never ourselves; only processes we own). Returns those killed."""
    killed = []
    sig = getattr(signal, "SIGKILL", signal.SIGTERM)  # Windows has no SIGKILL
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            if hasattr(os, "getuid") and os.stat(f"/proc/{pid}").st_uid != os.getuid():
                continue  # someone else's process: not ours to kill
            os.kill(pid, sig)
            killed.append(pid)
        except (ProcessLookupError, PermissionError, FileNotFoundError, OSError):
            continue
    return killed


def wait_free(gpu: int, why: str, free_mib: int = FREE_MIB,
              natural_s: float = WAIT_NATURAL_S, after_kill_s: float = WAIT_AFTER_KILL_S,
              sleep=time.sleep, baseline_mib: int = 0,
              keep_pids: frozenset[int] = frozenset()) -> tuple[bool, int | None]:
    """Block until GPU `gpu` holds <= baseline_mib + free_mib. Waits for a natural release, then
    kills the compute processes that appeared since the baseline (never `keep_pids`) and waits
    again. Returns (is_free, last_used_mib); (True, None) when memory cannot be read at all
    (no nvidia-smi: nothing to enforce)."""
    free_mib = baseline_mib + free_mib
    used = memory_used_mib(gpu)
    if used is None:
        return True, None
    waited = 0.0
    while used > free_mib and waited < natural_s:
        sleep(POLL_S)
        waited += POLL_S
        used = memory_used_mib(gpu)
        if used is None:
            return True, None
    if used > free_mib:
        pids = [p for p in compute_pids(gpu) if p not in keep_pids]
        killed = kill_pids(pids)
        log.warning("GPU %s still holds %s MiB (%s): killed leftover compute pids %s of %s",
                    gpu, used, why, killed, pids)
        waited = 0.0
        while used > free_mib and waited < after_kill_s:
            sleep(POLL_S)
            waited += POLL_S
            used = memory_used_mib(gpu)
            if used is None:
                return True, None
    return used <= free_mib, used
