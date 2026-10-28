"""Job spec builders (plan §7): every exp/e*.py emits these; queue_gen + scheduler consume."""
from __future__ import annotations

from typing import Any


SOFT_PREFIX = "attempted:"


def dep_id(dep: str) -> str:
    """Job id a dependency refers to ('attempted:X' is a soft dependency on X)."""
    return dep[len(SOFT_PREFIX):] if dep.startswith(SOFT_PREFIX) else dep


def job(jid: str, exp: str, kind: str, cmd: list[str], *, deps: list[str] | None = None,
        outputs: list[str] | None = None, est_min: float = 0.0,
        gpu_affinity: int | None = None, validator: str | None = None,
        gpu: int | None = None) -> dict[str, Any]:
    if kind not in ("gpu", "cpu"):
        raise ValueError(kind)
    if kind == "gpu" and gpu_affinity is not None:
        cmd = _pin_gpu(cmd, gpu_affinity)
    job_name, key = _marker(cmd)
    return {
        "id": jid, "exp": exp, "kind": kind, "cmd": cmd,
        "deps": list(deps or ()), "outputs": list(outputs or ()),
        "est_min": float(est_min), "gpu_affinity": gpu_affinity,
        "validator": validator, "gpu": gpu, "job_name": job_name, "key": key,
    }


def _pin_gpu(cmd: list[str], gpu: int) -> list[str]:
    """Hand the GPU to record_run (--gpu) so gpu_budget.csv books GPU-minutes and the E4
    exclusivity check runs; the scheduler only sets CUDA_VISIBLE_DEVICES."""
    if "--gpu" in cmd or "--" not in cmd:
        return cmd
    i = cmd.index("--")
    return cmd[:i] + ["--gpu", str(gpu)] + cmd[i:]


def _marker(cmd: list[str]) -> tuple[str | None, str | None]:
    """(job_name, key) from a record_run wrapper: [--exp E --job J --key K] positions."""
    try:
        i = next(j for j, t in enumerate(cmd) if t.endswith("record_run.py"))
        if cmd[i + 1] == "--exp" and cmd[i + 3] == "--job" and cmd[i + 5] == "--key":
            return cmd[i + 4], cmd[i + 6]
    except StopIteration:
        pass
    return None, None


def record_cmd(exp: str, job_name: str, key: str, inner: list[str], *, gpu: int | None = None,
               validator: str | None = None) -> list[str]:
    """Wrap an inner command in record_run.py so .done/run_meta/budget are handled."""
    cmd = [sys_exec(), "scripts/record_run.py", "--exp", exp, "--job", job_name,
           "--key", key]
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    if validator:
        cmd += ["--validator", validator]
    return cmd + ["--"] + inner


def sys_exec() -> str:
    import sys
    return sys.executable


def script(name: str) -> str:
    """Repo-relative path of a script under scripts/."""
    return f"scripts/{name}.py"