"""Job wrapper: .done markers, GPU pinning, run_meta.json, gpu_budget.csv (plan §5.5).

Usage:
  python scripts/record_run.py --exp E1 --job eval --key q8__gptq__tr__s0 \
      [--gpu 2] [--n-gpus 1] [--seed s0] [--force] \
      [--validator module:function] -- <command...>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import io, logging_setup, paths, run_meta  # noqa: E402


def import_validator(spec: str):
    mod_name, fn_name = spec.split(":")
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--n-gpus", type=int, default=None,
                    help="GPUs pinned by this job (default: 1 if --gpu given else 0)")
    ap.add_argument("--seed", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--validator", default=None, help="module:function -> validate_outputs(...) -> bool")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = [c for c in args.cmd if c != "--"] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("no command given after --")

    logger = logging_setup.setup("record_run")
    marker = paths.done_marker(args.exp, args.job, args.key)
    if marker.exists() and not args.force:
        logger.info("done marker exists, skipping: %s", marker)
        return 0

    if marker.exists():
        marker.unlink()  # --force: invalidate before rerun

    n_gpus = args.n_gpus if args.n_gpus is not None else (1 if args.gpu is not None else 0)
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if args.job == "speed":  # E4 requires an exclusive GPU
            idle = _gpu_idle(args.gpu)
            if not idle:
                logger.error("GPU %s is not idle; E4 requires exclusivity", args.gpu)
                return 1

    t0 = datetime.now(timezone.utc)
    wall0 = time.monotonic()
    proc = subprocess.run(cmd, env=env)
    t1 = datetime.now(timezone.utc)
    wall1 = time.monotonic()

    status = "ok" if proc.returncode == 0 else "fail"
    if proc.returncode == 0 and args.validator:
        try:
            fn = import_validator(args.validator)
            ok = fn(args)
            if not ok:
                status = "fail"
                logger.error("validator returned False for %s/%s/%s", args.exp, args.job, args.key)
        except Exception as exc:  # noqa: BLE001
            status = "fail"
            logger.exception("validator raised: %s", exc)

    meta = run_meta.build_meta(
        exp=args.exp, job=args.job, run_key=args.key, cmd=cmd, t0=t0, t1=t1,
        returncode=proc.returncode if status == "ok" else 1,
        n_gpus=n_gpus, seed=args.seed,
        extras={"wall_minutes": round((wall1 - wall0) / 60.0, 2)},
    )
    io.atomic_write_json(paths.run_meta_path(args.exp, args.key, args.job), meta)

    row = {
        "ts": t1.isoformat(timespec="seconds"),
        "exp": args.exp, "job": args.job, "run_key": args.key,
        "gpu": args.gpu, "wall_min": meta["extras"]["wall_minutes"],
        "gpu_min": meta["gpu_minutes"],
        "cmd_hash": hashlib.sha256(" ".join(cmd).encode()).hexdigest()[:12],
        "status": status,
    }
    _append_budget(row)

    if status == "ok":
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("", encoding="utf-8")
        logger.info("job ok: %s", marker)
        return 0
    logger.error("job failed: %s/%s/%s", args.exp, args.job, args.key)
    return 1


def _gpu_idle(gpu: int) -> bool:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True)
        return not out.stdout.strip()
    except Exception:
        return True  # cannot verify (e.g. local Windows smoke) — do not block


def _append_budget(row: dict) -> None:
    p = paths.budget_csv()
    p.parent.mkdir(parents=True, exist_ok=True)
    new_file = not p.exists()
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "exp", "job", "run_key", "gpu",
                                          "wall_min", "gpu_min", "cmd_hash", "status"])
        if new_file:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    sys.exit(main())