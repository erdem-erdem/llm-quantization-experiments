"""Aggregate every emitter's jobs into configs/queue.<profile>.json (runbook step 6).

Full profile: the DAG must be complete — duplicate ids or dangling deps raise.
Smoke profile: jobs whose deps never resolve within the smoke set are dropped
(iteratively), because most experiments only run at full scale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import logging_setup  # noqa: E402
from scripts.exp import gpuplan, jobspec  # noqa: E402

EMITTERS = ["t0_prep", "e0_reliability", "e1_grid", "e2_calib", "e3_pareto",
            "e4_speed", "e5_tensor", "e6_kv_head", "demo_hook", "master_build"]


def collect(profile: str) -> list[dict]:
    jobs: list[dict] = []
    for name in EMITTERS:
        mod = __import__(f"scripts.exp.{name}", fromlist=["build_jobs"])
        jobs.extend(mod.build_jobs(profile))
    return jobs


def validate(jobs: list[dict], profile: str) -> list[dict]:
    ids = [j["id"] for j in jobs]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        raise RuntimeError(f"duplicate job ids: {dups}")
    by_id = {j["id"]: j for j in jobs}
    if profile == "full":
        dangling = sorted({d for j in jobs for d in j["deps"] if jobspec.dep_id(d) not in by_id})
        if dangling:
            raise RuntimeError(f"dangling deps: {dangling}")
        return jobs
    # smoke: iteratively drop jobs whose deps are outside the emitted set
    kept = {j["id"] for j in jobs}
    while True:
        drop = {j["id"] for j in jobs if j["id"] in kept
                and any(jobspec.dep_id(d) not in kept for d in j["deps"])}
        if not drop:
            break
        kept -= drop
    dropped = [j["id"] for j in jobs if j["id"] not in kept]
    if dropped:
        print(f"smoke: {len(dropped)} jobs dropped (unresolvable deps): {sorted(dropped)}")
    return [j for j in jobs if j["id"] in kept]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["full", "smoke"])
    ap.add_argument("--out", default=None, help="override configs/queue.<profile>.json")
    args = ap.parse_args()
    logging_setup.setup("queue_gen")

    jobs = validate(collect(args.profile), args.profile)
    # every GPU job must name its GPU (scheduler does static per-GPU assignment)
    unassigned = sorted(j["id"] for j in jobs
                        if j["kind"] == "gpu" and j["gpu_affinity"] is None)
    if unassigned:
        raise RuntimeError(f"gpu jobs without gpu_affinity: {unassigned}")

    out = Path(args.out) if args.out else Path(f"configs/queue.{args.profile}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # every GPU job must sit on an existing GPU of the plan (static per-GPU assignment)
    gpus = gpuplan.n_gpus()
    off_plan = sorted(j["id"] for j in jobs
                      if j["kind"] == "gpu" and not 0 <= j["gpu_affinity"] < gpus)
    if off_plan:
        raise RuntimeError(f"gpu jobs pinned outside the {gpus}-GPU plan: {off_plan}")
    out.write_text(json.dumps({"profile": args.profile, "gpus": gpus, "jobs": jobs},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    gpu = sum(1 for j in jobs if j["kind"] == "gpu")
    print(f"{out}: {len(jobs)} jobs ({gpu} gpu, {len(jobs) - gpu} cpu), "
          f"est {sum(j['est_min'] for j in jobs) / 60:.1f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())