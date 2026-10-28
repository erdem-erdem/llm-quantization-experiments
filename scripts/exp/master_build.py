"""Analyze-stage emitter: build_master (results/master.csv) and the final figure render.
Deps cover every measurement job emitted by the other emitters, so both run last.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp import jobspec  # noqa: E402

EMITTERS = ["t0_prep", "e0_reliability", "e1_grid", "e2_calib", "e3_pareto",
            "e4_speed", "e5_tensor", "e6_kv_head", "demo_hook"]
MEASURE_PREFIXES = ("ev", "kl", "kv:", "needle", "gate", "serve", "sp:", "head:",
                    "analyze", "theory", "e5map", "e5recipe", "ud5", "demo:")


def all_job_ids(profile: str) -> list[str]:
    ids: list[str] = []
    for name in EMITTERS:
        mod = __import__(f"scripts.exp.{name}", fromlist=["build_jobs"])
        ids.extend(j["id"] for j in mod.build_jobs(profile))
    return ids


def build_jobs(profile: str = "full") -> list[dict]:
    py = jobspec.sys_exec()
    # analyze:E3 kendisi master:build'i bekler — çember olmasın diye hariç
    ids = sorted(set(i for i in all_job_ids(profile)
                     if i.startswith(MEASURE_PREFIXES) and i != "analyze:E3"))
    return [
        jobspec.job(
            "master:build", "analyze", "cpu",
            jobspec.record_cmd("analyze", "master", "build",
                               [py, "scripts/stats/build_master.py", "--exps", "all"]),
            deps=[f"attempted:{jid}" for jid in sorted(set(ids))], 
            outputs=["results/master.csv"], est_min=3),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["full", "smoke"])
    ap.add_argument("--stage", default="analyze", choices=["produce", "eval", "analyze"])
    args = ap.parse_args()
    print(json.dumps(build_jobs(args.profile), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())