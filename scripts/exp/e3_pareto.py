"""E3 — Pareto post-processing (plan §6.8): single analyze job over master.csv
(frontier + budget top-3 + example rows). Requires build_master to have run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp import jobspec  # noqa: E402


def build_jobs(profile: str = "full") -> list[dict]:
    return [jobspec.job(
        "analyze:E3", "E3", "cpu",
        jobspec.record_cmd("E3", "pareto", "analyze",
                           [jobspec.sys_exec(), "scripts/stats/pareto.py"]),
        deps=["master:build"], outputs=["results/E3/extras.json",
                                        "figures/E3_budget_top3.csv"],
        est_min=1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["full", "smoke"])
    ap.add_argument("--stage", default="analyze", choices=["produce", "eval", "analyze"])
    args = ap.parse_args()
    print(json.dumps(build_jobs(args.profile), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())