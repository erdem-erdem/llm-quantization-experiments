"""T0 emitter: own texts (demo paragraphs, needles, haystacks) — CPU, no GPU needed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp import jobspec  # noqa: E402


def build_jobs(profile: str = "full") -> list[dict]:
    return [jobspec.job(
        "texts:own", "t0", "cpu",
        jobspec.record_cmd("t0", "own_texts", "t0",
                           [jobspec.sys_exec(), "scripts/t0/own_texts.py"]),
        outputs=["data/own/haystacks"], est_min=3)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["full", "smoke"])
    ap.add_argument("--stage", default="produce", choices=["produce", "eval", "analyze"])
    args = ap.parse_args()
    print(json.dumps(build_jobs(args.profile), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())