"""Demo emitters (plan §6.8): hook blind comparison (q32 BF16 vLLM + 3 GGUF quants via
llama-server, GPU0) and token visualization (CPU, Pillow). The author picks the blind
pair afterwards; the key stays in hook_key.json (plan Q6).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config  # noqa: E402
from scripts.exp import gpuplan, jobspec  # noqa: E402


def build_jobs(profile: str = "full") -> list[dict]:
    demo = config.load("experiments")["demo"]
    py = jobspec.sys_exec()
    return [
        jobspec.job(
            "demo:hook", "demo", "gpu",
            jobspec.record_cmd("demo", "hook", "hookA",
                               [py, "scripts/demo/hook_run.py"]),
            deps=[f"dl:q32"] + [f"q:{_rk('q32', q)}" for q in demo["hook_quants"]],
            est_min=25, gpu_affinity=gpuplan.gpu_of_role("demo")),
        jobspec.job(
            "demo:tokens", "demo", "cpu",
            jobspec.record_cmd("demo", "tokens", "tokA",
                               [py, "scripts/demo/token_viz.py"]),
            deps=["texts:own"], est_min=10),
    ]


def _rk(model: str, quant: str) -> str:
    from scripts.common.ids import run_key
    return run_key(model, quant, "none" if quant == "bf16" else
                   config.load("experiments")["E1"]["calib"], "s0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["full", "smoke"])
    ap.add_argument("--stage", default="produce", choices=["produce", "eval", "analyze"])
    args = ap.parse_args()
    print(json.dumps(build_jobs(args.profile), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())