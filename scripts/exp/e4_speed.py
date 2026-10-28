"""E4 — speed (plan §6.8): q8 + q32 on an exclusive GPU; llama-bench + vLLM latency;
32B concurrency sweep (vLLM engines only). Theory analysis is an analyze job.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, ids  # noqa: E402
from scripts.exp import gpuplan, jobspec  # noqa: E402


def _run_key(model: str, quant: str) -> str:
    no_calib = set(config.load("quants")["no_calib"])
    calib = "none" if quant in no_calib else config.load("experiments")["E1"]["calib"]
    return ids.run_key(model, quant, calib, "s0")


def build_jobs(profile: str = "full") -> list[dict]:
    e4 = config.load("experiments")["E4"]
    py = jobspec.sys_exec()
    jobs: list[dict] = []
    for model in e4["models"]:
        gpu = gpuplan.gpu_of_model(model)
        hf = f"models/hf/{model}"
        # llama-bench cells: bf16 GGUF + the 4 GGUF quants
        for quant in ["bf16"] + config.load("quants")["gguf_quants"]:
            rk = _run_key(model, quant)
            jobs.append(jobspec.job(
                f"sp:{rk}__llama", "E4", "gpu",
                jobspec.record_cmd("E4", "speed", f"{rk}__llama",
                                   [py, "scripts/e4/bench.py", "--exp", "E4",
                                    "--llama-bench", rk]),
                deps=[f"gguf:{model}"] + ([f"q:{rk}"] if quant != "bf16" else []),
                est_min=8, gpu_affinity=gpu))
        # vLLM latency cells: bf16, fp8, gptq
        for quant in ["bf16", "fp8", "gptq"]:
            rk = _run_key(model, quant)
            mp = hf if quant == "bf16" else f"models/ct/{rk}"
            jobs.append(jobspec.job(
                f"sp:{rk}__vllm", "E4", "gpu",
                jobspec.record_cmd("E4", "speed", f"{rk}__vllm",
                                   [py, "scripts/e4/bench.py", "--exp", "E4",
                                    "--vllm-latency", rk, "--model-path", mp]),
                deps=[f"dl:{model}"] + ([f"q:{rk}"] if quant != "bf16" else []),
                est_min=10, gpu_affinity=gpu))
        # 32B concurrency sweep (vLLM engines, plan §6.8)
        if model == "q32":
            for quant in ["bf16", "fp8", "gptq"]:
                rk = _run_key(model, quant)
                mp = hf if quant == "bf16" else f"models/ct/{rk}"
                jobs.append(jobspec.job(
                    f"serve:{quant}", "E4", "gpu",
                    jobspec.record_cmd("E4", "speed", f"serve__{quant}",
                                       [py, "scripts/e4/bench.py", "--exp", "E4",
                                        "--vllm-serve", quant, "--model-path", mp,
                                        "--concs", ",".join(str(c) for c in e4["concurrency"])]),
                    deps=[f"dl:{model}"] + ([f"q:{rk}"] if quant != "bf16" else []),
                    est_min=45, gpu_affinity=gpu))
        # theory reads every E4 runs.jsonl row for this model, so wait for all its cells
        sp_ids = [j["id"] for j in jobs
                  if j["id"].startswith(f"sp:{model}__") or j["id"].startswith("serve:")]
        jobs.append(jobspec.job(
            f"theory:{model}", "E4", "cpu",
            jobspec.record_cmd("E4", "theory", model,
                               [py, "scripts/e4/theory.py", "--model", model]),
            deps=sorted(set(sp_ids)), est_min=1))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["full", "smoke"])
    ap.add_argument("--stage", default="produce", choices=["produce", "eval", "analyze"])
    args = ap.parse_args()
    print(json.dumps(build_jobs(args.profile), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())