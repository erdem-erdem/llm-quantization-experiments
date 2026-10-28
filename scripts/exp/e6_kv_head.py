"""E6 — KV cache + needle + lm_head (plan §6.8). GPUs from experiments.yaml gpu_plan: role
e6_kv for E6-A (kv_model q8), role e6_head for every E6-B lm_head variant.

A: kv runner (server memory + bench decode) → needle runner (225 paired generations,
F16@4K precondition). B: head runner (re-quantized lm_head GGUFs) → dequant + eval
(Belebele-tr, ARC-tr) + KL for each head variant.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config  # noqa: E402
from scripts.common.ids import run_key  # noqa: E402
from scripts.exp import gpuplan, jobspec  # noqa: E402


def build_jobs(profile: str = "full") -> list[dict]:
    e6 = config.load("experiments")["E6"]
    e1 = config.load("experiments")["E1"]
    calib = e1["calib"]
    tasks = ",".join(["belebele_tr", "arc_tr"])
    py = jobspec.sys_exec()
    ctxs_arg = (["--contexts", ",".join(str(c) for c in
                                        config.load("smoke")["needles"]["contexts"])]
                if profile == "smoke" else [])
    jobs: list[dict] = []

    # ---- A: KV cache + needle
    kv_model = e6["kv_model"]
    kv_gguf = run_key(kv_model, e6["kv_base_quant"], "none", "s0")
    jobs.append(jobspec.job(
        "kv:E6A", "E6", "gpu",
        jobspec.record_cmd("E6", "kv", "kvA",
                           [py, "scripts/e6/kv_cache.py"]),
        deps=[f"q:{kv_gguf}"], est_min=40, gpu_affinity=gpuplan.gpu_of_role("e6_kv")))
    jobs.append(jobspec.job(
        "needle:E6A", "E6", "gpu",
        jobspec.record_cmd("E6", "needle", "needleA",
                           [py, "scripts/e6/needle.py", *ctxs_arg]),
        deps=[f"q:{kv_gguf}", "kv:E6A", "texts:own"], est_min=60,
        gpu_affinity=gpuplan.gpu_of_role("e6_kv")))

    # ---- B: lm_head variants (accuracy + KL + sizes via head__<run_key>.json)
    for model in e6["head_models"]:
        gpu = gpuplan.gpu_of_role("e6_head")
        base_rk = run_key(model, e6["head_base_quant"], calib, "s0")
        for ty in e6["head_types"]:
            rk = f"{model}__h{ty}__{e6['head_base_quant']}__s0"
            jobs.append(jobspec.job(
                f"head:{rk}", "E6", "cpu",
                jobspec.record_cmd("E6", "head", rk,
                                   [py, "scripts/e6/lm_head.py", "--run-key", rk]),
                deps=[f"q:{base_rk}"], est_min=25))
            jobs.append(jobspec.job(
                f"deq6:{rk}", "E6", "cpu",
                jobspec.record_cmd("E6", "dequant", rk,
                                   [py, "scripts/q/dequant.py", "--run-key", rk],
                                   validator="scripts.q.dequant_validate:validate_outputs"),
                deps=[f"head:{rk}"], est_min=25))
            jobs.append(jobspec.job(
                f"ev6:{rk}", "E6", "gpu",
                jobspec.record_cmd("E6", "eval", rk,
                                   [py, "scripts/eval/lm_eval_run.py",
                                    "--run-key", rk, "--exp", "E6", "--engine", "vllm-deq",
                                    "--tasks", tasks]),
                deps=[f"deq6:{rk}"], est_min=12, gpu_affinity=gpu))
            jobs.append(jobspec.job(
                f"kl6:{rk}", "E6", "gpu",
                jobspec.record_cmd("E6", "kl", rk,
                                   [py, "scripts/kl/kl_llama.py", "--run-key", rk,
                                    "--exp", "E6"]),
                deps=[f"head:{rk}", f"klref:{model}"], est_min=6, gpu_affinity=gpu))
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