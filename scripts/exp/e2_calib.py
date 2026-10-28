"""E2 — calibration language & seed (plan §6.8): E2.models (q17, q4) × {gptq,q3_k_m} ×
{en,tr,mix} × {s0,s1}; spec §11 cut 2: s1 only for E2.s1_models (q4).

The (tr, s0) cells reuse E1 production (only ARC-en — and Belebele-en for models E1 skipped it
for — is new). Controls per model: rtn (calib none) and q3_k_m with no imatrix. Tasks:
Belebele-tr, ARC-tr, Belebele-en, ARC-en. GPU: gpu_plan role e2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, ids  # noqa: E402
from scripts.exp import gpuplan, jobspec  # noqa: E402


def build_jobs(profile: str = "full") -> list[dict]:
    e2 = config.load("experiments")["E2"]
    tasks = ",".join(e2["tasks"])
    py = jobspec.sys_exec()
    limit = config.load("smoke")["lm_eval_limit"] if profile == "smoke" else None
    jobs: list[dict] = []
    # every E2 job runs on the e2 role's GPU (2-GPU plan: production and evals share it)
    EVAL_GPU = PROD_GPU = {m: gpuplan.gpu_of_role("e2") for m in e2["models"]}
    e1_en_models = set(config.load("experiments")["E1"]["belebele_en_models"])

    def missing_in_e1(model: str) -> str:
        """Tasks E2 must add on top of E1's runs for a reused cell / the BF16 reference:
        ARC-en always, Belebele-en too when E1 skipped it for this model (spec §11 cut 3)."""
        return "arc_en" if model in e1_en_models else "belebele_en,arc_en"

    def eval_job(model: str, rk: str, gpu: int, deps: list[str],
                 task_list: str = tasks) -> dict:
        # GGUF quants are evaluated through their dequantized BF16 checkpoint (K-03)
        engine = "vllm-deq" if ids.parse_run_key(rk)[1] in ids.GGUF_QUANTS else "vllm"
        return jobspec.job(
            f"ev2:{rk}", "E2", "gpu",
            jobspec.record_cmd("E2", "eval", rk,
                               [py, "scripts/eval/lm_eval_run.py",
                                "--run-key", rk, "--exp", "E2", "--engine", engine,
                                "--tasks", task_list] + (["--limit", str(limit)] if limit else [])),
            deps=deps, est_min=16, gpu_affinity=gpu)

    for model in e2["models"]:
        for method in e2["methods"]:
            for calib in e2["calibs"]:
                for seed in e2["seeds"]:
                    if seed != "s0" and model not in e2.get("s1_models", e2["models"]):
                        continue  # spec §11 cut 2: extra seeds only on the listed models
                    rk = ids.run_key(model, method, calib, seed)
                    reused = calib == "tr" and seed == "s0"
                    deps: list[str] = []
                    if not reused:
                        if method == "gptq":
                            deps = [f"dl:{model}"]
                            jobs.append(jobspec.job(
                                f"q2:{rk}", "E2", "gpu",
                                jobspec.record_cmd("E2", "llmcompress", rk,
                                                   [py, "scripts/q/llmcompress.py",
                                                    "--run-key", rk]),
                                deps=deps, est_min=35, gpu_affinity=PROD_GPU[model]))
                        else:  # q3_k_m with calibration
                            jobs.append(jobspec.job(
                                f"imat2:{rk}", "E2", "gpu",
                                jobspec.record_cmd("E2", "imatrix", f"{model}__{calib}__{seed}",
                                                   [py, "scripts/q/imatrix.py",
                                                    "--model", model, "--calib", calib,
                                                    "--seed", seed]),
                                deps=[f"gguf:{model}"], est_min=15,
                                gpu_affinity=PROD_GPU[model]))
                            jobs.append(jobspec.job(
                                f"q2:{rk}", "E2", "cpu",
                                jobspec.record_cmd("E2", "gguf_quantize", rk,
                                                   [py, "scripts/q/gguf_quantize.py",
                                                    "--run-key", rk]),
                                deps=[f"gguf:{model}", f"imat2:{rk}"], est_min=20))
                            jobs.append(jobspec.job(
                                f"deq2:{rk}", "E2", "cpu",
                                jobspec.record_cmd("E2", "dequant", rk,
                                                   [py, "scripts/q/dequant.py",
                                                    "--run-key", rk],
                                                   validator="scripts.q.dequant_validate:validate_outputs"),
                                deps=[f"q2:{rk}"], est_min=25))
                        deps = [f"q2:{rk}" if method == "gptq" else f"deq2:{rk}"]
                        jobs.append(eval_job(model, rk, EVAL_GPU[model], deps))
                        if method != "gptq":
                            jobs.append(jobspec.job(
                                f"kl2:{rk}", "E2", "gpu",
                                jobspec.record_cmd("E2", "kl", rk,
                                                   [py, "scripts/kl/kl_llama.py",
                                                    "--run-key", rk, "--exp", "E2"]),
                                deps=[f"q2:{rk}", f"klref:{model}"], est_min=6,
                                gpu_affinity=PROD_GPU[model]))
                    else:
                        # production reused from E1; only ARC-en is missing from its eval set
                        ev_deps = [f"ev:{rk}"]
                        if method != "gptq":
                            jobs.append(jobspec.job(
                                f"deq2:{rk}", "E2", "cpu",
                                jobspec.record_cmd("E2", "dequant", rk,
                                                   [py, "scripts/q/dequant.py", "--run-key", rk],
                                                   validator="scripts.q.dequant_validate:validate_outputs"),
                                deps=[f"q:{rk}", f"ev:{rk}"], est_min=25))
                            ev_deps = [f"deq2:{rk}"]
                        jobs.append(eval_job(model, rk, EVAL_GPU[model], ev_deps,
                                             task_list=missing_in_e1(model)))

        # BF16 reference for ARC-en (E1 does not run it)
        jobs.append(eval_job(model, ids.run_key(model, "bf16", "none", "s0"), EVAL_GPU[model],
                             [f"dl:{model}"], task_list=missing_in_e1(model)))

        # controls: RTN-W4 (no calib) and Q3_K_M without imatrix
        for ctrl in e2["controls"]:
            quant = ctrl.split("__")[0] if "__" in ctrl else ctrl
            calib = ctrl.split("__")[1] if "__" in ctrl else "none"
            rk = ids.run_key(model, quant, calib, "s0")
            deps: list[str] = [f"dl:{model}"]
            if quant in ids.LLMC_QUANTS:
                jobs.append(jobspec.job(
                    f"q2:{rk}", "E2", "gpu",
                    jobspec.record_cmd("E2", "llmcompress", rk,
                                       [py, "scripts/q/llmcompress.py", "--run-key", rk]),
                    deps=deps, est_min=30, gpu_affinity=PROD_GPU[model]))
                deps = [f"q2:{rk}"]
            else:  # q3_k_m__none: quantize without imatrix
                jobs.append(jobspec.job(
                    f"q2:{rk}", "E2", "cpu",
                    jobspec.record_cmd("E2", "gguf_quantize", rk,
                                       [py, "scripts/q/gguf_quantize.py", "--run-key", rk]),
                    deps=[f"gguf:{model}"], est_min=20))
                jobs.append(jobspec.job(
                    f"deq2:{rk}", "E2", "cpu",
                    jobspec.record_cmd("E2", "dequant", rk,
                                       [py, "scripts/q/dequant.py", "--run-key", rk],
                                       validator="scripts.q.dequant_validate:validate_outputs"),
                    deps=[f"q2:{rk}"], est_min=25))
                deps = [f"deq2:{rk}"]
            jobs.append(eval_job(model, rk, EVAL_GPU[model], deps))
            if quant in ids.GGUF_QUANTS:
                jobs.append(jobspec.job(
                    f"kl2:{rk}", "E2", "gpu",
                    jobspec.record_cmd("E2", "kl", rk,
                                       [py, "scripts/kl/kl_llama.py", "--run-key", rk,
                                        "--exp", "E2"]),
                    deps=[f"q2:{rk}", f"klref:{model}"], est_min=6,
                    gpu_affinity=PROD_GPU[model]))
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