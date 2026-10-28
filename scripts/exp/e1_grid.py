"""E1 — main grid (plan §6.8): 6 models × 7 conditions = 42 evals + KL for GGUF quants.

Emits job specs; queue_gen.py consumes them. Production chain per (model, quant):
download → [gguf-convert → (imatrix) → quantize → dequant] | [llmcompress] → eval → (KL).
Quant calib comes from experiments.yaml E1.calib (tr); no_calib quants get calib=none.
GPU affinity comes from experiments.yaml gpu_plan (scripts/exp/gpuplan.py). Cuts (spec §11):
no accuracy eval for q32 q8_0/q3_k_m; Belebele-en only for E1.belebele_en_models.
Smoke profile: q06 + q8_0 only, lm_eval --limit from smoke.yaml; no KL (llama.cpp-free).
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
    quants_cfg = config.load("quants")
    e1 = config.load("experiments")["E1"]
    e0 = config.load("experiments")["E0"]
    calib, seed = e1["calib"], e1["seed"]
    tasks = list(e1["tasks_all"])
    gpqa_models = set(config.load("tasks")["gpqa_models"])
    if profile == "smoke":
        models, quants, limit = ["q06"], ["q8_0"], config.load("smoke")["lm_eval_limit"]
    else:
        models, quants, limit = list(ids.MODELS), ["bf16"] + quants_cfg["e1_quants"], None

    jobs: list[dict] = []
    seen: set[str] = set()

    def add(j: dict) -> None:
        if j["id"] not in seen:
            seen.add(j["id"])
            jobs.append(j)

    order = ["q8", "q32", "q14", "q4", "q17", "q06"]
    for model in order:
        if model not in models:
            continue
        gpu = gpuplan.gpu_of_model(model)
        # spec §11 cut 3: Belebele-en only for the models listed in experiments.yaml
        model_tasks = [t for t in tasks
                       if t != "belebele_en" or model in e1["belebele_en_models"]]
        model_tasks += ["gpqa_tr"] if model in gpqa_models else []

        dl_id = f"dl:{model}"
        add(jobspec.job(
            dl_id, "E1", "cpu",
            jobspec.record_cmd("E1", "download", model,
                               [jobspec.sys_exec(), "scripts/t0/download_models.py",
                                "--models", model]),
            outputs=[f"models/hf/{model}"], est_min=30 if model == "q32" else 10))
        gguf_id = f"gguf:{model}"
        add(jobspec.job(
            gguf_id, "E1", "cpu",
            jobspec.record_cmd("E1", "gguf_convert", model,
                               [jobspec.sys_exec(), "scripts/t0/gguf_convert.py",
                                "--models", model]),
            deps=[dl_id], est_min=40))
        # BF16 reference for llama.cpp KL (shared by every GGUF quant of this model)
        rk_ref = ids.run_key(model, "bf16", "none", seed)
        klref_id = f"klref:{model}"
        if profile != "smoke":
            add(jobspec.job(
                klref_id, "E1", "gpu",
                jobspec.record_cmd("E1", "kl_ref", rk_ref,
                                   [jobspec.sys_exec(), "scripts/kl/kl_llama.py",
                                    "--run-key", rk_ref, "--exp", "E1", "--reference-only"]),
                deps=[gguf_id], outputs=[f"refs/{model}.kld"], est_min=8, gpu_affinity=gpu))

        for quant in quants:
            qcalib = "none" if quant == "bf16" or quant in quants_cfg["no_calib"] else calib
            rk = ids.run_key(model, quant, qcalib, seed)
            # spec §11 cut 1: some (model, quant) cells are produced (E4 speed, KL) but not evaluated
            run_eval = quant not in e1.get("skip_eval", {}).get(model, [])
            deps: list[str] = []
            if quant == "bf16":
                deps.append(dl_id)
            elif quant in ids.LLMC_QUANTS:  # fp8, gptq (rtn is E2-only)
                add(jobspec.job(
                    f"q:{rk}", "E1", "gpu",
                    jobspec.record_cmd("E1", "llmcompress", rk,
                                       [jobspec.sys_exec(), "scripts/q/llmcompress.py",
                                        "--run-key", rk]),
                    deps=[dl_id], est_min=35, gpu_affinity=gpu))
                deps.append(f"q:{rk}")
            else:  # gguf quants
                deps.append(gguf_id)
                if qcalib != "none":  # imatrix only for calib-carrying GGUF quants
                    imat_id = f"imat:{model}:{qcalib}:{seed}"
                    add(jobspec.job(
                        imat_id, "E1", "gpu",
                        jobspec.record_cmd("E1", "imatrix", f"{model}__{qcalib}__{seed}",
                                           [jobspec.sys_exec(), "scripts/q/imatrix.py",
                                            "--model", model, "--calib", qcalib,
                                            "--seed", seed]),
                        deps=[gguf_id], est_min=15, gpu_affinity=gpu))
                    deps.append(imat_id)
                add(jobspec.job(
                    f"q:{rk}", "E1", "cpu",
                    jobspec.record_cmd("E1", "gguf_quantize", rk,
                                       [jobspec.sys_exec(), "scripts/q/gguf_quantize.py",
                                        "--run-key", rk]),
                    deps=deps, est_min=20))
                if run_eval:  # no dequantized checkpoint (65 GB for q32) without an eval
                    add(jobspec.job(
                        f"deq:{rk}", "E1", "cpu",
                        jobspec.record_cmd("E1", "dequant", rk,
                                           [jobspec.sys_exec(), "scripts/q/dequant.py",
                                            "--run-key", rk],
                                           validator="scripts.q.dequant_validate:validate_outputs"),
                        deps=[f"q:{rk}"], est_min=25))
                    deps = [f"deq:{rk}"]

            engine = "vllm-deq" if quant in ids.GGUF_QUANTS else "vllm"
            # smoke evals (--limit) must not leave .done markers / vectors under the real E1 keys;
            # E0's HF-side KL check still needs the dequantized checkpoint after the eval
            ev_exp = "E1smoke" if profile == "smoke" else "E1"
            keep_deq = profile == "smoke" or (model == e0["model"] and quant in e0["kl_path_quants"])
            if run_eval:
                add(jobspec.job(
                    f"ev:{rk}", ev_exp, "gpu",
                    jobspec.record_cmd(ev_exp, "eval", rk,
                                       [jobspec.sys_exec(), "scripts/eval/lm_eval_run.py",
                                        "--run-key", rk, "--exp", ev_exp, "--engine", engine,
                                        "--tasks", ",".join(model_tasks)]
                                       + (["--limit", str(limit)] if limit else [])
                                       + (["--keep-deq"] if keep_deq and engine == "vllm-deq"
                                          else [])),
                    deps=deps, est_min=18, gpu_affinity=gpu))

            if quant in ids.GGUF_QUANTS and profile != "smoke":
                add(jobspec.job(
                    f"kl:{rk}", "E1", "gpu",
                    jobspec.record_cmd("E1", "kl", rk,
                                       [jobspec.sys_exec(), "scripts/kl/kl_llama.py",
                                        "--run-key", rk, "--exp", "E1"]),
                    deps=[f"q:{rk}", klref_id], est_min=6, gpu_affinity=gpu))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["full", "smoke"])
    ap.add_argument("--stage", default="produce", choices=["produce", "eval", "analyze"],
                    help="kept for the emitter contract; E1 analyze runs via build_master")
    args = ap.parse_args()
    print(json.dumps(build_jobs(args.profile), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())