"""E5 — tensor-level sensitivity (plan §6.8): map sweep → recipe search → produce → eval.

Jobs (GPU: gpu_plan role e5): e5map (2-bit RTN per-tensor sweep, 252 cells) → e5recipe (tier
search, results/E5/recipes.json) → produce (mix, rand0, rand1; + the 8B transfer when
E5.stage3 is on) → dequant+eval
(Belebele-tr, ARC-tr) + KL on the map-excluded segment (--nomap). Controls: random recipes
seeds 0/1, Unsloth UD (nearest-to-Q3_K_M GGUF download), uniform = E1 q4__q3_k_m__tr__s0.
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
    GPU = gpuplan.gpu_of_role("e5")
    e5 = config.load("experiments")["E5"]
    stage3 = bool(e5.get("stage3", True))  # spec §11 cut 4 turns the 8B transfer stage off
    ref = e5["calib_ref"]
    ref_key = f"{ref['model']}__{ref['quant']}__{ref['calib']}__{ref['seed']}"
    e1 = config.load("experiments")["E1"]
    tasks = ",".join(e5["eval_tasks"])
    py = jobspec.sys_exec()
    layers_arg = (["--layers", str(config.load("smoke")["e5"]["layers"])]
                  if profile == "smoke" else [])
    jobs: list[dict] = []

    jobs.append(jobspec.job(
        "e5map", "E5", "gpu",
        jobspec.record_cmd("E5", "map", "map",
                           [py, "scripts/e5/map_sweep.py", *layers_arg]),
        # map_sweep reads the uniform Q3_K_M tensor dump (bytes per tensor) when it finishes
        deps=[f"dl:{ref['model']}", f"q:{ref_key}"], est_min=25, gpu_affinity=GPU))
    jobs.append(jobspec.job(
        "e5recipe", "E5", "cpu",
        jobspec.record_cmd("E5", "recipe", "search",
                           [py, "scripts/e5/recipe.py", *layers_arg]),
        deps=["e5map", f"q:{ref_key}"]
             + ([f"q:{e5['stage3_model']}__{ref['quant']}__{ref['calib']}__{ref['seed']}"]
                if stage3 else []),
        est_min=1))

    # KL on the map-excluded text: llama.cpp's KL run reads its tokens from the base .kld, so
    # E5 needs its own reference (refs/<m>__nomap.kld) and a uniform-Q3_K_M score on that text
    for m in (ref["model"], *([e5["stage3_model"]] if stage3 else [])):
        rk_ref = f"{m}__bf16__none__s0"
        jobs.append(jobspec.job(
            f"klref5:{m}", "E5", "gpu",
            jobspec.record_cmd("E5", "kl_ref", rk_ref,
                               [py, "scripts/kl/kl_llama.py", "--run-key", rk_ref,
                                "--exp", "E5", "--nomap", "--reference-only"]),
            deps=[f"gguf:{m}"], est_min=6, gpu_affinity=GPU))
        rk_uni = f"{m}__{ref['quant']}__{ref['calib']}__{ref['seed']}"
        jobs.append(jobspec.job(
            f"kl5:{rk_uni}", "E5", "gpu",
            jobspec.record_cmd("E5", "kl", rk_uni,
                               [py, "scripts/kl/kl_llama.py", "--run-key", rk_uni,
                                "--exp", "E5", "--nomap"]),
            deps=[f"q:{rk_uni}", f"klref5:{m}"], est_min=6, gpu_affinity=GPU))

    recipe_labels = ["q4__mix__tr__s0", "q4__rand0__tr__s0", "q4__rand1__tr__s0"]
    if stage3:
        recipe_labels += [f"{e5['stage3_model']}__mix__tr__s0",
                          f"{e5['stage3_model']}__rand0__tr__s0"]
    for label in recipe_labels:
        model = label.split("__")[0]
        jobs.append(jobspec.job(
            f"prod5:{label}", "E5", "cpu",
            jobspec.record_cmd("E5", "produce", label,
                               [py, "scripts/e5/produce.py", "--label", label]),
            deps=["e5recipe"], est_min=20))
        jobs.append(jobspec.job(
            f"deq5:{label}", "E5", "cpu",
            jobspec.record_cmd("E5", "dequant", label,
                               [py, "scripts/q/dequant.py", "--run-key", label],
                               validator="scripts.q.dequant_validate:validate_outputs"),
            deps=[f"prod5:{label}"], est_min=25))
        jobs.append(jobspec.job(
            f"ev5:{label}", "E5", "gpu",
            jobspec.record_cmd("E5", "eval", label,
                               [py, "scripts/eval/lm_eval_run.py",
                                "--run-key", label, "--exp", "E5", "--engine", "vllm-deq",
                                "--tasks", tasks]),
            deps=[f"deq5:{label}"], est_min=12, gpu_affinity=GPU))
        jobs.append(jobspec.job(
            f"kl5:{label}", "E5", "gpu",
            jobspec.record_cmd("E5", "kl", label,
                               [py, "scripts/kl/kl_llama.py", "--run-key", label,
                                "--exp", "E5", "--nomap"]),
            deps=[f"prod5:{label}", f"klref5:{model}"], est_min=6, gpu_affinity=GPU))

    # Unsloth UD control: the Q3_K_M-nearest file from unsloth/Qwen3-4B-GGUF
    jobs.append(jobspec.job(
        "ud5:download", "E5", "cpu",
        jobspec.record_cmd("E5", "ud_download", "q4__ud__tr__s0",
                           [py, "scripts/e5/ud_download.py"]),
        deps=["q:" + ref_key], est_min=10))
    jobs.append(jobspec.job(
        "deq5:q4__ud__tr__s0", "E5", "cpu",
        jobspec.record_cmd("E5", "dequant", "q4__ud__tr__s0",
                           [py, "scripts/q/dequant.py", "--run-key", "q4__ud__tr__s0"],
                           validator="scripts.q.dequant_validate:validate_outputs"),
        deps=["ud5:download"], est_min=25))
    jobs.append(jobspec.job(
        "ev5:q4__ud__tr__s0", "E5", "gpu",
        jobspec.record_cmd("E5", "eval", "q4__ud__tr__s0",
                           [py, "scripts/eval/lm_eval_run.py",
                            "--run-key", "q4__ud__tr__s0", "--exp", "E5",
                            "--engine", "vllm-deq", "--tasks", tasks]),
        deps=["deq5:q4__ud__tr__s0"], est_min=12, gpu_affinity=GPU))
    jobs.append(jobspec.job(
        "kl5:q4__ud__tr__s0", "E5", "gpu",
        jobspec.record_cmd("E5", "kl", "q4__ud__tr__s0",
                           [py, "scripts/kl/kl_llama.py", "--run-key", "q4__ud__tr__s0",
                            "--exp", "E5", "--nomap"]),
        deps=["ud5:download", "klref5:q4"], est_min=6, gpu_affinity=GPU))
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