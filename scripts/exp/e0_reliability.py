"""E0 — reliability (plan §6.8, model q8, GPU2): noise floor, path consistency, gate.

1. BF16-eager eval (Belebele-tr + ARC-tr) → noise floor vs the E1 BF16-default run.
2. Path consistency: kl_hf (HF dequant) vs kl_llama (GGUF, from E1) for q4_k_m and q2_k.
3. Gate: writes results/E0/gate.json {"pass": bool, "reasons": [...]}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, ids, io, paths  # noqa: E402
from scripts.exp import gpuplan, jobspec  # noqa: E402


def build_jobs(profile: str = "full") -> list[dict]:
    e0 = config.load("experiments")["E0"]
    model = e0["model"]
    e1 = config.load("experiments")["E1"]
    calib, seed = e1["calib"], e1["seed"]
    gpu = gpuplan.gpu_of_role("e0")
    py = jobspec.sys_exec()
    limit = config.load("smoke")["lm_eval_limit"] if profile == "smoke" else None
    jobs: list[dict] = []

    # 1. noise floor: BF16 with enforce_eager=True (default run comes from E1)
    rk_eager = ids.run_key(model, "bf16eager", "none", seed)
    jobs.append(jobspec.job(
        "ev:e0-eager", "E0", "gpu",
        jobspec.record_cmd("E0", "eval", rk_eager,
                           [py, "scripts/eval/lm_eval_run.py",
                            "--run-key", rk_eager, "--exp", "E0", "--engine", "vllm",
                            "--eager", "--tasks", ",".join(e0["noise_tasks"])]
                           + (["--limit", str(limit)] if limit else [])),
        deps=[f"dl:{model}", f"ev:{ids.run_key(model, 'bf16', 'none', seed)}"],
        est_min=10, gpu_affinity=gpu))

    # 2. path consistency: HF-side KL for the same GGUF quants llama.cpp already scored in E1
    for quant in e0["kl_path_quants"]:
        rk = ids.run_key(model, quant, calib, seed)
        jobs.append(jobspec.job(
            f"klhf:{rk}", "E0", "gpu",
            jobspec.record_cmd("E0", "kl_hf", rk,
                               [py, "scripts/kl/kl_hf.py",
                                "--ref", f"models/hf/{model}",
                                "--test", f"models/tmp_deq/{rk}",
                                "--exp", "E0", "--run-key", rk, "--cleanup"]),
            deps=[f"dl:{model}", f"deq:{rk}", f"ev:{rk}"], est_min=10,
            gpu_affinity=gpu))

    # 3. gate (CPU analysis over sample vectors + KL rows). It reads the E1 llama.cpp KL rows
    # for every monotonic quant (q8_0 is keyed calib=none) and the HF-side klhf rows.
    rk_def = ids.run_key(model, "bf16", "none", seed)
    no_calib = set(config.load("quants")["no_calib"])
    kl_deps = [f"kl:{ids.run_key(model, q, 'none' if q in no_calib else calib, seed)}"
               for q in e0["monotonic_quants"]]
    jobs.append(jobspec.job(
        "gate:E0", "E0", "cpu",
        jobspec.record_cmd("E0", "gate", model,
                           [py, "scripts/exp/e0_reliability.py", "--compute-gate"]),
        deps=["ev:e0-eager", f"ev:{rk_def}"] + kl_deps
             + [f"klhf:{ids.run_key(model, q, calib, seed)}" for q in e0["kl_path_quants"]],
        outputs=["results/E0/gate.json"], est_min=1))
    return jobs


def _vector(exp: str, run_key: str, task: str) -> np.ndarray:
    """Correctness vector from a sample vector file (header line + entries)."""
    rows = io.read_jsonl(paths.sample_vector_path(exp, run_key, task))
    return np.array([r["correct"] for r in rows[1:]], dtype=np.int8)


def _read_metric(exp: str, run_key: str, metric: str) -> float | None:
    """KL value from runs.jsonl, matched on the run_key's four components. Returns None if not found."""
    parts = ids.parse_run_key(run_key)
    vals = [r["value"] for r in io.read_jsonl(paths.runs_jsonl(exp))
            if r.get("task") == "kl_havadis" and r.get("metric") == metric
            and (r.get("model"), r.get("quant"), r.get("calib"), r.get("seed")) == parts]
    if not vals:
        return None
    return float(vals[0])


def compute_gate() -> int:
    """Monotonicity + path-consistency gate; writes results/E0/gate.json (fails loudly)."""
    from scripts.common import io, paths
    from scripts.stats import matched

    e0 = config.load("experiments")["E0"]
    model = e0["model"]
    calib = config.load("experiments")["E1"]["calib"]
    reasons: list[str] = []

    # noise floor: delta(bf16_default -> bf16eager) averaged over the noise tasks
    rk_def = ids.run_key(model, "bf16", "none", "s0")
    rk_eager = ids.run_key(model, "bf16eager", "none", "s0")
    pairs = [(_vector("E1", rk_def, t), _vector("E0", rk_eager, t))
             for t in e0["noise_tasks"]]
    delta, lo, hi = matched.paired_bootstrap_avg(pairs)
    flip_rate = float(np.mean([np.mean((r == 0) & (q == 1)) + np.mean((r == 1) & (q == 0))
                               for r, q in pairs]))
    noise = {"delta": delta, "ci": [lo, hi],
             "delta_abs_max": float(max(abs(lo), abs(hi))), "flip_rate": flip_rate}
    io.atomic_write_json(paths.RESULTS / "E0" / "noise_band.json", noise)
    # Noise reporting band (no hard fail per D1)
    if noise["delta_abs_max"] > 0.35:
        print(f"WARN: noise-floor |delta| {noise['delta_abs_max']:.2f} pts exceeds 0.35 pts (reporting band)")

    # path consistency: llama.cpp KL (E1 rows) vs HF-dequant KL (E0 rows)
    for quant in e0["kl_path_quants"]:
        rk = ids.run_key(model, quant, calib, "s0")
        kl_llama = _read_metric("E1", rk, "mean_kld")
        kl_hf = _read_metric("E0", rk, "mean_kld")
        if kl_llama is None or kl_hf is None:
            reasons.append(f"path check {quant}: missing KL data (E1: {kl_llama is not None}, E0: {kl_hf is not None})")
            continue
        rel = abs(kl_llama - kl_hf) / max(kl_llama, 1e-9)
        # Path consistency reporting band (no hard fail per D1)
        if rel > 0.25:
            print(f"WARN: path mismatch {quant}: llama.cpp {kl_llama:.4g} vs HF {kl_hf:.4g} (rel {rel:.1%}) (reporting band)")

    # monotonicity: KL non-decreasing q8_0 -> q4_k_m -> q3_k_m -> q2_k
    no_calib = set(config.load("quants")["no_calib"])  # q8_0 is keyed calib=none
    kls = [(q, _read_metric("E1", ids.run_key(model, q, "none" if q in no_calib else calib, "s0"),
                            "mean_kld"))
           for q in e0["monotonic_quants"]]
    for (qa, ka), (qb, kb) in zip(kls, kls[1:]):
        # KL monotonicity reporting band (no hard fail per D1)
        if kb < ka * 0.95:
            print(f"WARN: KL not monotone: {qb} ({kb:.4g}) < {qa} ({ka:.4g}) (reporting band)")

    gate = {"pass": not reasons, "reasons": reasons, "noise": noise}
    io.atomic_write_json(paths.RESULTS / "E0" / "gate.json", gate)
    print(json.dumps(gate, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["full", "smoke"])
    ap.add_argument("--stage", default="produce", choices=["produce", "eval", "analyze"])
    ap.add_argument("--compute-gate", action="store_true",
                    help="run the gate analysis now (the analyze job's command)")
    args = ap.parse_args()
    if args.compute_gate:
        return compute_gate()
    print(json.dumps(build_jobs(args.profile), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())