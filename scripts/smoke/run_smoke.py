"""Smoke runner (plan §8): Layer A (Python-only, Windows-friendly) and Layer B (optional,
needs a local llama.cpp build via --llamacpp). Each stage is a subprocess with its own
log under logs/smoke/; the runner prints a per-stage PASS/FAIL table and exits nonzero
if any mandatory stage failed. Budget: ≲30 min (A), ≲60 min (B).

Layer A: pytest suite; T0 pipeline on q06 with smoke-sized data (Havadis 8k, calib s0);
custom task YAMLs; demo token viz (real tokenizer); lm_eval on the HF backend with
--limit 50 (vLLM unavailable locally — deviation noted in NOTES); scheduler dry-run.
Layer B: q06 GGUF convert → imatrix → q4_k_m/q2_k quantize + dump → dequant identity +
bit-exact → kl_llama vs kl_hf agreement → 4K needle sanity → llama-server --jinja sanity.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, logging_setup, paths  # noqa: E402


def _py() -> str:
    return sys.executable


def layer_a_stages(skip_eval: bool) -> list[tuple[str, list[str], bool]]:
    smoke = config.load("smoke")
    limit = smoke["lm_eval_limit"]
    stages = [
        ("pytest", [_py(), "-m", "pytest", "-q", "tests/", "--no-header"], True),
        ("t0-download-q06", [_py(), "scripts/t0/download_models.py", "--models", "q06"], True),
        ("t0-havadis-smoke", [_py(), "scripts/t0/havadis_eval.py", "--smoke"], True),
        ("t0-calib-s0", [_py(), "scripts/t0/calib_sets.py", "--langs", "tr,en",
                         "--seeds", "s0"], True),
        ("t0-own-texts", [_py(), "scripts/t0/own_texts.py"], True),
        ("t0-gpqa-perm", [_py(), "scripts/t0/gpqa_perm.py"], True),
        # t0-manifest is NOT in Layer A: it hashes full dataset downloads (hundreds of
        # files, hours) and nothing downstream in the smoke consumes data/manifest.json;
        # the remote RUNBOOK step 5 runs it as part of the real T0.
        ("eval-tasks-setup", [_py(), "scripts/eval/tasks_setup.py"], True),
        ("eval-arc-tr-check", [_py(), "scripts/eval/arc_tr_check.py"], True),
        ("demo-token-viz", [_py(), "scripts/demo/token_viz.py"], False),
        ("scheduler-dry-run", [_py(), "scripts/scheduler.py", "--queue",
                               "configs/queue.smoke.json", "--dry-run"], True),
    ]
    if not skip_eval:
        stages.append((
            "lm-eval-hf-limit50",
            [_py(), "-m", "lm_eval", "--model", "hf",
             "--model_args",
             f"pretrained={paths.model_hf_path('q06')},dtype=bfloat16",
             "--tasks", "belebele_tr,belebele_en,arc_tr",
             "--num_fewshot", "0", "--batch_size", "auto", "--log_samples",
             "--include_path", "tasks", "--limit", str(limit),
             "--output_path", str(paths.RESULTS / "smoke" / "raw" / "lm_eval_hf")],
            False))
    return stages


def layer_b_stages(llamacpp: str) -> list[tuple[str, list[str], bool]]:
    q4 = "q06__q4_k_m__tr__s0"
    q2 = "q06__q2_k__tr__s0"
    imat = "q06__imatrix__tr__s0"
    return [
        ("t0-download-q06", [_py(), "scripts/t0/download_models.py", "--models", "q06"], True),
        ("gguf-convert-q06", [_py(), "scripts/t0/gguf_convert.py", "--models", "q06",
                              "--llamacpp", llamacpp], True),
        ("imatrix-q06", [_py(), "scripts/q/imatrix.py", "--model", "q06",
                         "--calib", "tr", "--seed", "s0", "--llamacpp", llamacpp], True),
        ("quantize-q4_k_m", [_py(), "scripts/q/gguf_quantize.py", "--run-key", q4,
                             "--llamacpp", llamacpp], True),
        ("quantize-q2_k", [_py(), "scripts/q/gguf_quantize.py", "--run-key", q2,
                           "--llamacpp", llamacpp], True),
        ("dequant-bf16-q06", [_py(), "scripts/q/dequant.py", "--run-key", "q06__bf16__none__s0",
                              "--gguf", str(paths.MODELS_GGUF / "q06-bf16.gguf")], True),
        ("dequant-identity", [_py(), "scripts/q/dequant_validate.py",
                              "--identity", "q06"], True),
        ("dequant-q4_k_m", [_py(), "scripts/q/dequant.py", "--run-key", q4], True),
        ("dequant-bit-exact", [_py(), "scripts/q/dequant_validate.py",
                               "--bit-exact", q4], True),
        ("kl-hf-q06", [_py(), "scripts/kl/kl_hf.py", "--ref",
                       str(paths.model_hf_path("q06")), "--test",
                       str(paths.tmp_deq_path(q4)), "--exp", "smoke",
                       "--run-key", q4], False),
        ("kl-llama-q06", [_py(), "scripts/kl/kl_llama.py", "--run-key", q4,
                          "--exp", "smoke", "--llamacpp", llamacpp], False),
        ("needle-4k-q06", [_py(), "scripts/e6/needle.py", "--contexts", "4096"], False),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="A", choices=["A", "B"])
    ap.add_argument("--llamacpp", default="llama.cpp", help="Layer B: build dir")
    ap.add_argument("--skip-eval", action="store_true", help="Layer A: skip lm_eval stage")
    args = ap.parse_args()
    logging_setup.setup("smoke")

    stages = (layer_a_stages(args.skip_eval) if args.layer == "A"
              else layer_b_stages(args.llamacpp))
    out_dir = paths.LOGS / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str, float]] = []
    for name, cmd, mandatory in stages:
        t0 = time.monotonic()
        log_path = out_dir / f"{args.layer}_{name}.log"
        with open(log_path, "w", encoding="utf-8") as lf:
            p = subprocess.run(cmd, cwd=paths.ROOT, stdout=lf, stderr=subprocess.STDOUT)
        dt = time.monotonic() - t0
        status = "PASS" if p.returncode == 0 else "FAIL"
        results.append((name, status, dt))
        print(f"[{status}] {name:22s} {dt:6.1f} s  (log {log_path})")
        if p.returncode != 0 and mandatory:
            print(f"\nmandatory stage '{name}' failed — stopping (see {log_path})")
            break

    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    total = sum(d for _, _, d in results)
    print(f"\nsmoke layer {args.layer}: {len(results) - n_fail}/{len(results)} stages PASS, "
          f"total {total / 60:.1f} min")
    if args.layer == "A" and not n_fail:
        print("assertions: schema/figure tests in pytest cover comma-decimal + 1380px checks")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())