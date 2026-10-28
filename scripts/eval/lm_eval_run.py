"""lm_eval wrapper (plan §6.5): one vLLM process per run evaluates all tasks for that run.

lm_eval --model vllm --model_args pretrained=<path>,dtype=bfloat16,gpu_memory_utilization=0.9,
max_model_len=4096,enable_prefix_caching=True --tasks <t1,t2> --num_fewshot 0 --batch_size auto
--log_samples --include_path tasks --output_path results/<exp>/raw/<run_key>
GGUF runs point --path at models/tmp_deq/<run_key> (engine=vllm-deq); GPTQ/FP8 at
models/ct/<run_key>; BF16 at models/hf/<model>. E0 eager runs add enforce_eager=True.
The wrapper asserts apply_chat_template=false and num_fewshot=0 and logs both in run_meta.
Task ids (belebele_tr, ...) are mapped to harness names via configs/tasks.yaml; on success the
samples are parsed into sample vectors + runs.jsonl rows (parse_lm_eval) and a vllm-deq
checkpoint (models/tmp_deq/<run_key>) is deleted unless --keep-deq.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, ids, io, logging_setup, paths  # noqa: E402


def resolve_path(run_key: str, engine: str) -> Path:
    model, quant, calib, seed = ids.parse_run_key(run_key)
    if engine == "vllm-deq":
        return paths.tmp_deq_path(run_key)
    if quant in ("fp8", "gptq", "rtn"):
        return paths.ct_path(run_key)
    return paths.model_hf_path(model)


def build_cmd(run_key: str, path: Path, tasks: list[str], exp: str, engine: str,
              eager: bool = False, limit: int | None = None) -> list[str]:
    model_args = [f"pretrained={path}", "dtype=bfloat16", "gpu_memory_utilization=0.9",
                  "max_model_len=4096", "enable_prefix_caching=True"]
    if eager:
        model_args.append("enforce_eager=True")
    out = paths.raw_dir(exp, run_key)
    cmd = ["lm_eval", "--model", "vllm",
           "--model_args", ",".join(model_args),
           "--tasks", ",".join(tasks),
           "--num_fewshot", "0", "--batch_size", "auto", "--log_samples",
           "--include_path", "tasks", "--output_path", str(out)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-key", required=True)
    ap.add_argument("--path", default=None, help="override the model path")
    ap.add_argument("--tasks", required=True, help="comma list")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--engine", default="vllm", choices=["vllm", "vllm-deq"])
    ap.add_argument("--eager", action="store_true", help="E0: enforce_eager=True")
    ap.add_argument("--limit", type=int, default=None, help="smoke: lm_eval --limit")
    ap.add_argument("--keep-deq", action="store_true",
                    help="keep models/tmp_deq/<run_key> (a later job still needs it)")
    args = ap.parse_args()
    logging_setup.setup("eval_lm_eval_run")

    path = Path(args.path) if args.path else resolve_path(args.run_key, args.engine)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing (dequant/quantize step first?)")
    tasks = args.tasks.split(",")
    harness = [config.load("tasks")["tasks"][t]["harness"] for t in tasks]
    cmd = build_cmd(args.run_key, path, harness, args.exp, args.engine,
                    eager=args.eager, limit=args.limit)

    # wrapper assertions, recorded in run_meta extras
    log = logging.getLogger("eval_lm_eval_run")
    log.info("assertions: apply_chat_template=False num_fewshot=0 engine=%s eager=%s",
             args.engine, args.eager)

    from scripts.common.run_meta import build_meta
    from datetime import datetime
    t0 = datetime.now()
    from scripts.common.shell import run
    proc = run(cmd, log_path=paths.LOGS / f"lm_eval_{args.exp}_{args.run_key}.log",
               check=False, cwd=paths.ROOT)
    meta = build_meta(exp=args.exp, job="lm_eval", run_key=args.run_key, cmd=cmd,
                      t0=t0, t1=datetime.now(), returncode=proc.returncode, n_gpus=1,
                      seed=ids.parse_run_key(args.run_key)[3],
                      extras={"tasks": tasks, "engine": args.engine, "eager": args.eager,
                              "path": str(path), "apply_chat_template": False,
                              "num_fewshot": 0})
    dest = paths.run_meta_path(args.exp, args.run_key, "lm_eval")
    dest.parent.mkdir(parents=True, exist_ok=True)
    io.atomic_write_text(json.dumps(meta, indent=1) + "\n", dest)
    log.info("run_meta written to %s", dest)
    if proc.returncode != 0:
        return 1
    from scripts.eval.parse_lm_eval import parse_run
    from scripts.q.sizes import run_sizes
    parse_run(args.exp, args.run_key, tasks, args.engine, run_sizes(args.run_key), args.limit)
    if args.engine == "vllm-deq" and not args.path and not args.keep_deq:
        from scripts.q.dequant_validate import cleanup
        cleanup(args.run_key)  # a failed eval keeps the checkpoint so the job can be retried
    return 0


if __name__ == "__main__":
    sys.exit(main())