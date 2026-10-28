"""Parse lm_eval sample logs (plan §6.5) → SampleVector + RunRow.

Reads results/<exp>/raw/<run_key>/**/samples_<harness task>_*.jsonl; per-sample correctness from the
metric field (acc for Belebele/GPQA, acc_norm for ARC-*); value = mean, n = count. Asserts
n matches configs/tasks.yaml and orders samples by doc_id. Appends to runs.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, ids, io, logging_setup, paths  # noqa: E402
from scripts.common.schemas import SampleEntry, SampleHeader, RunRow  # noqa: E402


def find_sample_file(raw_dir: Path, task: str) -> Path:
    # lm_eval nests its output under <output_path>/<sanitized model_args>/ and names samples
    # after the harness task (belebele_tur_Latn), not our task id
    harness = config.load("tasks")["tasks"][task]["harness"]
    cands = sorted(raw_dir.rglob(f"samples_{harness}_*.jsonl"), key=lambda p: p.name)
    if not cands:
        raise FileNotFoundError(f"no samples file for {task} in {raw_dir}")
    return cands[-1]  # lm_eval names with a timestamp; take the latest


def parse_samples(path: Path, metric: str, task: str, run_key: str,
                  limit: int | None = None) -> tuple[SampleHeader, list[SampleEntry], float]:
    rows = io.read_jsonl(path)
    entries: list[SampleEntry] = []
    for r in rows:
        doc_id = int(r["doc_id"])
        m = r[metric]
        correct = int(bool(m))
        entries.append(SampleEntry(doc_id=doc_id, correct=correct))
    entries.sort(key=lambda e: e.doc_id)
    n = len(entries)
    tcfg = config.load("tasks")["tasks"][task]
    expected = tcfg["n"] if limit is None else min(limit, tcfg["n"])  # smoke: lm_eval --limit
    if n != expected:
        raise RuntimeError(f"{task}: n={n} != expected {expected}")
    value = sum(e.correct for e in entries) / n
    header = SampleHeader(task=task, n=n, metric=tcfg["metric"],
                          dataset_sha=hashlib.sha256(path.read_bytes()).hexdigest(),
                          run_key=run_key)
    return header, entries, value


def parse_run(exp: str, run_key: str, tasks: list[str], engine: str, sizes: dict,
              limit: int | None = None) -> None:
    """Write one SampleVector + one RunRow per task for a finished lm_eval run."""
    model, quant, calib, seed = ids.parse_run_key(run_key)
    raw = paths.raw_dir(exp, run_key)
    runs = paths.runs_jsonl(exp)
    have = {r["run_id"] for r in io.read_jsonl(runs)} if runs.exists() else set()
    for task in tasks:
        metric_cfg = config.load("tasks")["tasks"][task]["metric"]
        f = find_sample_file(raw, task)
        header, entries, value = parse_samples(f, metric_cfg, task, run_key, limit)
        vec_path = paths.sample_vector_path(exp, run_key, task)
        io.atomic_write_text(
            json.dumps(header.model_dump(), ensure_ascii=False) + "\n"
            + "".join(json.dumps(e.model_dump()) + "\n" for e in entries),
            vec_path)
        run_id = f"{run_key}__{task}"
        if run_id in have:  # rerun of an already-parsed run: keep runs.jsonl free of duplicates
            print(f"{task}: run_id {run_id} already in {runs.name}, row not appended")
            continue
        # no delta fields here; build_master computes deltas from the vectors
        row = RunRow(exp=exp, model=model, quant=quant, calib=calib, seed=seed,
                     task=task, metric=header.metric, value=value, n=header.n,
                     bpw=sizes.get("bpw"), bpw_body=sizes.get("bpw_body"),
                     size_bytes=sizes.get("size_bytes"), engine=engine,
                     run_id=run_id).model_dump()
        io.jsonl_append(runs, row)
        print(f"{task}: n={header.n} value={value:.4f} -> {vec_path.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--run-key", required=True)
    ap.add_argument("--tasks", required=True, help="comma list, same as the eval run")
    ap.add_argument("--engine", default="vllm")
    ap.add_argument("--bpw", type=float, default=None)
    ap.add_argument("--size-bytes", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="the eval's lm_eval --limit (smoke)")
    args = ap.parse_args()
    logging_setup.setup("eval_parse_lm_eval")
    parse_run(args.exp, args.run_key, args.tasks.split(","), args.engine,
              {"bpw": args.bpw, "size_bytes": args.size_bytes}, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
