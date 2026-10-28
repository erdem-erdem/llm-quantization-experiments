"""E0 gate over rows shaped like kl_llama / kl_hf / parse_lm_eval output."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import io, paths  # noqa: E402
from scripts.exp import e0_reliability  # noqa: E402


def _vec(exp, rk, task, correct):
    p = paths.sample_vector_path(exp, rk, task)
    p.parent.mkdir(parents=True, exist_ok=True)
    head = {"task": task, "n": len(correct), "metric": "acc", "dataset_sha": "x", "run_key": rk}
    p.write_text(json.dumps(head) + "\n" + "".join(
        json.dumps({"doc_id": i, "correct": int(c)}) + "\n" for i, c in enumerate(correct)),
        encoding="utf-8")


def _kl(exp, quant, calib, value):
    io.jsonl_append(paths.runs_jsonl(exp), {
        "exp": exp, "model": "q8", "quant": quant, "calib": calib, "seed": "s0",
        "task": "kl_havadis", "metric": "mean_kld", "value": value, "n": 1,
        "engine": "x", "run_id": f"q8__{quant}__{calib}__s0__kl_havadis"})


def test_gate_passes_on_consistent_data(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results")
    rng = np.random.default_rng(1)
    for task, n in (("belebele_tr", 900), ("arc_tr", 1172)):
        v = (rng.random(n) < 0.7).astype(np.int8)
        _vec("E1", "q8__bf16__none__s0", task, v)
        _vec("E0", "q8__bf16eager__none__s0", task, v)  # identical -> zero noise floor
    for quant, calib, kl in (("q8_0", "none", 0.001), ("q4_k_m", "tr", 0.02),
                             ("q3_k_m", "tr", 0.06), ("q2_k", "tr", 0.30)):
        _kl("E1", quant, calib, kl)
    _kl("E0", "q4_k_m", "tr", 0.021)
    _kl("E0", "q2_k", "tr", 0.29)
    assert e0_reliability.compute_gate() == 0
    gate = json.loads((paths.RESULTS / "E0" / "gate.json").read_text(encoding="utf-8"))
    assert gate["pass"] is True, gate["reasons"]
