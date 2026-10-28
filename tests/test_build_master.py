"""build_master on rows shaped like parse_lm_eval output (no run_key column, cross-exp refs)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import paths  # noqa: E402
from scripts.stats import build_master  # noqa: E402


def _vector(exp: str, rk: str, task: str, correct: np.ndarray) -> None:
    p = paths.sample_vector_path(exp, rk, task)
    p.parent.mkdir(parents=True, exist_ok=True)
    head = {"task": task, "n": len(correct), "metric": "acc", "dataset_sha": "x", "run_key": rk}
    p.write_text(json.dumps(head) + "\n" + "".join(
        json.dumps({"doc_id": i, "correct": int(c)}) + "\n" for i, c in enumerate(correct)),
        encoding="utf-8")


def _row(exp, model, quant, calib, task, value):
    return {"exp": exp, "model": model, "quant": quant, "calib": calib, "seed": "s0",
            "task": task, "metric": "acc", "value": value, "n": 900, "bpw": None,
            "bpw_body": None, "size_bytes": None, "engine": "vllm",
            "run_id": f"{model}__{quant}__{calib}__s0__{task}"}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results")
    return tmp_path


def test_cross_exp_reference_mcnemar_and_holm(sandbox):
    rng = np.random.default_rng(0)
    ref = (rng.random(900) < 0.7).astype(np.int8)
    bad = ref.copy()
    bad[np.flatnonzero(ref == 1)[:90]] = 0      # 90 c2w flips, no w2c -> clearly significant
    same = ref.copy()
    for task in ("belebele_tr", "arc_tr"):
        _vector("E1", "q17__bf16__none__s0", task, ref)
    for task in ("belebele_tr", "arc_tr"):
        _vector("E2", "q17__gptq__en__s0", task, bad)
        _vector("E2", "q17__gptq__tr__s0", task, same)
    rows = [_row("E2", "q17", "gptq", "en", t, float(bad.mean())) for t in ("belebele_tr", "arc_tr")]
    rows += [_row("E2", "q17", "gptq", "tr", t, float(same.mean())) for t in ("belebele_tr", "arc_tr")]
    out = build_master.build_exp("E2", rows)
    by = {(r["run_key"], r["task"]): r for r in out}
    en = by[("q17__gptq__en__s0", "belebele_tr")]
    assert en["delta"] == pytest.approx(-10.0) and en["flip_c2w"] == pytest.approx(0.1)
    assert en["p_mcnemar"] < 1e-10 and en["holm_sig"] is True
    tr = by[("q17__gptq__tr__s0", "belebele_tr")]
    assert tr["delta"] == 0.0 and tr["holm_sig"] is False
    assert ("q17__gptq__en__s0", "tr_arc_avg") in by   # derived row found its E1 references


def test_defect_a_missing_sample_vector_handling(sandbox):
    """Test that non-MC metrics without sample vectors pass through unchanged,
    while MC tasks get marked with reason 'missing sample vector'."""
    # Create BF16 reference for MC task
    rng = np.random.default_rng(0)
    ref = (rng.random(900) < 0.7).astype(np.int8)
    _vector("E1", "q4__bf16__none__s0", "belebele_tr", ref)

    # Test rows: mix of MC tasks (no vector) and non-MC tasks (no vector)
    rows = [
        # BF16 baseline
        _row("E1", "q4", "bf16", "none", "belebele_tr", float(ref.mean())),
        # Q4_K_M KL (non-MC, no sample vector) - should pass through
        {"exp": "E1", "model": "q4", "quant": "q4_k_m", "calib": "tr", "seed": "s0",
         "task": "kl_havadis", "metric": "kl", "value": 0.25, "n": None, "bpw": 4.0,
         "bpw_body": None, "size_bytes": 1000000, "engine": "llama.cpp",
         "run_id": "q4__q4_k_m__tr__s0__kl"},
        # Q8_0 speed/serve (non-MC, no sample vector) - should pass through
        {"exp": "E1", "model": "q4", "quant": "q8_0", "calib": "none", "seed": "s0",
         "task": "serve", "metric": "output_tok_s", "value": 150.5, "n": None, "bpw": 8.0,
         "bpw_body": None, "size_bytes": 2000000, "engine": "vllm",
         "run_id": "q4__q8_0__none__s0__serve"},
        # E6 KV cache (non-MC, no sample vector) - should pass through
        {"exp": "E6", "model": "q8", "quant": "kv_q8_0", "calib": "none", "seed": "s0",
         "task": "kv_cache_4k", "metric": "mb", "value": 512.0, "n": None, "bpw": None,
         "bpw_body": None, "size_bytes": None, "engine": "llama.cpp",
         "run_id": "q8__kv_q8_0__none__s0__kv4k"},
        # Belebele-tr quantized (MC task, no vector) - should get reason
        _row("E1", "q4", "q4_k_m", "tr", "belebele_tr", 0.65),
    ]

    out = build_master.build_exp("E1", rows)
    by = {(r["run_key"], r["task"]): r for r in out}

    # Check KL row passes through unchanged
    kl_row = by[("q4__q4_k_m__tr__s0", "kl_havadis")]
    assert kl_row["value"] == 0.25
    assert kl_row.get("reason") is None or "reason" not in kl_row
    assert kl_row.get("delta") is None  # no comparison for non-MC

    # Check speed/serve row passes through unchanged
    serve_row = by[("q4__q8_0__none__s0", "serve")]
    assert serve_row["value"] == 150.5
    assert serve_row.get("reason") is None or "reason" not in serve_row

    # Check KV cache row passes through unchanged
    kv_row = by[("q8__kv_q8_0__none__s0", "kv_cache_4k")]
    assert kv_row["value"] == 512.0
    assert kv_row.get("reason") is None or "reason" not in serve_row

    # Check MC task without vector gets reason
    mc_row = by[("q4__q4_k_m__tr__s0", "belebele_tr")]
    assert mc_row["value"] == 0.65  # value preserved
    assert mc_row["delta"] is None
    assert mc_row.get("reason") == "missing sample vector"



def test_e2_tr_s0_cells_reuse_e1_rows_and_vectors(sandbox):
    rng = np.random.default_rng(1)
    ref = (rng.random(900) < 0.7).astype(np.int8)
    q = ref.copy()
    q[np.flatnonzero(ref == 1)[:45]] = 0
    for task in ("belebele_tr", "belebele_en", "arc_tr", "arc_en"):
        _vector("E1" if task != "arc_en" else "E2", "q17__bf16__none__s0", task, ref)
    for task in ("belebele_tr", "belebele_en", "arc_tr"):
        _vector("E1", "q17__gptq__tr__s0", task, q)
    _vector("E2", "q17__gptq__tr__s0", "arc_en", q)
    e1_rows = [_row("E1", "q17", "gptq", "tr", t, float(q.mean()))
               for t in ("belebele_tr", "belebele_en", "arc_tr")]
    p = paths.runs_jsonl("E1")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in e1_rows), encoding="utf-8")
    e2_rows = [_row("E2", "q17", "gptq", "tr", "arc_en", float(q.mean()))]
    out = build_master.build_exp("E2", e2_rows)
    by = {(r["run_key"], r["task"]): r for r in out}
    rk = "q17__gptq__tr__s0"
    assert (rk, "belebele_tr") in by and by[(rk, "belebele_tr")]["exp"] == "E2"
    assert (rk, "tr_arc_avg") in by and (rk, "en_avg") in by   # both derived rows now exist
    assert by[(rk, "en_avg")]["delta"] == pytest.approx(-5.0)


def test_tr_en_gap_row_and_k05_breaking_points_in_both_orders(sandbox):
    rng = np.random.default_rng(2)
    ref = (rng.random(900) < 0.7).astype(np.int8)
    q_tr = ref.copy()
    q_tr[np.flatnonzero(ref == 1)[:90]] = 0       # Turkish loses 10 points
    q_en = ref.copy()
    q_en[np.flatnonzero(ref == 1)[:18]] = 0       # English loses 2 points
    for t in ("belebele_tr", "belebele_en"):
        _vector("E1", "q8__bf16__none__s0", t, ref)
    _vector("E1", "q8__q4_k_m__tr__s0", "belebele_tr", q_tr)
    _vector("E1", "q8__q4_k_m__tr__s0", "belebele_en", q_en)
    rows = [_row("E1", "q8", "q4_k_m", "tr", t, 0.6) for t in ("belebele_tr", "belebele_en")]
    out = build_master.build_exp("E1", rows)
    gap = next(r for r in out if r["task"] == "tr_en_gap")
    assert gap["delta"] == pytest.approx(8.0)            # loss_tr 10 - loss_en 2, TR loses more
    assert gap["ci_low"] < 8.0 < gap["ci_high"]

    # K-05: bpw order puts gptq (4.2 bpw) below q4_k_m (4.9); nominal order ties both at 4 bits
    # and breaks the tie by bpw, so here the two orders pick different first-breaking conditions
    # when only gptq breaks and fp8/q8_0 do not.
    def cond(q, bpw, delta, hi):
        return {"model": "q8", "quant": q, "task": "tr_arc_avg", "bpw": bpw,
                "delta": delta, "ci_high": hi}
    rows = [cond("fp8", 8.0, -0.1, 0.5), cond("q8_0", 8.5, -0.2, 0.4),
            cond("q4_k_m", 4.9, -1.0, 0.2), cond("gptq", 4.2, -3.0, -0.5),
            cond("q3_k_m", 3.9, -5.0, -2.0), cond("q2_k", 3.35, -20.0, -15.0)]
    bp = build_master.breaking_points(rows)["q8"]
    assert bp["bpw_order"] == "gptq" and bp["nominal_order"] == "gptq" and bp["agree"]
    rows[2] = cond("q4_k_m", 4.9, -2.5, -0.3)           # now q4_k_m breaks too
    bp = build_master.breaking_points(rows)["q8"]
    assert bp["bpw_order"] == "q4_k_m"                  # 4.9 bpw comes before 4.2 going down
    assert bp["unordered"] == []
