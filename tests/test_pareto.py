"""E3 post-processing on synthetic E1 points: BF16 point, frontier, overlap marks, per-budget top-3."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.stats import pareto  # noqa: E402


def _m():
    rows = []

    def add(model, quant, task, value, size, **kw):
        rows.append({"exp": "E1", "model": model, "quant": quant, "task": task, "value": value,
                     "size_bytes": size * 1e9, "ci_low": None, "ci_high": None, **kw})
    for model, bf, size16 in (("q8", 0.80, 16.0), ("q4", 0.70, 8.0)):
        add(model, "bf16", "belebele_tr", bf, size16)
        add(model, "bf16", "arc_tr", bf, size16)
        add(model, "q4_k_m", "tr_arc_avg", bf - 0.02, size16 / 3.3, ci_low=-3.0, ci_high=-1.0)
        add(model, "q2_k", "tr_arc_avg", bf - 0.20, size16 / 5.0, ci_low=-22.0, ci_high=-18.0)
    return pd.DataFrame(rows)


def test_points_include_bf16_and_accuracy_scale_ci():
    pts = pareto.build_points(_m()).set_index(["model", "quant"])
    assert pts.loc[("q8", "bf16"), "acc"] == 0.80 and pts.loc[("q8", "bf16"), "ci_low"] == 0.80
    q = pts.loc[("q8", "q4_k_m")]
    assert abs(q["ci_low"] - 0.77) < 1e-9 and abs(q["ci_high"] - 0.79) < 1e-9  # bf16 + Δ CI/100


def test_frontier_overlap_and_budget_ranking_across_models():
    pts = pareto.build_points(_m())
    front = pareto.pareto_frontier(pts)
    assert front["acc"].is_monotonic_increasing                       # accuracy rises with size
    assert list(front["size_gb"]) == sorted(front["size_gb"])
    cards = {m: {"layers": 36, "kv_heads": 8, "head_dim": 128} for m in ("q8", "q4")}
    top = pareto.budget_top3(pts, cards, [8, 16, 24])
    assert len(top["8GB"]) <= 3 and all(r["est_mem_gb"] <= 8 for r in top["8GB"])
    accs = [r["acc"] for r in top["24GB"]]
    assert accs == sorted(accs, reverse=True)                          # ranked across all models
    assert {r["model"] for r in top["24GB"]} == {"q8", "q4"}
    a = {"model": "q4", "quant": "x", "size_gb": 1, "acc": .5, "ci_low": .48, "ci_high": .52}
    b = {"model": "q8", "quant": "y", "size_gb": 2, "acc": .51, "ci_low": .49, "ci_high": .53}
    assert pareto.indistinguishable_pairs(pd.DataFrame([a, b])) == ["q4__x ~ q8__y"]
