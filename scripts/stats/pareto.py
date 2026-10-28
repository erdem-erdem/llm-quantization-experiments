"""E3 — Pareto post-processing (plan §6.8, spec E3): no new runs, only E1's points.

Points = every E1 (model, condition) incl. BF16: size (GB) and tr+ARC mean accuracy. The
accuracy CI of a quantized point is its BF16 accuracy + the paired Δ CI (BF16 sampling noise is
ignored; BF16 itself has a zero-width interval). Frontier = non-dominated (size↓, accuracy↑).
Consecutive frontier points whose accuracy CIs overlap are "ayırt edilemez". For budgets
8/16/24 GB the top-3 (model, condition) combos whose estimated run memory — weights + KV for a
4K context + 1 GB — fits the budget, ranked by accuracy across ALL models.
Writes figures/E3_budget_top3.csv + results/E3/extras.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import io, logging_setup, paths  # noqa: E402

BUDGETS_GB = [8, 16, 24]
EXAMPLE_PAIRS = [  # example pairings, explicit rows (model, condition)
    ("q8", "q4_k_m"), ("q14", "q4_k_m"), ("q8", "q3_k_m"), ("q32", "q3_k_m"),
]


def kv_bytes_4k(card: dict) -> int:
    """kv_bytes = 2·layers·kv_heads·head_dim·4096·2 (plan §6.8)."""
    return 2 * int(card["layers"]) * int(card["kv_heads"]) * int(card["head_dim"]) \
        * 4096 * 2


def build_points(master: pd.DataFrame) -> pd.DataFrame:
    """E1 points: model, quant, size_gb, acc (fraction), ci_low, ci_high (accuracy scale)."""
    e1 = master[master["exp"] == "E1"]
    rows = []
    for model, g in e1.groupby("model"):
        bf = g[g["quant"] == "bf16"]
        accs = [bf[bf["task"] == t]["value"] for t in ("belebele_tr", "arc_tr")]
        if any(a.empty for a in accs):
            continue  # no BF16 reference for this model: nothing to anchor the CI on
        bf_acc = float(sum(float(a.iloc[0]) for a in accs) / 2)
        size = bf["size_bytes"].dropna()
        if len(size):
            rows.append({"model": model, "quant": "bf16", "size_gb": float(size.iloc[0]) / 1e9,
                         "acc": bf_acc, "ci_low": bf_acc, "ci_high": bf_acc})
        for quant, gq in g[g["quant"] != "bf16"].groupby("quant"):
            avg = gq[gq["task"] == "tr_arc_avg"]
            size = gq["size_bytes"].dropna()
            if avg.empty or size.empty:
                continue
            r = avg.iloc[0]
            rows.append({"model": model, "quant": quant, "size_gb": float(size.iloc[0]) / 1e9,
                         "acc": float(r["value"]),
                         "ci_low": bf_acc + float(r["ci_low"]) / 100.0,
                         "ci_high": bf_acc + float(r["ci_high"]) / 100.0})
    return pd.DataFrame(rows, columns=["model", "quant", "size_gb", "acc", "ci_low", "ci_high"])


def pareto_frontier(points: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated by (size up, accuracy down), ordered by size."""
    pts = points.sort_values(["size_gb", "acc"], ascending=[True, False])
    keep, best = [], -1.0
    for i, r in pts.iterrows():
        if float(r["acc"]) > best:
            keep.append(i)
            best = float(r["acc"])
    return pts.loc[keep]


def indistinguishable_pairs(front: pd.DataFrame) -> list[str]:
    """Consecutive frontier points with overlapping accuracy CIs."""
    out = []
    rows = list(front.to_dict("records"))
    for a, b in zip(rows, rows[1:]):
        if a["ci_low"] <= b["ci_high"] and b["ci_low"] <= a["ci_high"]:
            out.append(f"{a['model']}__{a['quant']} ~ {b['model']}__{b['quant']}")
    return out


def budget_top3(points: pd.DataFrame, cards: dict[str, dict], budgets: list[int]) -> dict:
    """Per budget: the 3 most accurate (model, condition) combos whose est. memory fits."""
    pts = points.copy()
    pts["kv_gb"] = pts["model"].map(lambda m: kv_bytes_4k(cards[m]) / 1e9 if m in cards else None)
    pts = pts.dropna(subset=["kv_gb"])
    pts["est_mem_gb"] = pts["size_gb"] + pts["kv_gb"] + 1.0
    out = {}
    for b in budgets:
        fit = pts[pts["est_mem_gb"] <= b].sort_values("acc", ascending=False).head(3)
        out[f"{b}GB"] = [{"rank": i + 1, "model": r["model"], "quant": r["quant"],
                          "size_gb": round(r["size_gb"], 2), "kv4k_gb": round(r["kv_gb"], 2),
                          "est_mem_gb": round(r["est_mem_gb"], 2), "acc": round(r["acc"], 4),
                          "budget_gb": b}
                         for i, r in enumerate(fit.to_dict("records"))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="8,16,24")
    args = ap.parse_args()
    logging_setup.setup("e3_pareto")

    master = pd.read_csv(paths.master_csv())
    pts = build_points(master)
    if pts.empty:
        raise RuntimeError("master.csv has no E1 points with a BF16 reference (build_master first)")
    cards = {r["model"]: r for r in pd.read_csv(paths.RESULTS / "model_cards.csv")
             .to_dict("records")}
    front = pareto_frontier(pts)
    top3 = budget_top3(pts, cards, [int(b) for b in args.budgets.split(",")])
    examples = pts[[(m, q) in EXAMPLE_PAIRS for m, q in zip(pts["model"], pts["quant"])]]
    extras = {"frontier": [{k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in r.items()} for r in front.to_dict("records")],
              "indistinguishable": indistinguishable_pairs(front),
              "budget_top3": top3,
              "examples": examples.round(4).to_dict("records")}
    io.atomic_write_json(paths.RESULTS / "E3" / "extras.json", extras)

    out = paths.FIGURES / "E3_budget_top3.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    flat = [row for rows3 in top3.values() for row in rows3]
    pd.DataFrame(flat).to_csv(out, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote E3/extras.json ({len(front)} frontier points) + {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
