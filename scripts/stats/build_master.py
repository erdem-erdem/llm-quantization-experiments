"""Build results/master.csv from runs.jsonl + sample vectors (plan §6.7).

Statistics live here and only here: figures/tables read master.csv (+ E0 noise band +
documented experiment extras) and never re-compute statistics.

Families for Holm: per (exp, task) across all quants/models for per-task rows; one family
for all tr_arc_avg rows of the experiment. Documented in the table footnotes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, io, logging_setup, paths  # noqa: E402
from scripts.stats import matched  # noqa: E402
from scripts.stats.tests import holm, mcnemar_exact  # noqa: E402

AVG_PAIRS = {"tr_arc_avg": ("belebele_tr", "arc_tr"), "en_avg": ("belebele_en", "arc_en")}
MC_TASKS = {"belebele_tr", "belebele_en", "arc_tr", "arc_en", "gpqa_tr"}
# Nominal weight bits per E1 condition (name-level; K-05 reports the breaking point under both
# this order and the measured-bpw order — the primary one — so the reader can see if they differ)
NOMINAL_BITS = {"fp8": 8, "q8_0": 8, "gptq": 4, "q4_k_m": 4, "q3_k_m": 3, "q2_k": 2}


def load_vector(exp: str, run_key: str, task: str) -> np.ndarray:
    p = paths.sample_vector_path(exp, run_key, task)
    if not p.exists():
        raise FileNotFoundError(p)
    rows = io.read_jsonl(p)
    header, entries = rows[0], rows[1:]
    assert header["task"] == task and header["n"] == len(entries), f"bad header in {p}"
    assert entries == sorted(entries, key=lambda e: e["doc_id"]), f"not sorted by doc_id: {p}"
    return np.array([e["correct"] for e in entries], dtype=np.int8)


def try_vector(exp: str, run_key: str, task: str) -> np.ndarray | None:
    try:
        return load_vector(exp, run_key, task)
    except FileNotFoundError:
        # E2's (tr, s0) cells reuse E1's production AND its evals (only ARC-en is new in E2)
        if exp == "E2":
            try:
                return load_vector("E1", run_key, task)
            except FileNotFoundError:
                return None
        return None


def reused_e1_rows(rows: list[dict]) -> list[dict]:
    """E2 run rows for its (tr, s0) cells that live in E1 (belebele_tr/en, arc_tr, KL …).

    Without them E2 has no tr_arc_avg / en_avg row for those cells, so the K-tr vs K-en gap
    and the seed gap |s0 - s1| could not be computed. Rows E2 measured itself win."""
    e2 = config.load("experiments")["E2"]
    p = paths.runs_jsonl("E1")
    if not p.exists():
        return []
    have = {(r["model"], r["quant"], r["calib"], r["seed"], r["task"], r["metric"])
            for r in rows}
    return [{**r, "exp": "E2"} for r in io.read_jsonl(p)
            if r["model"] in e2["models"] and r["quant"] in e2["methods"]
            and r["calib"] == "tr" and r["seed"] == "s0"
            and (r["model"], r["quant"], r["calib"], r["seed"], r["task"], r["metric"])
            not in have]


def ref_vector(exp: str, model: str, task: str) -> np.ndarray | None:
    """BF16 reference vector: the experiment's own, else E1's (E0/E2/E5/E6 reuse E1's BF16 runs)."""
    key = f"{model}__bf16__none__s0"
    v = try_vector(exp, key, task)
    return v if v is not None else try_vector("E1", key, task)


def build_exp(exp: str, rows: list[dict]) -> list[dict]:
    """rows: parsed RunRow dicts for one experiment; returns master rows."""
    if exp == "E2":
        rows = rows + reused_e1_rows(rows)
    out: list[dict] = []
    for r in rows:
        if r.get("value") is None:
            continue  # NA row, no sample vector
        m, q, c, s = r["model"], r["quant"], r["calib"], r["seed"]
        task = r["task"]
        rk = r.get("run_key") or f"{m}__{q}__{c}__{s}"
        if task in AVG_PAIRS:
            continue  # derived below
        vec_q = try_vector(exp, rk, task)
        base = {**r, "run_key": rk}  # RunRow has no run_key; downstream grouping needs it
        if q == "bf16":
            out.append({**base, "delta": None, "ci_low": None, "ci_high": None,
                        "flip_c2w": None, "flip_w2c": None, "flip_lo": None,
                        "flip_hi": None, "p_mcnemar": None, "holm_sig": False,
                        "chance": matched.chance_flag(vec_q) if vec_q is not None else None})
            continue
        if vec_q is None:
            # Non-BF16 row without sample vector
            if task in MC_TASKS:
                # MC accuracy tasks require sample vector for delta/flip computation
                out.append({**base, "delta": None, "ci_low": None, "ci_high": None,
                            "flip_c2w": None, "flip_w2c": None, "flip_lo": None, "flip_hi": None,
                            "p_mcnemar": None, "holm_sig": False, "chance": None,
                            "reason": "missing sample vector"})
            else:
                # Other metrics (KL, speed, KV cache) are valid measurements without per-item vectors
                out.append({**base, "delta": None, "ci_low": None, "ci_high": None,
                            "flip_c2w": None, "flip_w2c": None, "flip_lo": None, "flip_hi": None,
                            "p_mcnemar": None, "holm_sig": False, "chance": None})
            continue
        vec_ref = ref_vector(exp, m, task)
        if vec_ref is None:
            out.append({**base, "delta": None, "ci_low": None, "ci_high": None,
                        "flip_c2w": None, "flip_w2c": None, "flip_lo": None, "flip_hi": None,
                        "p_mcnemar": None, "holm_sig": False, "chance": None,
                        "reason": "missing bf16 reference vector"})
            continue
        delta, lo, hi = matched.paired_bootstrap(vec_ref, vec_q)
        c2w, w2c = matched.flips(vec_ref, vec_q)
        k = round((c2w + w2c) * len(vec_ref))
        flo, fhi = matched.wilson(k, len(vec_ref))
        # McNemar discordant pairs: b = ref right/quant wrong, c = ref wrong/quant right
        p = mcnemar_exact(int(((vec_ref == 1) & (vec_q == 0)).sum()),
                          int(((vec_ref == 0) & (vec_q == 1)).sum()))
        out.append({**base, "delta": delta, "ci_low": lo, "ci_high": hi,
                    "flip_c2w": c2w, "flip_w2c": w2c, "flip_lo": flo, "flip_hi": fhi,
                    "p_mcnemar": p, "holm_sig": False,
                    "chance": matched.chance_flag(vec_ref)})

    # derived average rows
    by_rk_task: dict[tuple[str, str], dict] = {}
    for r in out:
        by_rk_task.setdefault((r["run_key"], r["task"]), r)
    for avg_task, (t1, t2) in AVG_PAIRS.items():
        for (rk, task), r in by_rk_task.items():
            if task != t1:
                continue
            m, q, c, s = r["model"], r["quant"], r["calib"], r["seed"]
            if q == "bf16":
                continue
            r1 = by_rk_task.get((rk, t1))
            r2 = by_rk_task.get((rk, t2))
            v1 = try_vector(exp, rk, t1)
            v2 = try_vector(exp, rk, t2)
            u1 = ref_vector(exp, m, t1)
            u2 = ref_vector(exp, m, t2)
            if any(x is None for x in (r1, r2, v1, v2, u1, u2)):  # `in` would compare arrays
                continue
            delta, lo, hi = matched.paired_bootstrap_avg([(u1, v1), (u2, v2)])
            f_c2w = (r1["flip_c2w"] * len(v1) + r2["flip_c2w"] * len(v2)) / (len(v1) + len(v2))
            f_w2c = (r1["flip_w2c"] * len(v1) + r2["flip_w2c"] * len(v2)) / (len(v1) + len(v2))
            k = round((f_c2w + f_w2c) * (len(v1) + len(v2)))
            flo, fhi = matched.wilson(k, len(v1) + len(v2))
            out.append({
                "exp": exp, "model": m, "quant": q, "calib": c, "seed": s,
                "task": avg_task, "metric": "delta", "value": (r1["value"] + r2["value"]) / 2,
                "delta": delta, "ci_low": lo, "ci_high": hi,
                "flip_c2w": f_c2w, "flip_w2c": f_w2c, "flip_lo": flo, "flip_hi": fhi,
                "p_mcnemar": None, "holm_sig": False, "chance": None,
                "n": len(v1) + len(v2), "bpw": r["bpw"], "bpw_body": r["bpw_body"],
                "size_bytes": r["size_bytes"], "engine": r["engine"], "run_key": rk,
            })

    # TR-EN loss gap on the aligned Belebele rows (loss = reference - quant, in points, so a
    # positive gap = Turkish loses more than English); CI from one shared row resampling
    for (rk, task), r in list(by_rk_task.items()):
        if task != "belebele_tr" or r["quant"] == "bf16":
            continue
        m = r["model"]
        vs = [try_vector(exp, rk, "belebele_tr"), try_vector(exp, rk, "belebele_en"),
              ref_vector(exp, m, "belebele_tr"), ref_vector(exp, m, "belebele_en")]
        if any(v is None for v in vs):
            continue
        q_tr, q_en, r_tr, r_en = vs
        gap, lo, hi = matched.paired_bootstrap_lang_gap(r_tr, q_tr, r_en, q_en)
        out.append({
            "exp": exp, "model": m, "quant": r["quant"], "calib": r["calib"],
            "seed": r["seed"], "task": "tr_en_gap", "metric": "loss_gap", "value": gap,
            "delta": gap, "ci_low": lo, "ci_high": hi, "flip_c2w": None, "flip_w2c": None,
            "flip_lo": None, "flip_hi": None, "p_mcnemar": None, "holm_sig": False,
            "chance": None, "n": len(q_tr), "bpw": r["bpw"], "bpw_body": r["bpw_body"],
            "size_bytes": r["size_bytes"], "engine": r["engine"], "run_key": rk})

    # Holm families: per (exp, task) for per-task rows; one family per derived avg task
    fam_tasks = sorted({r["task"] for r in out})
    for fam in fam_tasks:
        # rows without a test (BF16 reference, derived averages) are not hypotheses of the family
        idx = [i for i, r in enumerate(out)
               if r["task"] == fam and r["p_mcnemar"] is not None]
        if not idx:
            continue
        pvals = [out[i]["p_mcnemar"] for i in idx]
        mask = holm(pvals)
        for i, sig in zip(idx, mask):
            out[i]["holm_sig"] = bool(sig)
    return out


def breaking_points(rows: list[dict]) -> dict:
    """K-05 per model from E1 tr_arc_avg rows: the first condition, going down in bits, with
    delta <= -2.0 and a bootstrap CI excluding 0. Primary order = measured bpw; the nominal-bit
    order (ties broken by bpw) is reported next to it. A condition without bpw is left out of
    the bpw order and listed under `unordered` (never guessed)."""
    out: dict[str, dict] = {}
    for model in sorted({r["model"] for r in rows}):
        conds = {r["quant"]: r for r in rows
                 if r["model"] == model and r["task"] == "tr_arc_avg"
                 and r.get("delta") is not None and r.get("ci_high") is not None}
        have_bpw = {q: r for q, r in conds.items() if r.get("bpw") is not None
                    and not pd.isna(r["bpw"])}
        deltas = {q: r["delta"] for q, r in conds.items()}
        highs = {q: r["ci_high"] for q, r in conds.items()}
        bpw = {q: float(r["bpw"]) for q, r in have_bpw.items()}
        by_bpw = matched.breaking_point(bpw, deltas, highs) if bpw else None
        nominal = {q: NOMINAL_BITS[q] + bpw.get(q, 0.0) / 1000.0 for q in conds
                   if q in NOMINAL_BITS}  # bpw only breaks ties between equal nominal bits
        by_nominal = matched.breaking_point(nominal, deltas, highs) if nominal else None
        out[model] = {"bpw_order": by_bpw, "nominal_order": by_nominal,
                      "agree": by_bpw == by_nominal,
                      "unordered": sorted(set(conds) - set(bpw))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", default="all", help="comma list or 'all'")
    args = ap.parse_args()
    logging_setup.setup("build_master")

    if args.exps == "all":
        exps = sorted(p.name for p in paths.RESULTS.iterdir()
                      if (p / "runs.jsonl").exists() and p.name.startswith(("E", "demo"))
                      and not p.name.endswith("smoke"))
    else:
        exps = args.exps.split(",")

    all_rows: list[dict] = []
    for exp in exps:
        rows = io.read_jsonl(paths.runs_jsonl(exp))
        built = build_exp(exp, rows)
        all_rows.extend(built)
        print(f"{exp}: {len(rows)} run rows -> {len(built)} master rows")

    cols = ["exp", "model", "quant", "calib", "seed", "task", "metric", "value",
            "delta", "ci_low", "ci_high", "flip_c2w", "flip_w2c", "flip_lo", "flip_hi",
            "p_mcnemar", "holm_sig", "chance", "n", "bpw", "bpw_body", "size_bytes",
            "engine", "run_key", "reason"]
    df = pd.DataFrame(all_rows, columns=cols)
    df.to_csv(paths.master_csv(), index=False, encoding="utf-8", lineterminator="\n")
    e1 = df[df["exp"] == "E1"].to_dict("records")
    if e1:
        bp = breaking_points(e1)
        io.atomic_write_json(paths.RESULTS / "E1" / "breaking_point.json", bp)
        for model, v in bp.items():
            print(f"K-05 {model}: bpw-order {v['bpw_order']}, nominal-order {v['nominal_order']}"
                  f"{'' if v['agree'] else '  (differ)'}")
    print(f"wrote {paths.master_csv()} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())