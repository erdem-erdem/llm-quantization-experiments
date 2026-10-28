"""E5 stage 2 — recipe search (plan §6.8): rank 252 tensors by KL/byte, tier assignment.

Tiers q6_k/q4_k/q3_k/q2_k with analytic bpw {6.5625, 4.5, 3.4375, 2.625}. n3 = 40% of
tensors, n4 = 2·n6, binary search n6 so the predicted size (targeted tensors at analytic
bpw, everything else at its actual Q3_K_M bytes) lands within ±1% of the uniform Q3_K_M
GGUF. Also builds the two random controls (same multiset per tensor type, shuffled across
layers → identical predicted size) and the q8 stage-3 transfer.
Writes results/E5/recipes.json: per-recipe llama-quantize --tensor-type args ("regex=TYPE").
Assign keys serialize as "layer|type".
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, logging_setup, paths  # noqa: E402

TIER_BPW = {"q6_k": 6.5625, "q4_k": 4.5, "q3_k": 3.4375, "q2_k": 2.625}
TENSOR_RE = re.compile(
    r"blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$")


def _read_csv(p: Path) -> list[dict]:
    import csv as _csv
    with open(p, encoding="utf-8") as f:
        return list(_csv.DictReader(f))


def load_map(layers: int) -> list[dict]:
    p = paths.RESULTS / "E5" / "map.csv"
    rows = _read_csv(p)
    if len(rows) != layers * 7:
        raise RuntimeError(f"{p}: expected {layers * 7} cells, got {len(rows)}")
    return rows


def dump_rows(run_key: str) -> list[dict]:
    return _read_csv(paths.tensor_types_csv(run_key))


def _key(l: int, t: str) -> str:
    return f"{l}|{t}"


def search_recipe(map_rows: list[dict], dump: list[dict], ref_key: str) -> dict:
    """Rank by KL/byte; n3 fixed at 40%, n4 = 2·n6, binary-search n6 within ±1% size."""
    rule = config.load("experiments")["E5"]["tier_rule"]
    numel: dict[tuple[int, str], int] = {}
    fixed_bytes = 0.0
    for row in dump:
        m = TENSOR_RE.match(row["name"])
        if m:
            numel[(int(m.group(1)), m.group(2))] = int(row["n_elements"])
        else:
            fixed_bytes += float(row["n_bytes"])  # embedding/output/norms keep Q3_K_M bytes
    scored = sorted(((float(r["kl"]) / float(r["bytes"]), int(r["layer"]), str(r["type"]))
                     for r in map_rows), reverse=True)
    n_tensors = len(scored)
    n3 = round(rule["n3_frac"] * n_tensors)

    def predicted(n6: int) -> float:
        n4 = round(rule["n4_per_n6"] * n6)
        tiers = (["q6_k"] * n6 + ["q4_k"] * n4 + ["q3_k"] * n3
                 + ["q2_k"] * max(0, n_tensors - n6 - n4 - n3))
        return fixed_bytes + sum(numel[(l, t)] * TIER_BPW[tier] / 8.0
                                 for tier, (_, l, t) in zip(tiers, scored))

    ref_size = sum(float(r["n_bytes"]) for r in dump_rows(ref_key))
    n6 = _binary_search(predicted, ref_size, rule["size_tol"], n_tensors, n3)
    if n6 is None:
        raise RuntimeError("no n6 lands within ±1% of the uniform Q3_K_M size")
    n4 = round(rule["n4_per_n6"] * n6)
    tiers = (["q6_k"] * n6 + ["q4_k"] * n4 + ["q3_k"] * n3
             + ["q2_k"] * max(0, n_tensors - n6 - n4 - n3))
    assign = {_key(l, t): tier for (_, l, t), tier in zip(scored, tiers)}
    return {"n6": n6, "n4": n4, "n3": n3, "n_tensors": n_tensors,
            "predicted_bytes": predicted(n6), "ref_bytes": ref_size, "assign": assign}


def _binary_search(predicted, ref_size: float, tol: float, n_tensors: int,
                   n3: int) -> int | None:
    lo, hi = 0, max(0, n_tensors - n3) // 3
    while lo <= hi:
        mid = (lo + hi) // 2
        size = predicted(mid)
        if abs(size - ref_size) <= tol * ref_size:
            return mid
        if size < ref_size:
            lo = mid + 1
        else:
            hi = mid - 1
    return None


def transfer_recipe(scored_order: list[tuple[float, int, str]], stage3_dump: list[dict],
                    stage3_ref_key: str) -> dict:
    """Stage 3: the q4 ranking applied to the stage-3 model's tensors, re-sized to its
    own Q3_K_M ±1% (counts re-searched, per-cell rule preserved)."""
    rule = config.load("experiments")["E5"]["tier_rule"]
    n_tensors = len(scored_order)
    n3 = round(rule["n3_frac"] * n_tensors)
    numel: dict[tuple[int, str], int] = {}
    fixed_bytes = 0.0
    for row in stage3_dump:
        m = TENSOR_RE.match(row["name"])
        if m:
            numel[(int(m.group(1)), m.group(2))] = int(row["n_elements"])
        else:
            fixed_bytes += float(row["n_bytes"])
    if set(numel) != {(l, t) for _, l, t in scored_order}:
        raise RuntimeError("stage-3 tensor set differs from the q4 map (layers/types)")
    n3_fixed = n3

    def predicted(n6: int) -> float:
        n4 = round(rule["n4_per_n6"] * n6)
        tiers = (["q6_k"] * n6 + ["q4_k"] * n4 + ["q3_k"] * n3_fixed
                 + ["q2_k"] * max(0, n_tensors - n6 - n4 - n3_fixed))
        return fixed_bytes + sum(numel[(l, t)] * TIER_BPW[tier] / 8.0
                                 for tier, (_, l, t) in zip(tiers, scored_order))

    ref_size = sum(float(r["n_bytes"]) for r in dump_rows(stage3_ref_key))
    n6 = _binary_search(predicted, ref_size, rule["size_tol"], n_tensors, n3_fixed)
    if n6 is None:
        raise RuntimeError("stage-3 transfer: no n6 lands within ±1%")
    n4 = round(rule["n4_per_n6"] * n6)
    tiers = (["q6_k"] * n6 + ["q4_k"] * n4 + ["q3_k"] * n3_fixed
             + ["q2_k"] * max(0, n_tensors - n6 - n4 - n3_fixed))
    assign = {_key(l, t): tier for (_, l, t), tier in zip(scored_order, tiers)}
    return {"n6": n6, "n4": n4, "n3": n3_fixed, "predicted_bytes": predicted(n6),
            "ref_bytes": ref_size, "assign": assign}


def quantize_args(assign: dict[str, str]) -> list[str]:
    """llama-quantize --tensor-type arguments: regex=TYPE per targeted tensor."""
    out = []
    for k in sorted(assign, key=lambda k: (int(k.split("|")[0]), k.split("|")[1])):
        l, t = k.split("|")
        out.append(f"^blk\\.{l}\\.{t}\\.weight$={assign[k].upper()}")
    return out


def random_recipe(assign: dict[str, str], seed: int) -> dict[str, str]:
    """Same tier multiset per tensor type, shuffled across layers → identical total size."""
    by_type: dict[str, list[str]] = {}
    for k, tier in assign.items():
        by_type.setdefault(k.split("|")[1], []).append(tier)
    rng = random.Random(seed)
    pools = {t: rng.sample(v, len(v)) for t, v in by_type.items()}
    return {k: pools[k.split("|")[1]].pop() for k in assign}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None, help="smoke override")
    ap.add_argument("--out", default="results/E5/recipes.json")
    args = ap.parse_args()
    logging_setup.setup("e5_recipe")

    e5 = config.load("experiments")["E5"]
    layers = args.layers if args.layers is not None else e5["layers"]
    ref = e5["calib_ref"]
    ref_key = f"{ref['model']}__{ref['quant']}__{ref['calib']}__{ref['seed']}"
    main_recipe = search_recipe(load_map(layers), dump_rows(ref_key), ref_key)
    scored_order = sorted(((float(r["kl"]) / float(r["bytes"]), int(r["layer"]),
                            str(r["type"])) for r in load_map(layers)), reverse=True)

    recipes = [
        {"label": "q4__mix__tr__s0", "model": ref["model"],
         "quantize_args": quantize_args(main_recipe["assign"])},
        {"label": "q4__rand0__tr__s0", "model": ref["model"],
         "quantize_args": quantize_args(random_recipe(main_recipe["assign"], 0))},
        {"label": "q4__rand1__tr__s0", "model": ref["model"],
         "quantize_args": quantize_args(random_recipe(main_recipe["assign"], 1))},
    ]
    stage3 = None
    if e5.get("stage3", True):  # spec §11 cut 4 turns the 8B transfer stage off
        stage3_ref = f"{e5['stage3_model']}__{ref['quant']}__{ref['calib']}__{ref['seed']}"
        stage3 = transfer_recipe(scored_order, dump_rows(stage3_ref), stage3_ref)
        recipes += [
            {"label": f"{e5['stage3_model']}__mix__tr__s0", "model": e5["stage3_model"],
             "quantize_args": quantize_args(stage3["assign"])},
            # stage 3 compares ours / uniform / one random recipe on the 8B (spec §E5 Aşama 3)
            {"label": f"{e5['stage3_model']}__rand0__tr__s0", "model": e5["stage3_model"],
             "quantize_args": quantize_args(random_recipe(stage3["assign"], 0))},
        ]
    data = {"main": main_recipe, "stage3": stage3, "recipes": recipes}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {out}: n6={main_recipe['n6']} n4={main_recipe['n4']} n3={main_recipe['n3']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())