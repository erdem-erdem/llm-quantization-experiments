"""Derived sizes from a tensor_types CSV (plan §6.3).

size_bytes = sum of tensor n_bytes; bpw = 8*weight_bytes/total_params; bpw_body excludes
token_embd.weight and output.weight. Tied models have no output.weight — lm_head bytes are
the embedding bytes by construction (documented in the bpw figures).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def size_from_csv(csv_path: Path, total_params: int | None = None) -> dict:
    df = pd.read_csv(csv_path)
    weight_bytes = int(df["n_bytes"].sum())
    emb = int(df.loc[df["name"] == "token_embd.weight", "n_bytes"].sum())
    out_w = int(df.loc[df["name"] == "output.weight", "n_bytes"].sum())
    body_bytes = weight_bytes - emb - out_w
    out = {"size_bytes": weight_bytes, "weight_bytes": weight_bytes,
           "body_bytes": body_bytes, "emb_bytes": emb, "lm_head_bytes": out_w}
    if total_params:
        out["bpw"] = 8.0 * weight_bytes / total_params
        out["bpw_body"] = 8.0 * body_bytes / total_params
    return out


def _total_params(model: str) -> int | None:
    from scripts.common import paths
    p = paths.RESULTS / "model_cards.csv"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("model") == model:
                return int(float(row["total_params"]))
    return None


def run_sizes(run_key: str) -> dict:
    """size_bytes / bpw (/ bpw_body) for any run key: GGUF-derived keys from their tensor dump,
    llm-compressor checkpoints and BF16 from their safetensors bytes."""
    from scripts.common import ids, paths
    model, quant, _, _ = ids.parse_run_key(run_key)
    total_params = _total_params(model)
    dump = paths.tensor_types_csv(run_key)
    if dump.exists():
        return size_from_csv(dump, total_params)
    d = paths.ct_path(run_key) if quant in ids.LLMC_QUANTS else paths.model_hf_path(model)
    nbytes = sum(f.stat().st_size for f in d.glob("*.safetensors"))
    if not nbytes:
        raise FileNotFoundError(f"{d}: no safetensors to size {run_key}")
    out = {"size_bytes": nbytes}
    if total_params:
        out["bpw"] = 8.0 * nbytes / total_params
    return out
