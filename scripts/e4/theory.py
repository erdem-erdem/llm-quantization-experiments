"""E4 theory line (plan §6.8): tok/s_max = 3.35e12 / bytes_decode.

bytes_decode = weight_bytes − embedding-table bytes (q8 and q32 are both untied). Weight bytes come
from the tensor_types CSV for GGUF quants, from the safetensors directory size for vLLM
cells. Measured decode tok/s comes from results/E4/runs.jsonl. Writes results/E4/extras.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, ids, io, logging_setup, paths  # noqa: E402

THEORY_BW = 3.35e12  # bytes/s, H100 SXM6 theoretical HBM bandwidth (plan §6.8)


def _size_bytes(run_key: str) -> dict:
    """{weight_bytes, embed_bytes} for a run key, from tensor dump or safetensors dir.

    embed_bytes is the embedding TABLE only (decode reads one row of it); the output/lm_head
    matrix is read in full every token and stays in bytes_decode. Tied models subtract nothing
    (spec §8 E4)."""
    model, quant, calib, seed = ids.parse_run_key(run_key)
    tied = model in ids.TIED
    if quant in ids.GGUF_QUANTS:
        csv = paths.tensor_types_csv(run_key)
        if not csv.exists():
            raise FileNotFoundError(f"{csv}: run tensor_dump first")
        import csv as _csv
        weight = embed = 0.0
        with open(csv, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                nb = float(row["n_bytes"])
                weight += nb
                if row["name"] == "token_embd.weight" and not tied:
                    embed += nb
        return {"weight_bytes": weight, "embed_bytes": embed, "source": "tensor_dump"}
    d = paths.ct_path(run_key) if quant in ids.LLMC_QUANTS else paths.model_hf_path(model)
    weight = sum(f.stat().st_size for f in d.glob("*.safetensors"))
    # embed_tokens is not a Linear layer: BF16 in every checkpoint (2 bytes/param)
    embed = 0.0 if tied else 2.0 * (_card_field(model, "embed_params") or 0)
    return {"weight_bytes": float(weight), "embed_bytes": embed, "source": "dir_size"}


ENGINE_LABEL = {"llama.cpp": "llama.cpp", "vllm": "vLLM"}  # fig_e4 marker keys


def analyze(model: str) -> dict:
    rows = io.read_jsonl(paths.runs_jsonl("E4"))
    per_quant: dict[str, dict] = {}
    for r in rows:
        if r.get("exp") != "E4" or r.get("model") != model or r.get("task") != "speed":
            continue
        # one cell per (quant, engine): BF16 is measured by both llama.cpp and vLLM
        cell = f"{r['model']}__{r['quant']}__{r['engine']}"
        entry = per_quant.setdefault(cell, {
            "model": r["model"], "quant": r["quant"],
            "engine": ENGINE_LABEL.get(r["engine"], r["engine"])})
        entry[r["metric"]] = r["value"]
    for entry in per_quant.values():
        if "decode_tok_s" not in entry:
            continue
        quant = entry["quant"]
        no_calib = set(config.load("quants")["no_calib"])
        calib = "none" if quant in no_calib else config.load("experiments")["E1"]["calib"]
        sizes = _size_bytes(ids.run_key(model, quant, calib, "s0"))
        bytes_decode = sizes["weight_bytes"] - sizes["embed_bytes"]
        theory = THEORY_BW / bytes_decode
        entry.update(bytes_decode_gb=bytes_decode / 1e9,
                     theory_tok_s=theory,
                     theory_pct=round(100.0 * entry["decode_tok_s"] / theory, 1),
                     size_source=sizes["source"])
    return {"per_quant": {k: v for k, v in per_quant.items() if "theory_pct" in v},
            "total_params": _card_field(model, "total_params")}


def _card_field(model: str, field: str) -> int | None:
    csv = paths.RESULTS / "model_cards.csv"
    if not csv.exists():
        return None
    import csv as _csv
    with open(csv, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            if row.get("model") == model:
                return int(float(row[field]))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    logging_setup.setup("e4_theory")
    out = paths.RESULTS / "E4" / f"extras_{args.model}.json"
    data = analyze(args.model)
    io.atomic_write_json(out, data)
    print(f"wrote {out} ({len(data['per_quant'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())