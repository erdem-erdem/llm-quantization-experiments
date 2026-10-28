"""E5 stage 1 — sensitivity map (plan §6.8): per-tensor 2-bit RTN patch → KL over 252 cells.

Loads the BF16 model, caches reference log-probs on the map segment (first `map_tokens`
Havadis tokens = 10 windows of 2048), then for each (layer, type): replace the tensor with
2-bit RTN, forward all windows, KL on the second half (llama.cpp rule), restore.
RTN definition (documented in NOTES): group size 128 along the input dim, symmetric,
scale = max|w|/2, q = clamp(round(w/scale), −2, 1), ŵ = q·scale.
Output results/E5/map.csv (layer,type,kl,bytes) — bytes from the uniform Q3_K_M dump so
the stage-2 score KL/bytes is per baseline byte.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, logging_setup, paths  # noqa: E402

TYPE_TO_MODULE = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
}


def rtn2(w: torch.Tensor, group: int = 128) -> torch.Tensor:
    """2-bit RTN along the input dim (last), group size 128 (plan §6.8 definition)."""
    out_dim, in_dim = w.shape
    pad = (-in_dim) % group
    wp = torch.nn.functional.pad(w, (0, pad)) if pad else w
    wp = wp.reshape(out_dim, -1, group)
    scale = wp.abs().amax(dim=-1, keepdim=True) / 2.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(wp / scale), -2, 1)
    return (q * scale).reshape(out_dim, -1)[:, :in_dim].to(w.dtype)


def _map_tokens() -> list[int]:
    from scripts.common.tokenize import load_qwen_tokenizer
    tok = load_qwen_tokenizer()
    text = (paths.DATA / "havadis_eval.txt").read_text(encoding="utf-8")
    ids = tok(text, add_special_tokens=False)["input_ids"][:config.load("experiments")
                                                          ["E5"]["map_tokens"]]
    return ids


def baseline_bytes(layers: int) -> dict[tuple[int, str], int]:
    """bytes per (layer, type) from the uniform Q3_K_M tensor dump (GGUF names)."""
    ref = config.load("experiments")["E5"]["calib_ref"]
    csv = paths.tensor_types_csv(
        f"{ref['model']}__{ref['quant']}__{ref['calib']}__{ref['seed']}")
    pat = re.compile(r"blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$")
    out: dict[tuple[int, str], int] = {}
    for row in _read_csv(csv):
        m = pat.match(row["name"])
        if m:
            out[(int(m.group(1)), m.group(2))] = int(row["n_bytes"])
    for l in range(layers):
        for t in TYPE_TO_MODULE:
            if (l, t) not in out:
                raise RuntimeError(f"no baseline bytes for (l={l}, {t}) in {csv}")
    return out


def _read_csv(p: Path) -> list[dict]:
    import csv as _csv
    with open(p, encoding="utf-8") as f:
        return list(_csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="default: experiments.yaml E5.map_model")
    ap.add_argument("--layers", type=int, default=None, help="smoke override")
    ap.add_argument("--out", default="results/E5/map.csv")
    args = ap.parse_args()
    logging_setup.setup("e5_map_sweep")

    e5 = config.load("experiments")["E5"]
    model_name = args.model or e5["map_model"]
    layers = args.layers if args.layers is not None else e5["layers"]

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        paths.model_hf_path(model_name), torch_dtype=torch.bfloat16, device_map="cuda:0")

    tok_ids = _map_tokens()
    windows = np.array(tok_ids[:(len(tok_ids) // 2048) * 2048], dtype=np.int64)
    windows = windows.reshape(-1, 2048)
    from scripts.kl.kl_hf import RefCache, score_against
    cache = RefCache(model, windows)  # fp32 log-probs on GPU

    modules = {}
    for l in range(layers):
        for t, mod in TYPE_TO_MODULE.items():
            modules[(l, t)] = model.get_submodule(f"model.layers.{l}.{mod}")

    results = []
    for (l, t), mod in modules.items():
        orig = mod.weight.data.clone()
        try:
            mod.weight.data = rtn2(orig.float()).to(orig.dtype)
            vals = score_against(cache, model)
        finally:
            mod.weight.data = orig
        results.append({"layer": l, "type": t, "kl": vals["mean_kld"],
                        "same_top_p": vals["same_top_p"]})
        print(f"l={l:2d} {t:11s} KL={vals['mean_kld']:.5g}")

    nbytes = baseline_bytes(layers)
    for r in results:
        r["bytes"] = nbytes[(r["layer"], r["type"])]
    import pandas as pd
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote {out} ({len(results)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())