"""GGUF → BF16 safetensors dequantization (plan §6.3, K-03 central riskiest component).

Streams tensor-by-tensor (never holds the full 32B dict in RAM), fails loudly on any
unmapped or unconsumed name (set difference both ways vs the HF index). The gguf-py name
map is HF→GGUF (`name_map.get_name(hf_name)`), so the inverse is built by mapping every
HF index name and reversing. GGUF dims are fastest-first → reshape with reversed shape.
Tied models: GGUF has token_embd.weight only → embed_tokens; lm_head.weight absent from
the HF index (assert). Untied: output.weight → lm_head.weight. Norm tensors are F32 in
GGUF → cast bf16. Output: models/tmp_deq/<run_key>/ with sharded safetensors (≤5 GB) +
index + copied config/generation_config/tokenizer files.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import ids, logging_setup, paths  # noqa: E402

SHARD_BYTES = 5_000_000_000


def hf_tensor_names(hf_dir: Path) -> set[str]:
    idx = hf_dir / "model.safetensors.index.json"
    if idx.exists():
        return set(json.loads(idx.read_text(encoding="utf-8"))["weight_map"].keys())
    from safetensors import safe_open
    with safe_open(next(hf_dir.glob("*.safetensors")), framework="pt", device="cpu") as f:
        return set(f.keys())


def _split_suffix(hf_name: str) -> tuple[str, str]:
    """Qwen3 tensors all end in .weight: (lookup base, suffix)."""
    if hf_name.endswith(".weight"):
        return hf_name[: -len(".weight")], ".weight"
    return hf_name, ""


def build_inverse(gguf_names: set[str], hf_names: set[str], n_blocks: int) -> dict[str, str]:
    """GGUF name → HF name. HF→GGUF via gguf-py map (suffix-less), then reversed; fails loudly."""
    import gguf

    name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3, n_blocks)
    inverse: dict[str, str] = {}
    hf_unmapped: list[str] = []
    for hf_name in hf_names:
        base, suffix = _split_suffix(hf_name)
        gguf_base = name_map.get_name(base)
        if gguf_base is None:
            hf_unmapped.append(hf_name)
        else:
            inverse[gguf_base + suffix] = hf_name
    if hf_unmapped:
        raise RuntimeError(f"HF names the QWEN3 map could not translate: {hf_unmapped[:8]}…")
    unmapped_gguf = sorted(gguf_names - set(inverse))
    missing_hf = sorted(hf_names - set(inverse.values()))
    if unmapped_gguf or missing_hf:
        raise RuntimeError(
            f"mapping incomplete: unmapped_gguf={unmapped_gguf[:8]} ({len(unmapped_gguf)}), "
            f"missing_hf={missing_hf[:8]} ({len(missing_hf)})")
    return inverse


def dequantize(gguf_file: Path, hf_dir: Path, out: Path) -> None:
    import gguf

    cfg = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    tied = bool(cfg["tie_word_embeddings"])
    hf_names = hf_tensor_names(hf_dir)
    if tied:
        assert "lm_head.weight" not in hf_names, \
            "tied model but lm_head.weight present in the HF index"

    r = gguf.GGUFReader(str(gguf_file))
    inverse = build_inverse({t.name for t in r.tensors}, hf_names, cfg["num_hidden_layers"])

    out.mkdir(parents=True, exist_ok=True)
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_idx = 0
    total_bytes = 0
    weight_map: dict[str, str] = {}

    def flush() -> None:
        nonlocal shard, shard_bytes, shard_idx
        if not shard:
            return
        from safetensors.torch import save_file
        name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        for k in shard:
            weight_map[k] = name
        save_file(shard, str(out / name))
        shard, shard_bytes, shard_idx = {}, 0, shard_idx + 1

    for t in r.tensors:
        hf_name = inverse[t.name]
        arr = gguf.quants.dequantize(t.data, t.tensor_type)  # float32 ndarray
        arr = arr.reshape(tuple(reversed(t.shape)))          # GGUF dims fastest-first
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.bfloat16)
        n_bytes = tensor.numel() * tensor.element_size()
        shard[hf_name] = tensor
        shard_bytes += n_bytes
        total_bytes += n_bytes
        if shard_bytes >= SHARD_BYTES:
            flush()
    flush()

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
                   indent=1) + "\n",
        encoding="utf-8", newline="\n")
    for fname in ("config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "vocab.json", "merges.txt"):
        src = hf_dir / fname
        if src.exists():
            shutil.copy2(src, out / fname)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-key", required=True)
    ap.add_argument("--gguf", default=None,
                    help="explicit GGUF (identity test: models/gguf/<m>-bf16.gguf, which "
                         "has no run-key-style name)")
    args = ap.parse_args()
    logging_setup.setup("q_dequant")
    model, quant, calib, seed = ids.parse_run_key(args.run_key)
    dequantize(Path(args.gguf) if args.gguf else paths.gguf_path(model, quant, calib, seed),
               paths.model_hf_path(model), paths.tmp_deq_path(args.run_key))
    print(f"wrote {paths.tmp_deq_path(args.run_key)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())