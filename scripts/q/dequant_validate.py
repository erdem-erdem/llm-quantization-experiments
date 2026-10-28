"""Dequantization validation (plan §6.3 / spec §6.4).

1. Identity test (smoke, 0.6B): dequantize the BF16 GGUF, compare every tensor with the
   original HF safetensors — must be bit-exact (torch.equal). Proves the name/shape map.
2. Bit-exact test (8B Q4_K_M): tmp_deq tensors equal gguf.quants.dequantize(...) computed
   afresh after the same bf16 cast.
3. --cleanup <run_key>: removes models/tmp_deq/<run_key>/ (called by lm_eval_run.py after
   a successful eval).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import ids, logging_setup, paths  # noqa: E402


def load_state(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    state: dict[str, torch.Tensor] = {}
    idx = path / "model.safetensors.index.json"
    if idx.exists():
        files = sorted({path / v for v in
                        json.loads(idx.read_text(encoding="utf-8"))["weight_map"].values()})
    else:
        files = [next(path.glob("*.safetensors"))]
    for f in files:
        state.update(load_file(str(f)))
    return state


def identity_test(model: str) -> bool:
    """Dequantized BF16 GGUF vs original HF safetensors — bit-exact for every tensor."""
    from safetensors import safe_open

    hf_dir = paths.model_hf_path(model)
    deq_dir = paths.MODELS_TMP_DEQ / f"{model}__bf16__none__s0"
    hf_state: dict[str, torch.Tensor] = {}
    for f in sorted(hf_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                hf_state[k] = sf.get_tensor(k)
    deq_state = load_state(deq_dir)
    if set(hf_state) != set(deq_state):
        raise RuntimeError(f"tensor name sets differ: "
                           f"hf-only={sorted(set(hf_state) - set(deq_state))[:5]} "
                           f"deq-only={sorted(set(deq_state) - set(hf_state))[:5]}")
    bad = [k for k in hf_state
           if not torch.equal(hf_state[k].to(torch.bfloat16), deq_state[k].to(torch.bfloat16))]
    if bad:
        raise RuntimeError(f"identity test failed for {len(bad)} tensors: {bad[:5]}")
    print(f"identity test OK: {len(hf_state)} tensors bit-exact")
    return True


def bit_exact_test(run_key: str) -> bool:
    """tmp_deq tensors equal a fresh gguf.quants.dequantize after the same cast."""
    import gguf

    model, quant, calib, seed = ids.parse_run_key(run_key)
    r = gguf.GGUFReader(str(paths.gguf_path(model, quant, calib, seed)))
    deq_state = load_state(paths.tmp_deq_path(run_key))
    import numpy as np
    # rebuild the inverse the same way dequant.py did
    from scripts.q.dequant import build_inverse
    cfg = json.loads((paths.model_hf_path(model) / "config.json").read_text(encoding="utf-8"))
    from scripts.q.dequant import hf_tensor_names
    inverse = build_inverse({t.name for t in r.tensors},
                            hf_tensor_names(paths.model_hf_path(model)),
                            cfg["num_hidden_layers"])
    checked = 0
    for t in r.tensors:
        hf_name = inverse[t.name]
        arr = gguf.quants.dequantize(t.data, t.tensor_type)
        arr = arr.reshape(tuple(reversed(t.shape)))
        expect = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.bfloat16)
        got = deq_state[hf_name]
        if not torch.equal(expect, got):
            raise RuntimeError(f"bit-exact test failed for {hf_name}")
        checked += 1
    print(f"bit-exact test OK: {checked} tensors")
    return True


def cleanup(run_key: str) -> None:
    d = paths.tmp_deq_path(run_key)
    if d.exists():
        shutil.rmtree(d)
        print(f"removed {d}")
    else:
        print(f"nothing to remove at {d}")


def validate_outputs(args) -> bool:
    """record_run validator hook for a dequant job: bit-exact check only — the checkpoint is
    deleted by lm_eval_run.py once the eval that consumes it has finished."""
    bit_exact_test(args.key)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity", metavar="MODEL", help="bit-exact vs original HF (bf16 GGUF)")
    ap.add_argument("--bit-exact", metavar="RUN_KEY")
    ap.add_argument("--cleanup", metavar="RUN_KEY")
    args = ap.parse_args()
    logging_setup.setup("q_dequant_validate")
    if args.identity:
        identity_test(args.identity)
    if args.bit_exact:
        bit_exact_test(args.bit_exact)
    if args.cleanup:
        cleanup(args.cleanup)
    if not (args.identity or args.bit_exact or args.cleanup):
        ap.error("pass --identity, --bit-exact and/or --cleanup")
    return 0


if __name__ == "__main__":
    sys.exit(main())