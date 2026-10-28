"""Tensor-type dump + derived sizes (plan §6.3).

results/tensor_types/<run_key>.csv per tensor: name, shape, tensor_type, n_elements,
n_bytes (+norm flag). q/sizes.py helpers: size_bytes = sum(n_bytes); bpw =
8*weight_bytes/total_params; bpw_body excludes token_embd/output.weight.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import io, logging_setup, paths  # noqa: E402
from scripts.q.sizes import size_from_csv  # noqa: E402

NORM_F32 = {"token_embd.weight", "output.weight"}  # excluded from bpw_body weight bytes


def _prod(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def dump_csv(gguf_path: Path, dest: Path) -> None:
    import gguf

    r = gguf.GGUFReader(str(gguf_path))
    rows = []
    for t in r.tensors:
        rows.append({
            "name": t.name,
            "shape": "x".join(str(d) for d in t.shape),
            "tensor_type": t.tensor_type.name,
            "n_elements": _prod(t.shape),
            "n_bytes": int(t.n_bytes),
            "is_norm_f32": "norm" in t.name or t.name in NORM_F32,
        })
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False, encoding="utf-8", lineterminator="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="path to .gguf")
    ap.add_argument("--run-key", required=True)
    ap.add_argument("--total-params", type=int, default=None, help="from model_cards.csv")
    args = ap.parse_args()
    logging_setup.setup("q_tensor_dump")
    dest = paths.tensor_types_csv(args.run_key)
    dump_csv(Path(args.gguf), dest)
    sizes = size_from_csv(dest, args.total_params)
    io.atomic_write_text(pd.Series(sizes).to_json(), dest.with_suffix(".sizes.json"))
    print(f"wrote {dest}; sizes={sizes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())