"""GGUF quantization (CPU, plan §6.3): llama-quantize with/without imatrix.

q8_0 and calib=none run without --imatrix; calibrated quants take the matching imatrix.
Output: models/gguf/<run_key>.gguf, then tensor_dump.py runs on it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, ids, logging_setup, paths  # noqa: E402
from scripts.common.shell import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-key", required=True)
    ap.add_argument("--llamacpp", default="llama.cpp", help="llama.cpp bin dir")
    ap.add_argument("--tensor-type", action="append", default=[],
                    help="regex=TYPE pass-through (E5 mixed recipes), repeatable")
    ap.add_argument("--output-tensor-type", default=None, help="E6-B lm_head quant")
    ap.add_argument("--token-embedding-type", default=None, help="E6-B tied-model fallback")
    args = ap.parse_args()
    logging_setup.setup("q_gguf_quantize")

    model, quant, calib, seed = ids.parse_run_key(args.run_key)
    qcfg = config.load("quants")["quants"][quant]
    assert qcfg["kind"] == "gguf", f"{quant} is not a GGUF quant"
    gguf_type = qcfg["gguf_type"].upper()  # llama-quantize expects Q4_K_M style

    base = paths.MODELS_GGUF / f"{model}-bf16.gguf"
    out = paths.gguf_path(model, quant, calib, seed)
    if out.exists():
        print(f"{out.name} already exists, skipping")
        return 0

    cmd = [str(Path(args.llamacpp) / "llama-quantize")]
    # calib=none (E2's imatrix-less Q3_K_M control) never gets an imatrix, whatever the type
    no_calib = quant in config.load("quants")["no_calib"] or calib == "none"
    if not no_calib:
        imat = paths.MODELS_GGUF / f"{model}__imatrix__{calib}__{seed}.gguf"
        if not imat.exists():
            raise FileNotFoundError(f"{imat}: run scripts/q/imatrix.py first")
        cmd += ["--imatrix", str(imat)]
    for tt in args.tensor_type:
        cmd += ["--tensor-type", tt]
    if args.output_tensor_type:
        cmd += ["--output-tensor-type", args.output_tensor_type]
    if args.token_embedding_type:
        cmd += ["--token-embedding-type", args.token_embedding_type]
    cmd += [str(base), str(out), gguf_type]

    log = paths.LOGS / f"gguf_quantize_{args.run_key}.log"
    run(cmd, log_path=log, check=True)
    print(f"wrote {out}")

    # tensor-type dump + sizes (plan §6.3: "then tensor_dump.py runs on it")
    from scripts.q.tensor_dump import dump_csv
    from scripts.q.sizes import size_from_csv
    dest = paths.tensor_types_csv(args.run_key)
    dump_csv(out, dest)
    total_params = _total_params(model)
    sizes = size_from_csv(dest, total_params)
    print(f"tensor dump: {dest}; sizes={sizes}")
    return 0


def _total_params(model: str) -> int | None:
    import csv as _csv
    p = paths.RESULTS / "model_cards.csv"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            if row.get("model") == model:
                return int(float(row["total_params"]))
    return None


if __name__ == "__main__":
    sys.exit(main())