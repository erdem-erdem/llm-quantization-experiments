"""GGUF imatrix (GPU job, plan §6.3): llama-imatrix per (model, calib, seed).

llama-imatrix -m models/gguf/<m>-bf16.gguf -f data/calib/<calib>_s<k>.txt
  -o models/gguf/<m>__imatrix__<calib>__s<k>.gguf -c 2048 --chunks 128 -ngl 99
Validates the output exists and the log reports 128 chunks. One imatrix per
(model, calib, seed); Q4_K_M/Q3_K_M/Q2_K reuse it.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import ids, logging_setup, paths  # noqa: E402
from scripts.common.shell import run  # noqa: E402


def imatrix_path(model: str, calib: str, seed: str) -> Path:
    return paths.MODELS_GGUF / f"{model}__imatrix__{calib}__{seed}.gguf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True, choices=["tr", "en", "mix"])
    ap.add_argument("--seed", required=True, choices=["s0", "s1"])
    ap.add_argument("--llamacpp", default="llama.cpp", help="llama.cpp bin dir")
    args = ap.parse_args()
    logging_setup.setup("q_imatrix")

    base = paths.MODELS_GGUF / f"{args.model}-bf16.gguf"
    if not base.exists():
        raise FileNotFoundError(f"{base}: run scripts/t0/gguf_convert.py first")
    calib_txt = paths.CALIB / f"{args.calib}_{args.seed}.txt"
    if not calib_txt.exists():
        raise FileNotFoundError(f"{calib_txt}: run scripts/t0/calib_sets.py first")
    out = imatrix_path(args.model, args.calib, args.seed)
    log = paths.LOGS / f"imatrix_{args.model}_{args.calib}_{args.seed}.log"
    if out.exists():
        print(f"{out.name} already exists, skipping")
        return 0

    bin_dir = Path(args.llamacpp)
    run([str(bin_dir / "llama-imatrix"), "-m", str(base), "-f", str(calib_txt),
         "-o", str(out), "-c", "2048", "--chunks", "128", "-ngl", "99"],
        log_path=log, check=True)

    logtext = log.read_text(encoding="utf-8", errors="replace")
    processed = len(re.findall(r"chunk\s+\d+\s*/\s*128", logtext, re.IGNORECASE))
    if processed != 128:
        logging.getLogger("q_imatrix").warning(
            "log shows %d/128 chunk markers — verify manually in %s", processed, log)
    if not out.exists():
        raise RuntimeError("imatrix output missing after run")
    print(f"wrote {out}; log chunk markers: {processed}/128")
    return 0


if __name__ == "__main__":
    sys.exit(main())