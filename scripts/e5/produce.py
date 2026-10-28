"""E5 stage 2 — produce a recipe GGUF (plan §6.8): llama-quantize with per-tensor types.

Reads the recipe's --tensor-type args from results/E5/recipes.json, runs llama-quantize
(base type q3_k_m, same imatrix as the uniform E1 baseline), dumps tensor types, then
validates: every targeted tensor must carry its requested type, otherwise fail loudly
and delete the GGUF so a broken artifact is never consumed downstream.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, ids, logging_setup, paths  # noqa: E402
from scripts.common.shell import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="e.g. q4__mix__tr__s0 (key in recipes.json)")
    ap.add_argument("--llamacpp", default="llama.cpp")
    args = ap.parse_args()
    logging_setup.setup("e5_produce")

    model, quant, calib, seed = args.label.split("__")
    e5 = config.load("experiments")["E5"]
    ref = e5["calib_ref"]
    recipes = json.loads((paths.RESULTS / "E5" / "recipes.json").read_text(encoding="utf-8"))
    rec = next((r for r in recipes["recipes"] if r["label"] == args.label), None)
    if rec is None:
        raise RuntimeError(f"{args.label} not in recipes.json")
    if not rec["quantize_args"]:
        raise RuntimeError(f"{args.label}: empty quantize args")

    base = paths.MODELS_GGUF / f"{model}-bf16.gguf"
    imat = paths.MODELS_GGUF / f"{model}__imatrix__{ref['calib']}__{ref['seed']}.gguf"
    out = paths.MODELS_GGUF / f"{args.label}.gguf"
    cmd = [str(Path(args.llamacpp) / "llama-quantize"), "--imatrix", str(imat)]
    for ta in rec["quantize_args"]:
        cmd += ["--tensor-type", ta]
    cmd += [str(base), str(out), "Q3_K_M"]  # base type = uniform baseline treatment

    log = paths.LOGS / f"e5_produce_{args.label}.log"
    run(cmd, log_path=log, check=True)

    from scripts.q.gguf_quantize import _total_params
    from scripts.q.sizes import size_from_csv
    from scripts.q.tensor_dump import dump_csv
    dest = paths.tensor_types_csv(args.label)
    dump_csv(out, dest)
    size_from_csv(dest, _total_params(model))
    validate(args.label, rec["quantize_args"], out)
    print(f"wrote {out}")
    return 0


def validate(run_key: str, quantize_args: list[str], gguf_path: Path) -> None:
    import gguf
    requested: list[tuple[re.Pattern, str]] = []
    for arg in quantize_args:
        regex, ttype = arg.split("=", 1)
        requested.append((re.compile(regex), ttype))
    r = gguf.GGUFReader(str(gguf_path))
    bad = []
    for t in r.tensors:
        for rx, want in requested:
            if rx.match(t.name):
                got = gguf.GGUFQuantizationType(t.tensor_type).name
                if got != want:
                    bad.append(f"{t.name}: want {want}, got {got}")
    if bad:
        gguf_path.unlink(missing_ok=True)
        raise RuntimeError(f"{run_key}: {len(bad)} tensors have the wrong type; "
                           f"GGUF deleted. first: {bad[:5]}")
    print(f"validated {len(requested)} requested tensor types on {gguf_path.name}")


if __name__ == "__main__":
    sys.exit(main())