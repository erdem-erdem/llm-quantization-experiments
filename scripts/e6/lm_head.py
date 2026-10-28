"""E6-B — lm_head quantization (plan §6.8): q4_k_m baseline with the head re-quantized.

For q8 (untied): llama-quantize base Q4_K_M with --output-tensor-type {Q8_0,Q4_K}.
For q06 (tied): the output tensor is token_embd; try --output-tensor-type first and
verify from the dump that the targeted tensor actually changed type; if not, delete and
redo with --token-embedding-type, logging which flag took effect. Writes per-run sizes +
head share to results/E6/head__<run_key>.json (accuracy/KL measured by the E6 eval jobs).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, io, logging_setup, paths  # noqa: E402

TY_TO_GGUF = {"q8_0": "Q8_0", "q4_k": "Q4_K"}
HEAD_TENSORS = ("output.weight", "token_embd.weight")


def _read_csv(p: Path) -> list[dict]:
    import csv
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def head_tensor_rows(rows: list[dict]) -> list[dict]:
    """The lm_head tensor of a dump: output.weight when the model has one (untied), else
    token_embd.weight (tied: the embedding doubles as the head). Never both — on an untied
    model token_embd is the input embedding, not the head."""
    live = [r for r in rows if float(r["n_bytes"]) > 0]
    out = [r for r in live if r["name"] == "output.weight"]
    return out or [r for r in live if r["name"] == "token_embd.weight"]


def head_share(run_key: str) -> dict:
    """Bytes + share of the lm_head tensor from the tensor dump."""
    rows = _read_csv(paths.tensor_types_csv(run_key))
    head_rows = head_tensor_rows(rows)
    total = sum(float(r["n_bytes"]) for r in rows)
    head = sum(float(r["n_bytes"]) for r in head_rows)
    return {"head_mb": head / 1e6, "share_pct": 100.0 * head / total,
            "head_tensors": {r["name"]: r["tensor_type"] for r in head_rows}}


def _quantize(model: str, rk: str, out: Path, ty: str, llamacpp: str,
              imat: Path) -> str:
    """Quantize with the head flag; returns the flag that took effect (fails loudly)."""
    from scripts.common.shell import run
    base = paths.MODELS_GGUF / f"{model}-bf16.gguf"
    gguf_type = "Q4_K_M"
    baseline = {r["name"]: r["tensor_type"]
                for r in _read_csv(paths.tensor_types_csv(
                    f"{model}__q4_k_m__tr__s0"))}
    from scripts.q.tensor_dump import dump_csv
    for flag in ("--output-tensor-type", "--token-embedding-type"):
        cmd = [str(Path(llamacpp) / "llama-quantize")]
        if imat.exists():
            cmd += ["--imatrix", str(imat)]
        cmd += [flag, TY_TO_GGUF[ty], str(base), str(out), gguf_type]
        run(cmd, log_path=paths.LOGS / f"e6_head_{rk}.log", check=True)
        dump_csv(out, paths.tensor_types_csv(rk))
        share = head_share(rk)
        changed = any(baseline.get(n) != t for n, t in share["head_tensors"].items())
        if changed:
            if flag == "--token-embedding-type":
                print(f"{rk}: --output-tensor-type had no effect (tied model); "
                      f"used --token-embedding-type (logged)")
            return flag
        out.unlink()
    raise RuntimeError(f"{rk}: neither head flag changed the tensor type")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llamacpp", default="llama.cpp")
    ap.add_argument("--run-key", default=None,
                    help="only this variant, e.g. q8__hq8_0__q4_k_m__s0 (default: all)")
    args = ap.parse_args()
    logging_setup.setup("e6_head")

    e6 = config.load("experiments")["E6"]
    from scripts.q.gguf_quantize import _total_params
    from scripts.q.sizes import size_from_csv
    from scripts.q.tensor_dump import dump_csv

    extras: dict[str, dict] = {}
    for model in e6["head_models"]:
        base_quant = e6["head_base_quant"]
        imat = paths.MODELS_GGUF / f"{model}__imatrix__tr__s0.gguf"
        for ty in e6["head_types"]:
            rk = f"{model}__h{ty}__{base_quant}__s0"
            if args.run_key and args.run_key != rk:
                continue
            out = paths.gguf_path(model, f"h{ty}", base_quant, "s0")
            if out.exists():
                dump_csv(out, paths.tensor_types_csv(rk))
                flag = "cached"
            else:
                flag = _quantize(model, rk, out, ty, args.llamacpp, imat)
            sizes = size_from_csv(paths.tensor_types_csv(rk), _total_params(model))
            share = head_share(rk)
            extras[rk] = {"flag": flag, "head": share,
                          "size_bytes": sizes["size_bytes"],
                          "bpw": sizes["bpw"]}
            print(f"{rk}: flag={flag} head={share['head_mb']:.1f}MB "
                  f"({share['share_pct']:.2f}%)")
    if not extras:
        raise RuntimeError(f"--run-key {args.run_key}: no such head variant")
    # one file per variant so parallel head jobs never overwrite each other
    for rk, entry in extras.items():
        io.atomic_write_json(paths.RESULTS / "E6" / f"head__{rk}.json", entry)
    print(f"wrote results/E6/head__<run_key>.json for {sorted(extras)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())