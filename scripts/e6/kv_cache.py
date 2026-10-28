"""E6-A — KV cache (plan §6.8): {f16,q8_0,q4_0} × ctx {4096,16384,32768} on q8 Q8_0 GGUF.

Per cell: start llama-server with the KV type, parse the llama_kv_cache size line, send a
full-context prompt while polling nvidia-smi for peak memory, and measure decode speed at
filled context via llama-bench -d (depth ctx−512, decode 512). Formula check:
2·36·8·128·ctx·2 bytes for f16. Writes results/E6/runs.jsonl rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, ids, io, logging_setup, paths  # noqa: E402
from scripts.common.schemas import RunRow  # noqa: E402
from scripts.e6.server import LlamaServer, kv_cache_mib_from_log, peak_gpu_mib  # noqa: E402

KV_BYTES_PER_TOKEN_F16 = 2 * 36 * 8 * 128 * 2  # layers × kv_heads × head_dim × f16 bytes


def _emit(run_key: str, task: str, metric: str, value: float, ctx: int,
          kvtype: str, run_id: str, reason: str | None = None) -> None:
    model, quant, calib, seed = ids.parse_run_key(run_key)
    io.jsonl_append(paths.runs_jsonl("E6"), RunRow(
        exp="E6", model=model, quant=quant, calib=calib, seed=seed,
        task=task, metric=metric, value=value, n=1, engine="llama.cpp",
        run_id=run_id, reason=reason).model_dump())


def _emit_kv(rows: dict, ctx: int, kvtype: str, gguf: Path, llamacpp: str) -> None:
    e6 = config.load("experiments")["E6"]
    run_id = f"kv__{kvtype}__{ctx}"
    with LlamaServer(gguf, ctx, cache_types=(kvtype, kvtype),
                     llamacpp=llamacpp) as srv:
        mib = kv_cache_mib_from_log(srv.logfile)
        if mib is None:
            raise RuntimeError(f"no llama_kv_cache size line in {srv.logfile}")

        def full_prompt() -> None:
            per_word = len(srv.tokenize("kelime " * 100)) / 100.0
            filler = "kelime " * int((ctx - 200) / per_word)
            n = len(srv.tokenize(filler))
            if not (ctx - 256) <= n <= ctx - 64:
                raise RuntimeError(f"full-context prompt is {n} tokens, expected ~{ctx - 256}")
            srv.complete(filler, max_tokens=8)

        peak = peak_gpu_mib(full_prompt)
    formula_mib = KV_BYTES_PER_TOKEN_F16 * ctx / (1024 * 1024)
    if kvtype == "q8_0":
        formula_mib *= 1.0625 / 2
    elif kvtype == "q4_0":
        formula_mib *= 0.5625 / 2
    _emit("q8__q8_0__none__s0", "kv_cache", "kv_mib", mib, ctx, kvtype, run_id)
    _emit("q8__q8_0__none__s0", "kv_cache", "peak_gb", peak / 1024.0, ctx, kvtype, run_id)
    rows.setdefault(kvtype, {})[ctx] = {"kv_mib": mib, "peak_gb": peak / 1024.0,
                                        "formula_mib": formula_mib}


def decode_speed(gguf: Path, ctx: int, kvtype: str, llamacpp: str) -> float:
    """Decode tok/s at a filled context: llama-bench -p 0 -n 512 -d ctx−512 (llama-bench takes
    cache types in lowercase; -d prefills the depth before the timed decode)."""
    cmd = [str(Path(llamacpp) / "llama-bench"), "-m", str(gguf), "-ngl", "99", "-fa", "1",
           "-ctk", kvtype, "-ctv", kvtype, "-p", "0", "-n", "512", "-d", str(ctx - 512),
           "-r", "1", "-o", "json"]
    import subprocess
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600)
    tg = [float(c["avg_ts"]) for c in json.loads(out.stdout)
          if c["n_gen"] > 0 and c["n_prompt"] == 0]
    if not tg:
        raise RuntimeError(f"llama-bench JSON lacks a tg test: {out.stdout[:300]}")
    return tg[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llamacpp", default="llama.cpp")
    args = ap.parse_args()
    logging_setup.setup("e6_kv")

    e6 = config.load("experiments")["E6"]
    model = e6["kv_model"]
    rk = ids.run_key(model, e6["kv_base_quant"], "none", "s0")
    gguf = paths.gguf_path(model, e6["kv_base_quant"], "none", "s0")
    if not gguf.exists():
        raise FileNotFoundError(f"{gguf} (E1 production first)")

    rows: dict = {}
    for kvtype in e6["kv_types"]:
        for ctx in e6["contexts"]:
            _emit_kv(rows, ctx, kvtype, gguf, args.llamacpp)
            tok_s = decode_speed(gguf, ctx, kvtype, args.llamacpp)
            _emit(rk, "kv_cache", "decode_tok_s", tok_s, ctx, kvtype,
                  f"kv__{kvtype}__{ctx}__bench")
            rows[kvtype][ctx]["decode_tok_s"] = tok_s
            print(f"kv {kvtype} ctx={ctx}: {rows[kvtype][ctx]}")

    out = paths.RESULTS / "E6" / "kv.json"
    io.atomic_write_json(out, rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())