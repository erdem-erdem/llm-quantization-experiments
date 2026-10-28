"""KL divergence via llama.cpp (plan §6.6; GGUF quants only, K-02).

Once per model: llama-perplexity on the BF16 GGUF writes refs/<m>.kld. The KL run takes its
tokens from that file (the -f text is ignored), so the E5 map-excluded text (--nomap) needs its
own reference, refs/<m>__nomap.kld.
Per quant: same command with --kl-divergence-base. Parses the tool's output (llama.cpp logs
to stderr) for Mean KLD, 99.0% KLD, Same top p, Mean PPL(Q); missing fields fail loudly.
Raw log kept in results/<exp>/raw/.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, ids, io, logging_setup, paths  # noqa: E402
from scripts.common.shell import run  # noqa: E402

# llama.cpp prints e.g. "Mean    KLD:   0.000112 ±  0.000002", "99.0%   KLD:   0.000727",
# "Same top p: 99.659 ± 0.015 %", "Mean PPL(Q)                   :   9.123456 ±  0.05"
PATTERNS = {
    "mean_kld": r"Mean\s+KLD\s*:\s*([0-9.eE+-]+)",
    "p99_kld": r"99\.0%\s+KLD\s*:\s*([0-9.eE+-]+)",
    "same_top_p": r"Same top p\s*:\s*([0-9.eE+-]+)",
    "ppl": r"Mean PPL\(Q\)\s*:\s*([0-9.eE+-]+)",
}


def parse_kl(stdout: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, stdout)
        if m is None:
            raise RuntimeError(f"KL parse failed: field '{key}' not found (format drift?)")
        out[key] = float(m.group(1))
    return out


def kl_cmd(model_gguf: Path, kl_text: Path, base_kld: Path | None,
           exe: str = "llama-perplexity") -> list[str]:
    cmd = [exe, "-m", str(model_gguf), "-f", str(kl_text),
           "-c", "2048", "-ngl", "99"]
    if base_kld is not None:
        cmd += ["--kl-divergence-base", str(base_kld), "--kl-divergence"]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-key", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--llamacpp", default="llama.cpp", help="llama.cpp build dir")
    ap.add_argument("--nomap", action="store_true",
                    help="E5 mode: KL on the map-excluded segment")
    ap.add_argument("--reference-only", action="store_true",
                    help="score the BF16 GGUF and write refs/<m>.kld")
    args = ap.parse_args()
    logging_setup.setup("kl_llama")
    model, quant, calib, seed = ids.parse_run_key(args.run_key)

    bin_dir = Path(args.llamacpp)
    kl_text = paths.DATA / ("havadis_kl64k_nomap.txt" if args.nomap else "havadis_kl64k.txt")
    model_gguf = paths.gguf_path(model, quant, calib, seed) if quant != "bf16" \
        else paths.MODELS_GGUF / f"{model}-bf16.gguf"

    exe = str(bin_dir / "llama-perplexity")
    base = paths.REFS / (f"{model}__nomap.kld" if args.nomap else f"{model}.kld")
    if args.reference_only:
        paths.REFS.mkdir(parents=True, exist_ok=True)
        # reference run: compute and store the base distribution (no --kl-divergence)
        cmd = [exe, "-m", str(model_gguf), "-f", str(kl_text),
               "-c", "2048", "-ngl", "99", "--kl-divergence-base", str(base)]
    else:
        if not base.exists():
            raise FileNotFoundError(f"{base}: run --reference-only for {model} first")
        cmd = kl_cmd(model_gguf, kl_text, base, exe)

    log = paths.raw_dir(args.exp, args.run_key) / f"kl_{args.run_key}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    t0 = datetime.now()
    proc = run(cmd, log_path=log, check=False, cwd=paths.ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-perplexity failed rc={proc.returncode}")
    if args.reference_only:
        if not base.exists():
            raise RuntimeError(f"llama-perplexity wrote no {base}")
        vals: dict[str, float] = {}  # the base run reports no KL statistics
    else:
        vals = parse_kl(proc.stdout + "\n" + proc.stderr)

    from scripts.common.run_meta import build_meta
    from scripts.common.schemas import RunRow
    meta = build_meta(exp=args.exp, job="kl_llama", run_key=args.run_key, cmd=cmd,
                      t0=t0, t1=datetime.now(), returncode=0, n_gpus=1,
                      seed=seed, extras={"metrics": vals, "kl_text": str(kl_text)})
    io.atomic_write_text(json.dumps(meta, indent=1) + "\n",
                         paths.run_meta_path(args.exp, args.run_key, "kl_llama"))
    for metric, v in vals.items():
        row = RunRow(exp=args.exp, model=model, quant=quant, calib=calib, seed=seed,
                     task="kl_havadis", metric=metric, value=v, n=1, engine="llama.cpp",
                     run_id=f"{args.run_key}__kl_havadis").model_dump()
        io.jsonl_append(paths.runs_jsonl(args.exp), row)
    print(f"KL {args.run_key}: {vals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())