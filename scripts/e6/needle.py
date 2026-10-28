"""E6-A needle protocol (plan §6.8): 3 ctx × 5 depths × 5 trials × 3 KV types = 225 gens.

Prompt = haystack with the needle's fact inserted at the sentence boundary nearest the
target depth + the needle question. Total tokens ≤ ctx−64 and < 32,768 (asserted via
/tokenize; 32K target ≈ 31,500). Trial j uses haystack j mod 5 and needle j — identical
across KV types (paired). Score: first integer in the reply equals the answer.
Precondition: F16@4K must be 5/5 at every depth before other KV types run.
Writes results/E6/needle.json {kvtype: {(ctx,depth): k}}.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, io, logging_setup, paths  # noqa: E402
from scripts.e6.server import LlamaServer
from scripts.t0.own_texts import NEEDLES


def _haystacks() -> list[str]:
    files = sorted((paths.OWN / "haystacks").glob("haystack_*.txt"))
    if len(files) < 5:
        raise FileNotFoundError(f"{paths.OWN / 'haystacks'}: run own_texts.py first")
    return [f.read_text(encoding="utf-8") for f in files[:5]]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p for p in parts if p.strip()]


def build_prompt(haystack: str, needle: dict, ctx: int, depth_pct: int,
                 srv: LlamaServer) -> str:
    """Fill the context with haystack sentences up to ctx−250 tokens (≈31,500 for the 32K
    setting, plan §6.8), insert the needle at the sentence boundary nearest depth_pct of the
    kept text, append the question. Token counts come from the server's /tokenize; asserts
    ≤ ctx−64 and < 32,768."""
    sents = _sentences(haystack)
    fact, question = needle["needle"], needle["question"]
    target = min(ctx - 250, 31_500)

    def assemble(k: int) -> str:
        keep = sents[:k]
        total = sum(len(x) + 1 for x in keep)
        c, idx = 0, len(keep)
        for i, x in enumerate(keep):
            c += len(x) + 1
            if c >= depth_pct / 100.0 * total:
                idx = i + 1
                break
        return ("\n".join(keep[:idx]) + f"\n{fact}\n" + "\n".join(keep[idx:])
                + f"\n\n{question}")

    lo, hi = 1, len(sents)  # largest k whose prompt fits the token target
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(srv.tokenize(assemble(mid))) <= target:
            lo = mid
        else:
            hi = mid - 1
    prompt = assemble(lo)
    n = len(srv.tokenize(prompt))
    if n > ctx - 64 or n >= 32_768 or (ctx == 32_768 and n < 31_000):
        raise RuntimeError(f"prompt token budget violated: {n} tokens (ctx={ctx})")
    return prompt


def first_int(text: str) -> int | None:
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llamacpp", default="llama.cpp")
    ap.add_argument("--contexts", default=None, help="smoke override, e.g. 4096")
    args = ap.parse_args()
    logging_setup.setup("e6_needle")

    e6 = config.load("experiments")["E6"]
    model = e6["kv_model"]
    ctxs = [int(args.contexts)] if args.contexts else list(e6["contexts"])
    depths = e6["depths"]
    n_trials = e6["n_trials"]
    gguf = paths.gguf_path(model, e6["kv_base_quant"], "none", "s0")
    haystacks = _haystacks()

    results: dict[str, dict] = {}
    for kvtype in e6["kv_types"]:
        results[kvtype] = {}
        for ctx in ctxs:
            with LlamaServer(gguf, ctx, cache_types=(kvtype, kvtype),
                             llamacpp=args.llamacpp) as srv:
                for depth in depths:
                    k = 0
                    for j in range(n_trials):
                        prompt = build_prompt(haystacks[j % 5], NEEDLES[j], ctx,
                                              depth, srv)
                        reply = srv.complete(prompt, max_tokens=16)
                        k += int(first_int(reply) == int(NEEDLES[j]["answer"]))
                    results[kvtype][f"{ctx},{depth}"] = k
                    print(f"needle {kvtype} ctx={ctx} depth={depth}: {k}/{n_trials}")
            # precondition (plan §6.8): F16 at 4K must be 5/5 at every depth
            if kvtype == "f16" and ctx == 4096:
                fails = [d for d in depths if results["f16"][f"4096,{d}"] < n_trials]
                if fails:
                    raise RuntimeError(
                        f"F16@4K precondition failed at depths {fails}: fix the prompt, "
                        f"freeze it, then rerun")
        io.atomic_write_json(paths.RESULTS / "E6" / "needle.json", results)
    print("wrote results/E6/needle.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())