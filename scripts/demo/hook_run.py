"""Demo — hook blind comparison (plan §6.8): Qwen3-32B BF16 (vLLM) vs GGUF quants
(q4_k_m, q3_k_m, q2_k via llama-server), identical settings (greedy, 256 tokens,
enable_thinking=False), prompt from data/own/hook_prompt.txt.

Writes results/demo/hook_blind.md — labels A–D assigned by a seeded shuffle, no hint of
origin — plus hook_key.json (label → system), token counts, and a word-level diff of each
system against BF16 (difflib.SequenceMatcher on whitespace tokens). The author picks the
pair; the key is revealed only afterwards (plan Q6).
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, io, logging_setup, paths  # noqa: E402
from scripts.e6.server import LlamaServer

GGUF_QUANTS = ["q4_k_m", "q3_k_m", "q2_k"]
LABELS = ["A", "B", "C", "D"]


def vllm_reply(prompt: str, max_tokens: int) -> str:
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(paths.model_hf_path("q32")), dtype="bfloat16",
              max_model_len=2048, gpu_memory_utilization=0.85, enforce_eager=True)
    tok = llm.get_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    out = llm.generate(rendered, SamplingParams(temperature=0, max_tokens=max_tokens))
    return out[0].outputs[0].text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args()
    logging_setup.setup("demo_hook")

    demo = config.load("experiments")["demo"]
    max_tokens = args.max_tokens or demo["max_tokens"]
    prompt = (paths.OWN / "hook_prompt.txt").read_text(encoding="utf-8").strip()

    replies: dict[str, str] = {}
    for quant in GGUF_QUANTS:
        with LlamaServer(paths.gguf_path("q32", quant, "tr", "s0"), 4096) as srv:
            replies[quant] = srv.complete(prompt, max_tokens=max_tokens)
    # vLLM last: its engine keeps 85% of the GPU, which would starve the llama-server loads
    replies["bf16"] = vllm_reply(prompt, max_tokens)

    systems = ["bf16"] + GGUF_QUANTS
    rng = random.Random(20250901)
    labels = systems[:]
    rng.shuffle(labels)  # labels[i] is the system shown under LABELS[i]
    key = {lab: sys_name for lab, sys_name in zip(LABELS, labels)}

    # The blind file shows ONLY the answers: a per-label diff-from-BF16 line would reveal which
    # label is BF16 (0.0%) and order the rest by bit width. Counts and diffs go to the key.
    blocks = [f"## {lab}\n\n{replies[key[lab]]}\n" for lab in LABELS]
    blind = ("# Kör karşılaştırma\n\n"
             "Aşağıdaki dört cevap aynı soruya, aynı ayarlarla üretilmiştir "
             "(greedy, 256 token). Hangi ikisi en çok ayrışıyor?\n\n" + "\n".join(blocks))
    ddir = paths.RESULTS / "demo"
    ddir.mkdir(parents=True, exist_ok=True)
    io.atomic_write_text(blind, ddir / "hook_blind.md")
    from scripts.common.tokenize import load_qwen_tokenizer
    tok = load_qwen_tokenizer()
    base = replies["bf16"].split()
    io.atomic_write_json(ddir / "hook_key.json", {
        "key": key,
        "token_counts": {s: len(tok(replies[s], add_special_tokens=False)["input_ids"])
                         for s in systems},
        "word_diff_vs_bf16_pct": {
            s: round(100 * (1 - difflib.SequenceMatcher(None, base, replies[s].split()).ratio()), 1)
            for s in systems},
        "max_tokens": max_tokens})
    print(f"wrote {ddir / 'hook_blind.md'} (key separate in hook_key.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())