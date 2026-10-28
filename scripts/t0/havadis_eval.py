"""T0: Havadis eval corpus — stream, filter, accumulate >= 100K Qwen3 tokens (smoke: 8192).

Writes data/havadis_eval.txt (docs separated by \\n\\n). Segments (constants in
common/config.py via experiments.yaml): map segment = first 20,480 tokens (E5 stage 1);
KL segment = first 64K tokens, excluding the map segment for E5 recipe evals. The derived
text files for llama.cpp KL are data/havadis_kl64k.txt and data/havadis_kl64k_nomap.txt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, io, logging_setup, paths  # noqa: E402
from scripts.common.tokenize import alphabetic_ratio, load_qwen_tokenizer, passes_filter  # noqa: E402

DOC_SEED = 20240617


def kl_segment_offsets(tok, eval_text: str, map_tokens: int, kl_tokens: int) -> dict:
    """Token offsets of the map and KL segments within havadis_eval.txt (documented constants)."""
    ids = tok(eval_text, add_special_tokens=False)["input_ids"]
    return {"total_tokens": len(ids), "map_tokens": map_tokens,
            "kl_tokens": kl_tokens, "full": len(ids) >= kl_tokens}


def derived_kl_files(tok, eval_text: str, map_tokens: int, kl_tokens: int) -> list[Path]:
    """Write data/havadis_kl64k.txt (first 64K tokens, spec §7.2) and the -nomap variant (the
    same 64K minus the E5 map segment, i.e. tokens [map_tokens, 64K))."""
    ids = tok(eval_text, add_special_tokens=False)["input_ids"]
    full = ids[:kl_tokens]
    nomap = ids[map_tokens:kl_tokens]
    detok = tok.batch_decode
    p1 = paths.DATA / "havadis_kl64k.txt"
    p2 = paths.DATA / "havadis_kl64k_nomap.txt"
    io.atomic_write_text(detok([full])[0], p1)
    io.atomic_write_text(detok([nomap])[0], p2)
    return [p1, p2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="turkish-nlp-suite/Havadis",
                    help="Havadis HF dataset id")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    logging_setup.setup("t0_havadis_eval")

    from datasets import load_dataset

    if args.smoke:
        smoke = config.load("smoke").get("havadis", {})
        target = int(smoke.get("eval_tokens", 8192))
        map_tokens = int(smoke.get("map_tokens", 2048))
    else:
        target, map_tokens = 100_000, 20_480
    kl_tokens = 64_000

    tok = load_qwen_tokenizer()
    docs: list[str] = []
    total = 0
    stream = load_dataset(args.source, split="train", streaming=True).shuffle(
        seed=DOC_SEED, buffer_size=50_000)
    for row in stream:
        text = str(row.get("text") or row.get("content") or "")
        if not passes_filter(text):
            continue
        docs.append(text.strip())
        # ~4 chars/token pre-check avoids retokenizing the whole corpus each step
        if len(docs) >= 3 and sum(len(d) for d in docs) > target * 4:
            total = len(tok("\n\n".join(docs), add_special_tokens=False)["input_ids"])
            if total >= target:
                break
        if len(docs) % 1000 == 0:
            print(f"{len(docs)} docs")
    text = "\n\n".join(docs)
    total = len(tok(text, add_special_tokens=False)["input_ids"])
    if total < target:
        raise RuntimeError(f"Havadis corpus exhausted at {total} tokens < {target}")

    paths.DATA.mkdir(parents=True, exist_ok=True)
    dest = paths.DATA / "havadis_eval.txt"
    io.atomic_write_text(text, dest)
    info = kl_segment_offsets(tok, text, map_tokens, kl_tokens)
    files = derived_kl_files(tok, text, map_tokens, kl_tokens)
    print(f"wrote {dest}: {len(docs)} docs, {total} tokens; kl files: {[p.name for p in files]}")
    print(f"segment info: {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())