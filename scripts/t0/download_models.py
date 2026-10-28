"""T0: download the six Qwen3 models and write results/model_cards.csv.

Card fields: layers, hidden, q_heads, kv_heads, head_dim, vocab, tie, total_params,
embed_params. Asserts: tie is True exactly for {q06, q17, q4}; identical tokenizer.json
SHA256 across models; head_dim read from config (128, explicit in Qwen3 — never hidden/heads).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from filelock import FileLock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import ids, io, logging_setup, paths  # noqa: E402

ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"]
TOK_FILE = "tokenizer.json"


def _card(model: str) -> dict:
    p = paths.model_hf_path(model)
    cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
    n_params = 0
    for st in sorted(p.glob("*.safetensors")):
        from safetensors import safe_open
        with safe_open(st, framework="pt", device="cpu") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                n = 1
                for d in sl.get_shape():
                    n *= d
                n_params += n
    card = {
        "model": model,
        "repo": ids.HF_REPO[model],
        "layers": cfg["num_hidden_layers"],
        "hidden": cfg["hidden_size"],
        "q_heads": cfg["num_attention_heads"],
        "kv_heads": cfg["num_key_value_heads"],
        "head_dim": cfg["head_dim"],  # explicit field in Qwen3; do NOT derive hidden/heads
        "vocab": cfg["vocab_size"],
        "tie": bool(cfg["tie_word_embeddings"]),
        "total_params": n_params,
        "embed_params": cfg["vocab_size"] * cfg["hidden_size"],
    }
    return card


def download(models: list[str]) -> None:
    from huggingface_hub import snapshot_download

    for m in models:
        dest = paths.model_hf_path(m)
        if dest.exists() and (dest / "config.json").exists() and any(dest.glob("*.safetensors")):
            print(f"{m}: already present, skipping")
            continue
        print(f"{m}: downloading {ids.HF_REPO[m]} …")
        snapshot_download(ids.HF_REPO[m], local_dir=dest, allow_patterns=ALLOW_PATTERNS)


def merge_cards(out_csv: Path, cards: list[dict], models: list[str]) -> pd.DataFrame:
    """Merge this call's cards into model_cards.csv under a lock, written atomically.

    The per-model dl: jobs run concurrently and each call only knows its own models."""
    with FileLock(str(out_csv) + ".lock", timeout=120):
        if out_csv.exists():
            old_cards = pd.read_csv(out_csv)
            cards = [c for c in old_cards.to_dict("records") if c["model"] not in models] + cards
        cards = sorted(cards, key=lambda c: ids.MODELS.index(c["model"]))
        df = pd.DataFrame(cards)
        io.atomic_write_text(df.to_csv(index=False, lineterminator="\n"), out_csv)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ids.MODELS))
    args = ap.parse_args()
    logging_setup.setup("t0_download_models")
    models = args.models.split(",")

    download(models)

    # tokenizer.json SHA256 must be identical across all six
    shas: dict[str, str] = {}
    for m in models:
        tok_p = paths.model_hf_path(m) / TOK_FILE
        shas[m] = hashlib.sha256(tok_p.read_bytes()).hexdigest()
    if len(set(shas.values())) != 1:
        raise RuntimeError(f"tokenizer.json hashes differ across models: {shas}")

    cards = [_card(m) for m in models]
    tied = {c["model"] for c in cards if c["tie"]}
    if tied != ids.TIED & set(models):  # scoped to what was downloaded (smoke fetches q06 only)
        raise RuntimeError(f"tied embeddings {sorted(tied)} != expected {sorted(ids.TIED & set(models))}")

    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = paths.RESULTS / "model_cards.csv"
    df = merge_cards(out_csv, cards, models)
    print(f"wrote {out_csv} ({len(df)} rows); tokenizer sha ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())