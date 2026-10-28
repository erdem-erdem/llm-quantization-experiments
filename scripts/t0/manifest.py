"""T0: data/manifest.json — per dataset: HF id, config/split, revision hash, rows,
Qwen3 tokens, sha256 of the materialized file. Also writes data/LICENSES.md with the
Havadis and BellaTurca license texts.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import io, logging_setup, paths  # noqa: E402
from scripts.common.tokenize import load_qwen_tokenizer  # noqa: E402

DATA_SOURCES = [
    {"name": "havadis_eval", "hf": "turkish-nlp-suite/Havadis", "config": None, "split": "train",
     "file": "havadis_eval.txt"},
    {"name": "calib_tr", "hf": "turkish-nlp-suite/BellaTurca", "config": "OzenliDerlem", "split": "train",
     "file": "calib/tr_s0.jsonl"},
    {"name": "calib_en", "hf": "allenai/c4", "config": "en", "split": "train",
     "file": "calib/en_s0.jsonl"},
    {"name": "gpqa_tr", "hf": "ytu-ce-cosmos/gpqa-extended_tr", "config": None, "split": "train",
     "file": "tasks/gpqa_tr_perm.json"},
]

LICENSES = {
    "Havadis": "turkish-nlp-suite/Havadis",
    "BellaTurca": "turkish-nlp-suite/BellaTurca",
}


def main() -> int:
    logging_setup.setup("t0_manifest")
        from huggingface_hub import HfApi

    api = HfApi()
    tok = load_qwen_tokenizer()
    entries = []
    for src in DATA_SOURCES:
        if src["file"].startswith("tasks/"):
            p = paths.TASKS / src["file"].removeprefix("tasks/")
        else:
            p = paths.DATA / src["file"]
        info = api.dataset_info(src["hf"])
        rev = info.sha
        rows = None
        try:
            # Use HF API metadata to get row count without materializing the corpus
            if hasattr(info, "splits") and info.splits:
                split_name = src["split"]
                for split in info.splits:
                    if split.name == split_name:
                        rows = split.num_examples
                        break
        except Exception as exc:  # noqa: BLE001 — manifest must not block on streaming repos
            print(f"WARNING: could not count rows for {src['hf']}: {exc}")
        n_tokens = None
        if p.exists() and src["name"].startswith("havadis"):
            n_tokens = len(tok(p.read_text(encoding="utf-8"),
                               add_special_tokens=False)["input_ids"])
        entries.append({**src, "revision": rev, "rows": rows, "qwen3_tokens": n_tokens,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None,
                        "file_exists": p.exists()})

    man = paths.DATA / "manifest.json"
    io.atomic_write_text(json.dumps({"sources": entries}, ensure_ascii=False, indent=1) + "\n", man)
    print(f"wrote {man}")

    lic = paths.DATA / "LICENSES.md"
    lines = ["# Lisanslar", ""]
    for title, repo in LICENSES.items():
        info = api.dataset_info(repo)
        card = getattr(info, "card_data", None)
        lic_text = (getattr(card, "license", None) or "license metadata missing — see dataset page")
        lines += [f"## {title}", f"- Kaynak: https://huggingface.co/datasets/{repo}",
                  f"- Lisans (metadata): {lic_text}", ""]
    io.atomic_write_text("\n".join(lines), lic)
    print(f"wrote {lic}")
    return 0


if __name__ == "__main__":
    sys.exit(main())