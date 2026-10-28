"""T0: calibration sets (K-01) — {tr: BellaTurca OzenliDerlem, en: allenai/c4-en} × {s0, s1}.

Per (lang, seed): shuffle-stream a doc pool (seed = seed_id*1000 + 17), filter
(passes_filter), drop MinHash overlap with Havadis (tr only), pack 128 windows of exactly
2048 tokens (EOS-joined, tail discarded). Outputs per lang/seed:
  data/calib/<lang>_s<k>.jsonl  {"input_ids": [...2048...]} per window (GPTQ consumer)
  data/calib/<lang>_s<k>.txt    decode of the same windows joined by \\n\\n (imatrix consumer)
  data/calib/<lang>_s<k>.docs   content-hash doc-id list
plus mix_s<k>.jsonl/.txt = 64 tr + 64 en windows interleaved in fixed order.
Doc ids are content hashes, so s0/s1 disjointness is real: the s1 pool excludes any doc
whose hash already appears in s0 (resample continues the stream until the pool refills).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import ids, io, logging_setup, paths  # noqa: E402
from scripts.common.tokenize import load_qwen_tokenizer, pack_tracked, passes_filter  # noqa: E402

N_WINDOWS = 128
POOL_SIZE = 5000
POOL_SCAN_CAP = 50_000
MINHASH_PERM = 128

SOURCES = {"tr": ("turkish-nlp-suite/BellaTurca", "OzenliDerlem"),
           "en": ("allenai/c4", "en")}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def doc_id(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()[:16]


def load_havadis_minhash():
    """MinHashLSH over the Havadis eval docs (>=200 chars), 5-word shingles."""
    from datasketch import MinHash, MinHashLSH

    text = (paths.DATA / "havadis_eval.txt").read_text(encoding="utf-8")
    docs = [d for d in text.split("\n\n") if len(d) >= 200]
    lsh = MinHashLSH(threshold=0.8, num_perm=MINHASH_PERM)
    for j, d in enumerate(docs):
        lsh.insert(f"h{j}", minhash_of(d))
    return lsh


def minhash_of(doc: str):
    from datasketch import MinHash

    words = doc.split()
    m = MinHash(num_perm=MINHASH_PERM)
    for i in range(0, max(0, len(words) - 5)):
        m.update(" ".join(words[i:i + 5]).encode("utf-8"))
    return m


def overlaps_havadis(text: str, lsh) -> bool:
    return len(lsh.query(minhash_of(text))) > 0


def build_lang_seed(lang: str, seed: str, tok, lsh, exclude_ids: set[str]) -> dict:
    src_repo, src_cfg = SOURCES[lang]
    from datasets import load_dataset

    stream = load_dataset(src_repo, src_cfg, split="train", streaming=True).shuffle(
        buffer_size=50_000, seed=ids.seed_id(seed) * 1000 + 17)
    docs: list[tuple[str, str]] = []
    scanned = dropped_overlap = dropped_filter = 0
    for row in stream:
        scanned += 1
        if scanned > POOL_SCAN_CAP:
            break
        text = str(row.get("text", ""))
        if not passes_filter(text):
            dropped_filter += 1
            continue
        did = doc_id(text)
        if did in exclude_ids:
            continue  # keep s1 disjoint from s0 (resample = keep scanning)
        if lang == "tr" and overlaps_havadis(text, lsh):
            dropped_overlap += 1
            continue
        docs.append((did, text.strip()))
        if len(docs) >= POOL_SIZE:
            break

    windows, n_used = pack_tracked([t for _, t in docs], tok, target_windows=N_WINDOWS)
    if len(windows) < N_WINDOWS:
        raise RuntimeError(f"{lang}/{seed}: only {len(windows)} windows < {N_WINDOWS}")
    windows = windows[:N_WINDOWS]
    docs = docs[:n_used]  # .docs lists the docs actually packed, not the whole 5000-doc pool

    paths.CALIB.mkdir(parents=True, exist_ok=True)
    jl = paths.CALIB / f"{lang}_{seed}.jsonl"
    txt = paths.CALIB / f"{lang}_{seed}.txt"
    docids = paths.CALIB / f"{lang}_{seed}.docs"
    io.atomic_write_text("".join(json.dumps({"input_ids": w}) + "\n" for w in windows), jl)
    io.atomic_write_text("\n\n".join(tok.decode(w) for w in windows), txt)
    io.atomic_write_text("\n".join(k for k, _ in docs) + "\n", docids)
    return {"lang": lang, "seed": seed, "docs": len(docs), "windows": len(windows),
            "scanned": scanned, "dropped_filter": dropped_filter,
            "minhash_dropped": dropped_overlap,
            "sha256_jsonl": _sha(jl), "sha256_txt": _sha(txt)}


def build_mix(seed: str, tok) -> dict:
    """64 tr + 64 en windows, interleaved in fixed order (tr0, en0, tr1, en1, ...)."""
    tr = [json.loads(l)["input_ids"] for l in
          (paths.CALIB / f"tr_{seed}.jsonl").read_text(encoding="utf-8").splitlines()]
    en = [json.loads(l)["input_ids"] for l in
          (paths.CALIB / f"en_{seed}.jsonl").read_text(encoding="utf-8").splitlines()]
    half_t, half_e = tr[:64], en[:64]
    mixed = [w for pair in zip(half_t, half_e) for w in pair]
    jl = paths.CALIB / f"mix_{seed}.jsonl"
    txt = paths.CALIB / f"mix_{seed}.txt"
    io.atomic_write_text("".join(json.dumps({"input_ids": w}) + "\n" for w in mixed), jl)
    io.atomic_write_text("\n\n".join(tok.decode(w) for w in mixed), txt)
    return {"lang": "mix", "seed": seed, "windows": len(mixed),
            "sha256_jsonl": _sha(jl), "sha256_txt": _sha(txt)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="tr,en")
    ap.add_argument("--seeds", default="s0,s1")
    args = ap.parse_args()
    logging_setup.setup("t0_calib_sets")

    tok = load_qwen_tokenizer()
    lsh = load_havadis_minhash()

    entries = []
    for lang in args.langs.split(","):
        s0_ids: set[str] = set()
        if "s0" in args.seeds and lang in args.langs.split(","):
            s0_path = paths.CALIB / f"{lang}_s0.docs"
            if s0_path.exists():
                s0_ids = set(s0_path.read_text(encoding="utf-8").split())
        for seed in args.seeds.split(","):
            # s0 is built from scratch (a stale s0.docs would shift it on every rerun); every
            # other seed excludes the s0 docs, including those from the s0 built just above
            e = build_lang_seed(lang, seed, tok, lsh,
                                exclude_ids=set() if seed == "s0" else s0_ids)
            if seed == "s0":
                s0_ids = set((paths.CALIB / f"{lang}_s0.docs").read_text(encoding="utf-8").split())
            entries.append(e)
            print(f"{lang}/{seed}: {e['docs']} docs, {e['windows']} windows, "
                  f"minhash dropped {e['minhash_dropped']}, filter dropped {e['dropped_filter']}")

    # s0/s1 disjointness assertion (real ids: content hashes)
    for lang in ("tr", "en"):
        p0 = paths.CALIB / f"{lang}_s0.docs"
        p1 = paths.CALIB / f"{lang}_s1.docs"
        if p0.exists() and p1.exists():
            s0 = set(p0.read_text(encoding="utf-8").split())
            s1 = set(p1.read_text(encoding="utf-8").split())
            assert not (s0 & s1), f"{lang}: s0/s1 doc overlap: {sorted(s0 & s1)[:3]}"

    for seed in args.seeds.split(","):
        entries.append(build_mix(seed, tok))

    man = paths.DATA / "manifest_calib.json"
    io.atomic_write_text(json.dumps(
        {"entries": entries,
         "hashes": {p.name: _sha(p) for p in sorted(paths.CALIB.glob("*")) if p.is_file()}},
        ensure_ascii=False, indent=1) + "\n", man)
    print(f"wrote {man}")
    return 0


if __name__ == "__main__":
    sys.exit(main())