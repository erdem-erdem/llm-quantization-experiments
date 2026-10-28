"""Tokenizer helpers shared by T0 scripts (packing, filters)."""
from __future__ import annotations

from transformers import AutoTokenizer

from . import ids


def load_qwen_tokenizer(model: str = "q8"):
    """Qwen3 tokenizer (all six models must share it; hashed and asserted at download)."""
    return AutoTokenizer.from_pretrained(ids.HF_REPO[model])


def alphabetic_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(c.isalpha() for c in s) / len(s)


def repeated_line_ratio(s: str) -> float:
    """Share of lines that appear more than once (filter: < 0.3)."""
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) < 2:
        return 0.0
    seen: dict[str, int] = {}
    for ln in lines:
        seen[ln] = seen.get(ln, 0) + 1
    dupes = sum(c - 1 for c in seen.values() if c > 1)
    return dupes / len(lines)


def passes_filter(text: str, min_chars: int = 200, min_alpha: float = 0.7,
                  max_repeat: float = 0.3) -> bool:
    if len(text) < min_chars:
        return False
    if alphabetic_ratio(text) < min_alpha:
        return False
    if repeated_line_ratio(text) >= max_repeat:
        return False
    return True


def pack_tracked(docs: list[str], tok, target_windows: int,
                 window: int = 2048) -> tuple[list[list[int]], int]:
    """Concatenate docs with EOS separators, cut into exact `window`-token slices.

    Returns (windows, n_docs_used): the first n_docs_used docs are the ones whose tokens went
    into the windows (the last one possibly only in part)."""
    eos = tok.eos_token_id
    stream: list[int] = []
    windows: list[list[int]] = []
    for n, d in enumerate(docs, start=1):
        ids_ = tok(d, add_special_tokens=False)["input_ids"]
        stream.extend(ids_)
        stream.append(eos)
        while len(stream) >= window:
            windows.append(stream[:window])
            stream = stream[window:]
            if len(windows) >= target_windows:
                return windows[:target_windows], n
    return windows, len(docs)


def pack(docs: list[str], tok, target_windows: int, window: int = 2048) -> list[list[int]]:
    """Windows only; see pack_tracked."""
    return pack_tracked(docs, tok, target_windows, window)[0]