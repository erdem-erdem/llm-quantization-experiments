"""Demo — token visualization (plan §6.8): Qwen3 tokenizer offset_mapping, rendered with
Pillow (a font with Turkish glyphs: DejaVu Sans / Inter). Each paragraph pair renders the
TR line then the EN line, alternating pastel backgrounds per token, wrapped; per-line
token count at line end; exports a 1380 px wide PNG. Ratio stats (TR/EN tokens per
paragraph, overall, and over the 900 aligned Belebele passages) go to
results/demo/token_ratio.json. Paragraph text is never edited.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import io, logging_setup, paths  # noqa: E402
from scripts.t0.own_texts import DEMO_SENTENCES

WIDTH_PX = 1380
MARGIN = 24
LINE_H = 34
FONT_SIZE = 22
PASTELS = ["#DCE6F5", "#FCEBDD", "#E3F0E8", "#F5E3EF", "#F5F1DC"]

TR_KEYS = ["tr_1", "tr_2", "tr_3"]
EN_KEYS = ["en_1", "en_2", "en_3"]


def _font():
    from PIL import ImageFont
    # matplotlib ships DejaVu Sans (Turkish glyphs); a bare Ubuntu box may have no system fonts
    from matplotlib import font_manager
    for name in ("DejaVuSans.ttf", "Inter-Regular.ttf", font_manager.findfont("DejaVu Sans")):
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


def tokenize_offsets(text: str, tok) -> list[tuple[int, int]]:
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(zip(enc["input_ids"], enc["offset_mapping"]))


def render(paragraphs: list[dict], tok, out_png: Path) -> dict:
    from PIL import Image, ImageDraw
    font = _font()
    # layout: for each paragraph, TR line block then EN line block
    lines: list[tuple[str, list[tuple[str, str]], str]] = []  # (lang, [(word, color)], key)
    for i, p in enumerate(paragraphs):
        for key, text in (f"tr_{i + 1}", p["tr"]), (f"en_{i + 1}", p["en"]):
            toks = tokenize_offsets(text, tok)
            words: list[tuple[str, str]] = []
            for j, (_, (a, b)) in enumerate(toks):
                words.append((text[a:b], PASTELS[j % len(PASTELS)]))
            lines.append((key[:2], words, key))

    # wrap into rows of (token, color) with measured widths
    tmp = Image.new("RGB", (WIDTH_PX, 10))
    meas = ImageDraw.Draw(tmp)
    rows = []
    for lang, words, key in lines:
        cur: list[tuple[str, str]] = []
        w = MARGIN
        for tok_text, color in words:
            tw = meas.textlength(tok_text, font=font)
            if w + tw > WIDTH_PX - MARGIN and cur:
                rows.append((lang, cur, key))
                cur, w = [], MARGIN
            cur.append((tok_text, color))
            w += tw
        if cur:
            rows.append((lang, cur, key))
    h = MARGIN + len(rows) * (LINE_H + 6) + MARGIN
    img = Image.new("RGB", (WIDTH_PX, h), "white")
    d = ImageDraw.Draw(img)
    counts: dict[str, int] = {}
    y = MARGIN
    for lang, words, key in rows:
        x = MARGIN
        for tok_text, color in words:
            tw = d.textlength(tok_text, font=font)
            d.rectangle([x - 1, y, x + tw + 1, y + LINE_H - 4], fill=color)
            d.text((x, y + 4), tok_text, font=font, fill="black")
            x += tw
        counts[key] = counts.get(key, 0) + len(words)
        d.text((WIDTH_PX - MARGIN - 40, y + 4), str(counts[key]), font=font, fill="grey")
        y += LINE_H + 6
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png, "PNG")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    logging_setup.setup("demo_tokens")

    from scripts.common.tokenize import load_qwen_tokenizer
    tok = load_qwen_tokenizer()
    paragraphs = DEMO_SENTENCES  # list of {"tr", "en"} — never edited
    counts = render(paragraphs, tok, paths.RESULTS / "demo" / "token_viz.png")

    ratios = {}
    for i, _ in enumerate(paragraphs):
        tr, en = counts[f"tr_{i + 1}"], counts[f"en_{i + 1}"]
        ratios[f"p{i + 1}"] = {"tr": tr, "en": en, "ratio": tr / en}

    belebele = _belebele_ratios(tok)
    # the keys fig_demo.py reads (tests/synth.py fixes this shape)
    out = {"tr_per_en_paragraphs": [ratios[f"p{i + 1}"]["ratio"]
                                    for i in range(len(paragraphs))],
           "belebele_tr_per_en_total": belebele["total_ratio"],
           "belebele_mean_ratio": belebele["mean_ratio"],
           "chars_per_token_tr": belebele["chars_per_tok_tr"],
           "chars_per_token_en": belebele["chars_per_tok_en"]}
    io.atomic_write_json(paths.RESULTS / "demo" / "token_ratio.json", out)
    print(f"wrote token_viz.png + token_ratio.json "
          f"(belebele mean {out['belebele_mean_ratio']:.3f})")
    return 0


def _belebele_ratios(tok) -> dict:
    """900 aligned Belebele passages: total ratio, mean per-passage ratio, chars/token."""
    import numpy as np
    from datasets import load_dataset
    tr = load_dataset("facebook/belebele", "tur_Latn", split="test")
    en = load_dataset("facebook/belebele", "eng_Latn", split="test")
    # alignment: same doc ids in the same order (dataset layouts are parallel)
    assert len(tr) == len(en) == 900
    rt, re_ = [], []
    chars_t = chars_e = toks_t = toks_e = 0
    for a, b in zip(tr, en):
        ta, tb = a["flores_passage"].replace("\n", " "), b["flores_passage"].replace("\n", " ")
        na = len(tok(ta, add_special_tokens=False)["input_ids"])
        nb = len(tok(tb, add_special_tokens=False)["input_ids"])
        if nb == 0:
            continue
        rt.append(na / nb)
        chars_t += len(ta)
        chars_e += len(tb)
        toks_t += na
        toks_e += nb
    return {"n": len(rt), "total_ratio": toks_t / toks_e,
            "mean_ratio": float(np.mean(rt)),
            "chars_per_tok_tr": chars_t / toks_t, "chars_per_tok_en": chars_e / toks_e}


if __name__ == "__main__":
    sys.exit(main())