"""T0: own texts — demo paragraphs (byte-exact, spec §5.4), needles, hook prompt, haystacks.

Outputs: data/own/demo_sentences.json, data/own/needles.jsonl, data/own/hook_prompt.txt,
data/own/haystacks/haystack_{0..4}.txt (>= 31K Qwen3 tokens each, sentence-boundary cut).
Haystack text has every 4-digit number stripped so only needle answers match scoring.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import io, logging_setup, paths  # noqa: E402
from scripts.common.tokenize import load_qwen_tokenizer  # noqa: E402

# spec §5.4 — verbatim; a test byte-compares this file's JSON against the spec extract.
DEMO_SENTENCES = [
    {"tr": "Cehalet korkuya, korku nefrete, nefret ise şiddete yol açar. Bu denklem, aklını kullanmayı reddeden toplumların kaçınılmaz kaderidir. Hakikati aramak yerine dogmaların karanlığına sığınanlar, en nihayetinde kendi inşa ettikleri o zifiri karanlıkta birbirlerini yok etmeye mahkûmdurlar.",
     "en": "Ignorance leads to fear, fear leads to hate, and hate leads to violence. This equation is the inevitable fate of societies that refuse to use their intellect. Those who take refuge in the darkness of dogmas instead of seeking the truth are ultimately doomed to destroy each other in the pitch darkness they have built themselves."},
    {"tr": "Sen taş, kaya veya kurumuş bir toprak değilsin; içinde sonsuz bir potansiyel barındıran eşsiz bir ruhsun. Kendi dar kalıplarına ve korkularına hapsolmaktan vazgeç, zira senin içindeki o ilahi kıvılcım uyandığında, karanlıkları aydınlatacak ve seni gökyüzünün ötesine taşıyacak güce sahiptir.",
     "en": "You are not a stone, a rock, or dried earth; you are a unique soul harboring infinite potential within. Cease being trapped in your own narrow molds and fears, for when that divine spark within you awakens, it possesses the power to illuminate the darkness and carry you beyond the skies."},
    {"tr": "Kuru bir ağaca su vermek onu nasıl yeşertmezse, kibre ve kine bulanmış bir kalbe de binlerce kitap okutmak o kalbi aydınlatmaz. İnsanın asıl tahsili, kâinatın yaradılışındaki o gizli sevgi sırrını çözmek ve her nefese şefkatle muamele etmeyi öğrenmektir.",
     "en": "Just as watering a dry tree will not make it green, making a heart stained with arrogance and malice read thousands of books will not illuminate that heart. The true education of a human is to decipher that hidden secret of love in the creation of the universe and to learn to treat every breath with compassion."},
]

HOOK_PROMPT = "Bir arkadaşına, bilgisayarının zamanla neden yavaşladığını üç cümleyle anlat."

# five needles, distinct names/objects, numeric 4-digit answers
NEEDLES = [
    {"needle": "Selin'in bisiklet kilidinin şifresi 4721.",
     "question": "Selin'in bisiklet kilidinin şifresi kaçtır? Yalnızca sayıyı yaz.", "answer": "4721"},
    {"needle": "Kaan'ın evdeki kasasının şifresi 6053.",
     "question": "Kaan'ın evdeki kasasının şifresi kaçtır? Yalnızca sayıyı yaz.", "answer": "6053"},
    {"needle": "Deniz'in valizindeki kilit kodu 9318.",
     "question": "Deniz'in valizindeki kilit kodu kaçtır? Yalnızca sayıyı yaz.", "answer": "9318"},
    {"needle": "Elif'in kilitli çekmecesinin kodu 2764.",
     "question": "Elif'in kilitli çekmecesinin kodu kaçtır? Yalnızca sayıyı yaz.", "answer": "2764"},
    {"needle": "Bora'nın garaj kapısının şifresi 5087.",
     "question": "Bora'nın garaj kapısının şifresi kaçtır? Yalnızca sayıyı yaz.", "answer": "5087"},
]

MIN_HAYSTACK_TOKENS = 31_000
FOUR_DIGITS = re.compile(r"\d{4}")


def write_static() -> list[Path]:
    paths.OWN.mkdir(parents=True, exist_ok=True)
    out = []
    p = paths.OWN / "demo_sentences.json"
    io.atomic_write_text(json.dumps(DEMO_SENTENCES, ensure_ascii=False, indent=1) + "\n", p)
    out.append(p)
    nl = paths.OWN / "needles.jsonl"
    io.atomic_write_text("".join(json.dumps(n, ensure_ascii=False) + "\n" for n in NEEDLES), nl)
    out.append(nl)
    hp = paths.OWN / "hook_prompt.txt"
    io.atomic_write_text(HOOK_PROMPT + "\n", hp)
    out.append(hp)
    return out


def build_haystacks(tok, target_tokens: int = MIN_HAYSTACK_TOKENS,
                    count: int = 5) -> list[Path]:
    """Concatenate BellaTurca AkademikDerlem docs, cut on sentence boundary, strip 4-digit numbers."""
    from datasets import load_dataset

    hdir = paths.OWN / "haystacks"
    hdir.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(count):
        dest = hdir / f"haystack_{i}.txt"
        stream = load_dataset("turkish-nlp-suite/BellaTurca", "AkademikDerlem",
                              split="train", streaming=True).shuffle(seed=100 + i, buffer_size=20_000)
        text = ""
        n_tokens = 0
        for row in stream:
            doc = str(row.get("text") or row.get("document") or "")
            if len(doc) < 200:
                continue
            text += doc.strip() + "\n\n"
            if len(text) > 200_000:  # tokenize at chunk boundaries to bound memory
                n_tokens = len(tok(text, add_special_tokens=False)["input_ids"])
                if n_tokens > target_tokens:
                    break
        text = text.strip()
        if text:
            # cut on the last sentence boundary before the target
            cut = text.rfind(". ")
            if cut > 0:
                text = text[: cut + 1]
        text = FOUR_DIGITS.sub("", text)
        ids_ = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids_) < target_tokens:
            raise RuntimeError(f"haystack_{i}: only {len(ids_)} tokens < {target_tokens}")
        io.atomic_write_text(text, dest)
        out.append(dest)
        print(f"haystack_{i}: {len(ids_)} tokens")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-haystacks", action="store_true", help="static files only")
    args = ap.parse_args()
    logging_setup.setup("t0_own_texts")

    out = write_static()
    print(f"static files: {[p.name for p in out]}")
    if not args.skip_haystacks:
        tok = load_qwen_tokenizer()
        out += build_haystacks(tok)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())