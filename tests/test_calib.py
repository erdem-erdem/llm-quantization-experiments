"""T0 tests: pack/decode parity, filters, minhash self-overlap, needles, hook prompt."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import ids  # noqa: E402
from scripts.common.tokenize import load_qwen_tokenizer, pack, pack_tracked, passes_filter  # noqa: E402
from scripts.t0.calib_sets import doc_id, load_havadis_minhash, minhash_of  # noqa: E402
from scripts.t0.own_texts import DEMO_SENTENCES, HOOK_PROMPT, NEEDLES  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    return load_qwen_tokenizer()


def test_passes_filter_bounds():
    assert not passes_filter("kısa")
    long_tr = ("Bu cümle tamamen harflerden oluşuyor ve uzundur. " * 20).strip()
    assert passes_filter(long_tr)
    digits = "0123456789 " * 40
    assert not passes_filter(digits)  # alpha ratio ~0
    repeated = "aynı satır\n" * 50
    assert not passes_filter(repeated)


def test_pack_exact_windows_and_parity(tok):
    docs = ["kelime " * 3000, "başka bir belge burada. " * 2000, "son belge " * 1500]
    windows = pack(docs, tok, target_windows=4, window=2048)
    assert len(windows) == 4
    assert all(len(w) == 2048 for w in windows)
    # re-tokenizing the decoded text stays within ±0.5% of 4*2048 (BPE seam drift)
    decoded = "\n\n".join(tok.decode(w) for w in windows)
    re_ids = tok(decoded, add_special_tokens=False)["input_ids"]
    assert abs(len(re_ids) - 4 * 2048) <= 0.005 * 4 * 2048


def test_doc_id_stable():
    assert doc_id("abc") == doc_id("abc")
    assert doc_id("abc") != doc_id("abd")


def test_minhash_self_overlap(tmp_path, monkeypatch):
    fake = tmp_path / "havadis_eval.txt"
    fake.write_text("bu bir haber metnidir ve yeterince uzundur. " * 100, encoding="utf-8")
    monkeypatch.setattr("scripts.common.paths.DATA", tmp_path)
    lsh = load_havadis_minhash()
    assert overlaps_self(lsh, "bu bir haber metnidir ve yeterince uzundur. " * 100)
    assert not overlaps_self(lsh, "tamamen farklı başka bir metin bu. " * 100)


def overlaps_self(lsh, text: str) -> bool:
    return len(lsh.query(minhash_of(text))) > 0


def test_needles_wellformed():
    assert len(NEEDLES) == 5
    answers = [n["answer"] for n in NEEDLES]
    assert len(set(answers)) == 5
    assert all(len(a) == 4 and a.isdigit() for a in answers)
    for n in NEEDLES:
        assert n["needle"].rstrip(".").endswith(n["answer"])
        assert "Yalnızca sayıyı yaz." in n["question"]
        # no OTHER 4-digit number anywhere in the needle fields (scoring collisions)
        rest = (n["needle"].replace(n["answer"], "") + " " + n["question"])
        assert not re.search(r"\d{4}", rest), f"stray 4-digit number in {n}"


def test_demo_paragraphs_byte_exact():
    """Byte-compare against spec §5.4 when AGENT_TALIMATLARI.md is available."""
    spec = Path(__file__).resolve().parents[2] / "AGENT_TALIMATLARI.md"
    if not spec.exists():
        pytest.skip("spec not present (remote)")
    text = spec.read_text(encoding="utf-8")
    start = text.index("```json", text.index("demo_sentences"))
    end = text.index("```", start + 7)
    spec_json = json.loads(text[start + 7:end])
    assert json.loads(json.dumps(DEMO_SENTENCES, ensure_ascii=False)) == spec_json
    assert HOOK_PROMPT == "Bir arkadaşına, bilgisayarının zamanla neden yavaşladığını üç cümleyle anlat."


def test_models_registry():
    assert len(ids.MODELS) == 6
    assert ids.seed_int("s0") == 0 and ids.seed_int("s1") == 1
    assert ids.run_key("q8", "gptq", "tr", "s0") == "q8__gptq__tr__s0"

def test_pack_tracked_counts_only_docs_that_fed_the_windows(tok):
    docs = ["kelime " * 3000, "başka bir belge burada. " * 2000, "son belge " * 1500]
    windows, n_used = pack_tracked(docs, tok, target_windows=2, window=2048)
    assert len(windows) == 2
    assert n_used == 1  # the first doc alone is > 2 windows long; the other two were never read
    assert pack(docs, tok, target_windows=2, window=2048) == windows
