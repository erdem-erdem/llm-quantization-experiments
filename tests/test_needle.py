"""needle prompt builder: token-budgeted fill and needle depth (fake tokenizer: 1 token/word)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.e6.needle import build_prompt  # noqa: E402
from scripts.t0.own_texts import NEEDLES  # noqa: E402


class FakeServer:
    def tokenize(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def _haystack(n_sentences: int) -> str:
    return " ".join(f"Bu cümle {i} numaralı akademik metin parçasıdır ve oldukça uzundur."
                    for i in range(n_sentences))


@pytest.mark.parametrize("ctx", [4096, 16384])
@pytest.mark.parametrize("depth", [5, 50, 95])
def test_prompt_fills_context_and_places_needle(ctx, depth):
    hay = _haystack(20000)
    prompt = build_prompt(hay, NEEDLES[0], ctx, depth, FakeServer())
    n = len(prompt.split())
    assert ctx - 250 - 20 <= n <= ctx - 250           # filled to the token target, not a few chars
    pos = prompt.index(NEEDLES[0]["needle"]) / len(prompt)
    assert abs(pos - depth / 100) < 0.05


def test_32k_stays_below_limit():
    prompt = build_prompt(_haystack(20000), NEEDLES[1], 32768, 50, FakeServer())
    assert 31_000 <= len(prompt.split()) <= 31_500 < 32_768
