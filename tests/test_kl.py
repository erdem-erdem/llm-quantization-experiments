"""KL parsing + windowing tests (CPU, no models)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.kl.kl_llama import parse_kl  # noqa: E402

SAMPLE_LLAMA_STDOUT = """
| Perplexity: Fizz + Buzz | #Token: 64 | ...
Mean PPL(Q)   : 8.1234
Mean KLD      : 0.04561
99.0% KLD     : 0.12345
Same top p    : 87.50%
"""


def test_parse_kl_all_fields():
    v = parse_kl(SAMPLE_LLAMA_STDOUT)
    assert abs(v["mean_kld"] - 0.04561) < 1e-9
    assert abs(v["p99_kld"] - 0.12345) < 1e-9
    assert abs(v["same_top_p"] - 87.50) < 1e-9
    assert abs(v["ppl"] - 8.1234) < 1e-9


# verbatim LOG formats of tools/perplexity/perplexity.cpp (checked against ggml-org/llama.cpp master)
REAL_LLAMA_OUTPUT = """
====== Perplexity statistics ======
Mean PPL(Q)                   :   9.123456 ±   0.051234
Mean PPL(base)                :   9.100000 ±   0.050000

====== KL divergence statistics ======
Mean    KLD:   0.000112 ±  0.000002
Maximum KLD:   0.015631
99.9%   KLD:   0.002139
99.0%   KLD:   0.000727
Median  KLD:   0.000040

====== Token probability statistics ======
Same top p: 99.659 ±  0.015 %
"""


def test_parse_kl_real_llama_cpp_format():
    v = parse_kl(REAL_LLAMA_OUTPUT)
    assert v == {"mean_kld": 0.000112, "p99_kld": 0.000727, "same_top_p": 99.659,
                 "ppl": 9.123456}


def test_parse_kl_missing_field_fails_loudly():
    import pytest
    with pytest.raises(RuntimeError, match="same_top_p"):
        parse_kl(SAMPLE_LLAMA_STDOUT.replace("Same top p    : 87.50%\n", ""))


def test_windows_non_overlapping():
    from scripts.kl.kl_hf import _windows
    ids = list(range(10))
    w = _windows(ids, 4)
    assert w.shape == (2, 4)
    assert w.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]  # tail dropped


def test_kl_math_identical_models():
    """KL(p||p) = 0 and top-1 agreement = 100% when ref == test (synthetic logits)."""
    torch = None
    try:
        import torch
    except ImportError:
        return
    V = 100
    rng = np.random.default_rng(0)
    logits = torch.tensor(rng.normal(size=(1, 16, V)), dtype=torch.float32)
    lp = torch.log_softmax(logits, dim=-1)[0]
    kl = (lp.exp() * (lp - lp)).sum(-1)
    assert float(kl.abs().max()) == 0.0
    top = (lp.argmax(-1) == lp.argmax(-1)).float().mean()
    assert float(top) == 1.0

def test_hf_kl_p99_is_per_token_not_per_window_mean():
    import types

    import numpy as np
    import torch

    from scripts.kl.kl_hf import score_against

    torch.manual_seed(0)
    ctx, score_from, vocab, n_win = 16, 8, 11, 3
    ids = torch.zeros(n_win, ctx, dtype=torch.long)
    ref_logits = torch.randn(n_win, ctx, vocab)
    test_logits = ref_logits + 0.5 * torch.randn(n_win, ctx, vocab)
    lp_ref = torch.log_softmax(ref_logits, -1)[:, score_from:ctx - 1]
    cache = types.SimpleNamespace(ids=ids, logprobs=lp_ref, device="cpu")
    it = iter(test_logits)

    class Model:
        device = "cpu"

        def __call__(self, input_ids):
            return types.SimpleNamespace(logits=next(it)[None])

    out = score_against(cache, Model(), ctx=ctx, score_from=score_from)
    lp_test = torch.log_softmax(test_logits, -1)[:, score_from:ctx - 1]
    kl = (lp_ref.exp() * (lp_ref - lp_test)).sum(-1).numpy().ravel()
    assert abs(out["mean_kld"] - kl.mean()) < 1e-6
    assert abs(out["p99_kld"] - np.percentile(kl, 99)) < 1e-6
    assert out["scored_positions"] == ctx - 1 - score_from
