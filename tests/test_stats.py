"""Layer A tests for stats: bootstrap CI coverage, flips, Wilson, McNemar, Holm."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.stats import matched  # noqa: E402
from scripts.stats.tests import holm, mcnemar_exact  # noqa: E402


def _vectors(n: int = 1000, delta_frac: float = 0.10, seed: int = 3):
    """ref correct on first delta_frac*n extra items; quant loses those, ties elsewhere."""
    rng = np.random.default_rng(seed)
    ref = (rng.random(n) < 0.7).astype(np.int8)
    q = ref.copy()
    idx = np.where(ref == 1)[0][: int(delta_frac * n)]
    q[idx] = 0
    return ref, q


def test_paired_bootstrap_covers_true_delta():
    ref, q = _vectors()
    true_delta = (q.mean() - ref.mean()) * 100  # delta is defined quant minus ref
    d, lo, hi = matched.paired_bootstrap(ref, q, B=2000, seed=0)
    assert abs(d - true_delta) < 0.5
    assert lo <= true_delta <= hi
    assert lo < d < hi


def test_paired_bootstrap_identical_vectors_zero():
    rng = np.random.default_rng(1)
    v = (rng.random(500) < 0.5).astype(np.int8)
    d, lo, hi = matched.paired_bootstrap(v, v, B=1000, seed=0)
    assert d == 0.0 and lo == 0.0 and hi == 0.0


def test_flips_directionality():
    ref = np.array([1, 1, 0, 0, 1, 0], dtype=np.int8)
    q = np.array([0, 1, 1, 0, 1, 1], dtype=np.int8)
    c2w, w2c = matched.flips(ref, q)
    assert c2w == 1 / 6 and w2c == 2 / 6  # idx0 c2w; idx2,idx5 w2c


def test_wilson_known_interval():
    # k=0, n=10: Wilson upper bound ~0.308 at z=1.96
    lo, hi = matched.wilson(0, 10)
    assert lo == 0.0
    assert 0.27 <= hi <= 0.35
    # symmetric sanity: k=n gives lower bound mirrored
    lo2, _ = matched.wilson(10, 10)
    assert 0.65 <= lo2 <= 0.73


def test_chance_flag_around_025():
    # 25/100 exactly: Wilson CI of 0.25 contains 0.25 -> flag True
    v = np.zeros(100, dtype=np.int8)
    v[:25] = 1
    assert matched.chance_flag(v) is True
    v2 = np.zeros(1000, dtype=np.int8)
    v2[:600] = 1
    assert matched.chance_flag(v2) is False


def test_mcnemar_matches_scipy_and_edge_cases():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0  # symmetric
    p = mcnemar_exact(10, 0)
    assert abs(p - 2 ** -10 * 2) < 1e-9  # two-sided binomial with p=0.5, n=10


def test_holm_step_down():
    # sorted [0.01, 0.03, 0.04]: 0.01<=0.0167 ok; 0.03>0.025 -> stop
    assert list(holm([0.01, 0.04, 0.03])) == [True, False, False]
    assert list(holm([0.03, 0.04])) == [False, False]  # smallest fails 0.025
    assert list(holm([0.001, 0.04])) == [True, True]
    assert list(holm([0.001, 0.03, 0.9])) == [True, False, False]
    assert list(holm([1.0])) == [False]


def test_paired_bootstrap_lang_gap():
    # deterministic: tr quant loses 10pp, en quant loses 2pp -> gap = 8pp
    tr = np.ones(100, dtype=np.int8); tr[-20:] = 0
    en = np.ones(100, dtype=np.int8); en[-20:] = 0
    q_tr = tr.copy(); q_tr[:10] = 0
    q_en = en.copy(); q_en[:2] = 0
    d, lo, hi = matched.paired_bootstrap_lang_gap(tr, q_tr, en, q_en, B=1000, seed=0)
    assert lo <= d <= hi
    assert abs(d - 8.0) < 0.5