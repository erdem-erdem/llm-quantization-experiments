"""Matched-sample statistics (plan §6.7). All statistics live here; figures only read master.csv."""
from __future__ import annotations

import numpy as np

DEFAULT_B = 10_000


def paired_bootstrap(correct_ref: np.ndarray, correct_q: np.ndarray,
                     B: int = DEFAULT_B, seed: int = 0) -> tuple[float, float, float]:
    """Δacc (pp), 95% percentile CI, resampling the SAME question indices for both models."""
    ref = np.asarray(correct_ref, dtype=np.int8)
    q = np.asarray(correct_q, dtype=np.int8)
    assert ref.shape == q.shape, f"sample vectors differ: {ref.shape} vs {q.shape}"
    n = len(ref)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    d = (q[idx].mean(axis=1) - ref[idx].mean(axis=1)) * 100.0
    delta = float((q.mean() - ref.mean()) * 100.0)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return delta, float(lo), float(hi)


def paired_bootstrap_avg(pairs: list[tuple[np.ndarray, np.ndarray]],
                         B: int = DEFAULT_B, seed: int = 0) -> tuple[float, float, float]:
    """CI of the mean delta across tasks with DIFFERENT questions (e.g. tr+ARC mean).

    Within each replicate, indices are drawn independently per task, then the per-task
    deltas are averaged (spec §7.1: "iki görevin bootstrap örnekleri birlikte çekilerek").
    """
    arrays = [(np.asarray(r, dtype=np.int8), np.asarray(q, dtype=np.int8)) for r, q in pairs]
    for r, q in arrays:
        assert r.shape == q.shape
    n_tasks = len(arrays)
    rng = np.random.default_rng(seed)
    deltas = np.empty((B, n_tasks))
    point = float(np.mean([(q.mean() - r.mean()) * 100.0 for r, q in arrays]))
    for j, (r, q) in enumerate(arrays):
        n = len(r)
        idx = rng.integers(0, n, size=(B, n))
        deltas[:, j] = (q[idx].mean(axis=1) - r[idx].mean(axis=1)) * 100.0
    avg = deltas.mean(axis=1)
    lo, hi = np.percentile(avg, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_bootstrap_lang_gap(ref_tr: np.ndarray, q_tr: np.ndarray,
                              ref_en: np.ndarray, q_en: np.ndarray,
                              B: int = DEFAULT_B, seed: int = 0) -> tuple[float, float, float]:
    """TR−EN loss gap: the SAME row indices are drawn for both languages (aligned Belebele rows)."""
    rt, qt, re_, qe = (np.asarray(a, dtype=np.int8) for a in (ref_tr, q_tr, ref_en, q_en))
    assert rt.shape == qt.shape and re_.shape == qe.shape
    assert len(rt) == len(re_), "Belebele tr/en rows must be aligned"
    n = len(rt)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    # loss = accuracy delta; loss_gap = loss_tr - loss_en
    d_tr = (rt[idx].mean(axis=1) - qt[idx].mean(axis=1)) * 100.0
    d_en = (re_[idx].mean(axis=1) - qe[idx].mean(axis=1)) * 100.0
    gap = d_tr - d_en
    point = float((rt.mean() - qt.mean()) * 100.0 - (re_.mean() - qe.mean()) * 100.0)
    lo, hi = np.percentile(gap, [2.5, 97.5])
    return point, float(lo), float(hi)


def flips(correct_ref: np.ndarray, correct_q: np.ndarray) -> tuple[float, float]:
    """flip_c2w (ref right, quant wrong) and flip_w2c, as fractions."""
    ref = np.asarray(correct_ref, dtype=np.int8)
    q = np.asarray(correct_q, dtype=np.int8)
    assert ref.shape == q.shape
    c2w = float(np.mean((ref == 1) & (q == 0)))
    w2c = float(np.mean((ref == 0) & (q == 1)))
    return c2w, w2c


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(center - half), float(center + half)


def chance_flag(correct_ref: np.ndarray, B: int = DEFAULT_B, seed: int = 0) -> bool:
    """True when BF16 accuracy's 95% CI contains 0.25 (spec §7.1 Şans kuralı)."""
    ref = np.asarray(correct_ref, dtype=np.int8)
    n = len(ref)
    k = int(ref.sum())
    lo, hi = wilson(k, n)
    return bool(lo <= 0.25 <= hi)


def breaking_point(quant_bpws: dict[str, float], deltas: dict[str, float],
                   ci_highs: dict[str, float]) -> str | None:
    """First quant (bpw descending) with tr_arc_avg delta <= -2.0 and ci_high < 0 (K-05)."""
    order = sorted(quant_bpws, key=lambda q: quant_bpws[q], reverse=True)
    for q in order:
        if deltas[q] <= -2.0 and ci_highs[q] < 0:
            return q
    return None