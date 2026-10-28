"""Significance tests: McNemar exact, Holm correction."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar p-value. b = ref right/quant wrong, c = ref wrong/quant right."""
    if b + c == 0:
        return 1.0
    m = min(b, c)
    return float(sps.binomtest(m, b + c, 0.5).pvalue)


def holm(pvals: list[float] | np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm step-down: returns boolean mask of hypotheses rejected at alpha."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    out = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if p[i] > alpha / (m - rank):
            break  # step-down: this and all larger p-values survive
        out[i] = True
    return out