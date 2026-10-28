"""T0: GPQA-tr option permutations. p_i = rng(1234 + i).permutation(4); never print question text.

Writes tasks/gpqa_tr_perm.json: {"seed_base", "n", "perm": [[...]...], "sha256"} — fixed once,
reused by every run (the lm_eval task YAML consumes it to reorder choices deterministically).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import io, logging_setup, paths  # noqa: E402

SEED_BASE = 1234


def main() -> int:
    logging_setup.setup("t0_gpqa_perm")
    from datasets import load_dataset

    ds = load_dataset("ytu-ce-cosmos/gpqa-extended_tr", split="train")
    perm = []
    for i in range(len(ds)):
        p = np.random.default_rng(SEED_BASE + i).permutation(4)
        perm.append([int(x) for x in p])
    n = len(perm)
    assert n == 546, f"gpqa_tr rows = {n}, expected 546"
    body = json.dumps({"seed_base": SEED_BASE, "n": n, "perm": perm}, ensure_ascii=False)
    paths.TASKS.mkdir(parents=True, exist_ok=True)
    dest = paths.TASKS / "gpqa_tr_perm.json"
    io.atomic_write_text(body + "\n", dest)
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    io.atomic_write_text(json.dumps({"seed_base": SEED_BASE, "n": n, "perm": perm,
                                     "sha256": sha}, ensure_ascii=False) + "\n", dest)
    print(f"wrote {dest}: n={n}, sha256={sha[:12]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())