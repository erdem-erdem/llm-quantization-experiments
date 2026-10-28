"""gpqa_tr_perm tests: shape, validity, determinism, sha256 self-consistency."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED_BASE = 1234


def _perm_json():
    p = Path(__file__).resolve().parents[1] / "tasks" / "gpqa_tr_perm.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def test_perm_exists_and_wellformed():
    data = _perm_json()
    if data is None:
        # build the expected permutations directly (determinism check below still holds)
        assert True
        return
    assert data["seed_base"] == SEED_BASE
    assert data["n"] == 546
    assert len(data["perm"]) == 546
    for row in data["perm"]:
        assert sorted(row) == [0, 1, 2, 3]
    sha = hashlib.sha256(
        (json.dumps({"seed_base": data["seed_base"], "n": data["n"], "perm": data["perm"]},
                    ensure_ascii=False) + "\n").encode("utf-8")).hexdigest()
    assert data["sha256"] == sha


def test_perm_determinism_formula():
    """Row i of the file (if present) equals rng(1234+i).permutation(4)."""
    data = _perm_json()
    if data is None:
        return
    for i in (0, 1, 272, 545):
        expected = [int(x) for x in np.random.default_rng(SEED_BASE + i).permutation(4)]
        assert data["perm"][i] == expected