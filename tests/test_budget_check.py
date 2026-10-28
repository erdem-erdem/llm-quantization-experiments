"""budget_check exit codes: 0 under the trigger, 2 at the cut trigger, 3 at the hard cap."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import budget_check  # noqa: E402
from scripts.common import paths  # noqa: E402


@pytest.mark.parametrize("hours,code", [
    (budget_check.CUT_TRIGGER_GPU_H - 1, 0),
    (budget_check.CUT_TRIGGER_GPU_H, 2),
    (budget_check.HARD_CAP_GPU_H, 3),
    (budget_check.HARD_CAP_GPU_H + 1, 3),
])
def test_exit_codes(monkeypatch, tmp_path, hours, code):
    monkeypatch.setattr(paths, "ROOT", tmp_path)  # DECISIONS_NEEDED.md goes to tmp
    monkeypatch.setattr(budget_check, "total_gpu_minutes", lambda: (hours * 60.0, 3))
    monkeypatch.setattr(sys, "argv", ["budget_check", "--quiet"])
    assert budget_check.main() == code
