"""arc_tr_check.check_rows: the doc_to_target mapping must be valid for every row."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval import arc_tr_check  # noqa: E402


def _row(key, labels="ABCD"):
    return {"question": "q?", "answerKey": key,
            "choices": {"label": list(labels), "text": [f"t{c}" for c in labels]}}


def test_valid_rows_pass_even_with_mixed_label_formats(monkeypatch):
    monkeypatch.setattr(arc_tr_check, "EXPECTED_ROWS", 3)
    assert arc_tr_check.check_rows([_row("A"), _row("2", "1234"), _row(" C", "ABC")]) == []


def test_answer_key_outside_labels_and_wrong_count_are_flagged(monkeypatch):
    monkeypatch.setattr(arc_tr_check, "EXPECTED_ROWS", 3)
    problems = arc_tr_check.check_rows([_row("A"), _row("E")])
    assert any("row count" in p for p in problems)
    assert any("not in choices.label" in p for p in problems)
