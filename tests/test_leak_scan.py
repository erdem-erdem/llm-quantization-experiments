"""verify_checklist c9: GPQA leak scan flags dataset field names, never reports content."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import verify_checklist  # noqa: E402
from scripts.common import paths  # noqa: E402


def test_scan_flags_field_names_in_reports_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "reports" / "E1.md").write_text("tablo: Δ ve CI", encoding="utf-8")
    ok, _ = verify_checklist.c9_gpqa_leak_scan()
    assert ok
    (tmp_path / "logs" / "lm_eval.log").write_text("row: Incorrect Answer 1 = x", encoding="utf-8")
    ok, msg = verify_checklist.c9_gpqa_leak_scan()
    assert not ok and "lm_eval.log" in msg and "= x" not in msg  # name + pattern only
