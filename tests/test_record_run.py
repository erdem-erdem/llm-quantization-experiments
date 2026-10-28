"""record_run.py lifecycle test (M0 definition of done)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.common.paths as paths
import scripts.record_run as rr


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch):
    for attr in ("JOBS", "RESULTS", "LOGS"):
        monkeypatch.setattr(paths, attr, tmp_path / attr.lower())
    return tmp_path


def _run(sandbox: Path, monkeypatch, extra: list[str] | None = None) -> int:
    cmd = ["python", "-c", "print('hi')"]
    argv = ["record_run.py", "--exp", "T0", "--job", "demo", "--key", "k0"] + (extra or []) + cmd
    monkeypatch.setattr(sys, "argv", argv)
    return rr.main()


def test_record_run_writes_all_artifacts(sandbox: Path, monkeypatch):
    assert _run(sandbox, monkeypatch) == 0
    marker = paths.done_marker("T0", "demo", "k0")
    assert marker.exists()
    meta_path = paths.run_meta_path("T0", "k0", "demo")
    assert meta_path.exists()
    meta = __import__("json").loads(meta_path.read_text(encoding="utf-8"))
    assert meta["status"] == "ok" and meta["returncode"] == 0
    assert meta["cmd"] == ["python", "-c", "print('hi')"]
    assert meta["gpu_minutes"] == 0.0  # CPU job (no --gpu)
    budget = paths.budget_csv()
    rows = list(csv.DictReader(budget.open(encoding="utf-8", newline="\n")))
    assert rows and rows[0]["status"] == "ok" and rows[0]["gpu_min"] == "0.0"


def test_record_run_done_marker_skips(sandbox: Path, monkeypatch):
    assert _run(sandbox, monkeypatch) == 0
    marker = paths.done_marker("T0", "demo", "k0")
    marker.write_text("sentinel", encoding="utf-8")
    assert _run(sandbox, monkeypatch) == 0  # skipped
    assert marker.read_text(encoding="utf-8") == "sentinel"
    # force reruns
    assert _run(sandbox, monkeypatch, ["--force"]) == 0
    assert marker.read_text(encoding="utf-8") == ""


def test_record_run_failure_writes_fail_row(sandbox: Path, monkeypatch):
    argv = ["record_run.py", "--exp", "T0", "--job", "demo", "--key", "bad",
            "python", "-c", "import sys; sys.exit(3)"]
    monkeypatch.setattr(sys, "argv", argv)
    assert rr.main() == 1
    rows = list(csv.DictReader(paths.budget_csv().open(encoding="utf-8", newline="\n")))
    assert rows[-1]["status"] == "fail"
    assert not paths.done_marker("T0", "demo", "bad").exists()


def test_record_run_validator_gate(sandbox: Path, monkeypatch):
    argv = ["record_run.py", "--exp", "T0", "--job", "demo", "--key", "v",
            "--validator", "tests.test_record_run:_always_false",
            "python", "-c", "print('hi')"]
    monkeypatch.setattr(sys, "argv", argv)
    assert rr.main() == 1


def _always_false(args):  # pragma: no cover - imported by name above
    return False