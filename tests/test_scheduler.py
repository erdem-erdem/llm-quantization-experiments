"""Scheduler: topological dry-run, .done resume, dep-skip on failure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import paths  # noqa: E402
from scripts.scheduler import Scheduler, dry_run  # noqa: E402

OK = [sys.executable, "-c", "pass"]
BAD = [sys.executable, "-c", "import sys; sys.exit(3)"]


def _job(jid, cmd, deps=(), kind="cpu", gpu=None, exp="T", job_name=None, key="k"):
    return {"id": jid, "exp": exp, "kind": kind, "cmd": cmd, "deps": list(deps),
            "outputs": [], "est_min": 1.0, "gpu_affinity": gpu, "validator": None,
            "gpu": None, "job_name": job_name or jid, "key": key}


@pytest.fixture(autouse=True)
def _tmp_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(paths, "JOBS", tmp_path / "logs" / "jobs")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results")  # markers/run_meta stay in tmp
    return tmp_path


def _run(jobs, n_gpus=2):
    sch = Scheduler(jobs, n_gpus)
    sch.resume_done()
    return sch, sch.run()


def test_topological_order(tmp_path, capsys):
    jobs = [
        _job("b", OK, deps=["a"]),
        _job("a", OK),
        _job("g0", OK, deps=["b"], kind="gpu", gpu=0),
        _job("g1", OK, kind="gpu", gpu=1),
        _job("c", OK, deps=["a"]),
    ]
    q = tmp_path / "queue.json"
    q.write_text(json.dumps({"gpus": 2, "jobs": jobs}), encoding="utf-8")
    dry_run(jobs, 2)
    out = capsys.readouterr().out
    ia, ib = out.index(" a "), out.index(" b ")
    assert ia < ib, "a must precede b"
    for jid in ("a", "b", "g0", "g1", "c"):
        assert jid in out


def test_cycle_raises():
    jobs = [_job("a", OK, deps=["b"]), _job("b", OK, deps=["a"])]
    with pytest.raises(RuntimeError, match="cycle"):
        dry_run(jobs, 1)


def test_runs_all(_tmp_logs):
    jobs = [
        _job("a", OK, job_name="ja", key="ka"),
        _job("b", OK, deps=["a"], job_name="jb", key="kb"),
        _job("g", OK, deps=["b"], kind="gpu", gpu=0, job_name="jg", key="kg"),
        _job("g1", OK, kind="gpu", gpu=1, job_name="jg", key="k1"),
    ]
    sch, rc = _run(jobs)
    assert rc == 0
    assert all(s == "done" for s in sch.state.values())


def test_record_run_writes_marker_at_same_path(_tmp_logs, monkeypatch):
    """record_run (in-process) writes .done exactly where the scheduler looks for it."""
    from scripts import record_run
    monkeypatch.setattr(sys, "argv",
                        ["record_run.py", "--exp", "T", "--job", "ja", "--key", "ka",
                         "--", sys.executable, "-c", "pass"])
    assert record_run.main() == 0
    assert paths.done_marker("T", "ja", "ka").exists()


def test_done_marker_resume_skips_rerun(_tmp_logs):
    a = _job("a", BAD, job_name="ja", key="ka")  # would fail if rerun
    b = _job("b", OK, deps=["a"], job_name="jb", key="kb")
    paths.done_marker("T", "ja", "ka").parent.mkdir(parents=True, exist_ok=True)
    paths.done_marker("T", "ja", "ka").write_text("")
    sch, rc = _run([a, b])
    assert sch.state["a"] == "done" and sch.state["b"] == "done"
    assert rc == 0


def test_failed_dep_skips_dependents(_tmp_logs):
    bad = _job("bad", BAD, job_name="jbad", key="k")
    child = _job("child", OK, deps=["bad"], job_name="jchild", key="k")
    other = _job("other", OK, job_name="jother", key="k")
    sch, rc = _run([bad, child, other])
    assert sch.state["bad"] == "failed"
    assert sch.state["child"] == "skipped"
    assert sch.state["other"] == "done"
    assert rc == 1

def test_defect_c_attempted_dep_on_skipped_and_missing(_tmp_logs):
    """Test that attempted:X is satisfied when X is terminal (done/failed/skipped/missing)."""
    # Job A fails
    a = _job("a", BAD, job_name="ja", key="ka")
    # Job B depends on A (normal hard dependency), so B gets skipped
    b = _job("b", OK, deps=["a"], job_name="jb", key="kb")
    # Job C depends on A with soft dependency attempted:A
    # C should run because A is in terminal state (failed)
    c = _job("c", OK, deps=["attempted:a"], job_name="jc", key="kc")
    # Job D depends on B with soft dependency attempted:B
    # D should run because B is in terminal state (skipped)
    d = _job("d", OK, deps=["attempted:b"], job_name="jd", key="kd")
    
    sch, rc = _run([a, b, c, d])
    # A fails, B skips, but C and D should still run
    assert sch.state["a"] == "failed"
    assert sch.state["b"] == "skipped"
    assert sch.state["c"] == "done", f"c should run after failed A, got {sch.state['c']}"
    assert sch.state["d"] == "done", f"d should run after skipped B, got {sch.state['d']}"
    # rc should be 1 because A failed
    assert rc == 1


def test_resume_reruns_job_whose_declared_output_is_gone(_tmp_logs):
    a = _job("a", OK, job_name="ja", key="ka")
    a["outputs"] = ["out/a.txt"]  # relative to paths.ROOT (= tmp)
    paths.done_marker("T", "ja", "ka").parent.mkdir(parents=True, exist_ok=True)
    paths.done_marker("T", "ja", "ka").write_text("")
    sch = Scheduler([a], 1)
    assert sch.resume_done() == 0 and sch.state["a"] == "pending"  # marker alone is not enough
    (_tmp_logs / "out").mkdir()
    (_tmp_logs / "out" / "a.txt").write_text("x")
    sch = Scheduler([a], 1)
    assert sch.resume_done() == 1 and sch.state["a"] == "done"
