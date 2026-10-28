"""2-GPU safety: free-before/after checks, out-of-service cards, heavy-job cap, process groups."""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import scheduler as sched_mod  # noqa: E402
from scripts.common import gpu as gpu_util  # noqa: E402
from scripts.common import paths  # noqa: E402
from scripts.scheduler import Scheduler, run_isolated  # noqa: E402

OK = [sys.executable, "-c", "pass"]


def _job(jid, deps=(), kind="cpu", gpu=None, job_name=None):
    return {"id": jid, "exp": "T", "kind": kind, "cmd": OK, "deps": list(deps), "outputs": [],
            "est_min": 1.0, "gpu_affinity": gpu, "validator": None, "gpu": None,
            "job_name": job_name or jid, "key": "k"}


@pytest.fixture(autouse=True)
def _tmp_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(paths, "JOBS", tmp_path / "logs" / "jobs")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results")
    return tmp_path


# ---- gpu.wait_free ---------------------------------------------------------------------------

def test_wait_free_returns_when_memory_releases_naturally(monkeypatch):
    seq = iter([40000, 30000, 500])
    monkeypatch.setattr(gpu_util, "memory_used_mib", lambda g: next(seq))
    killed = []
    monkeypatch.setattr(gpu_util, "kill_pids", lambda p: killed.append(p) or p)
    ok, used = gpu_util.wait_free(1, "t", sleep=lambda s: None)
    assert ok and used == 500 and killed == []


def test_wait_free_kills_leftover_compute_processes_then_succeeds(monkeypatch):
    state = {"mem": 60000}
    monkeypatch.setattr(gpu_util, "memory_used_mib", lambda g: state["mem"])
    monkeypatch.setattr(gpu_util, "compute_pids", lambda g: [4242])

    def kill(pids):
        state["mem"] = 0  # the orphan dies, the driver releases the memory
        return pids
    monkeypatch.setattr(gpu_util, "kill_pids", kill)
    ok, used = gpu_util.wait_free(0, "t", natural_s=6.0, sleep=lambda s: None)
    assert ok and used == 0


def test_wait_free_gives_up_on_a_card_that_never_frees(monkeypatch):
    monkeypatch.setattr(gpu_util, "memory_used_mib", lambda g: 70000)
    monkeypatch.setattr(gpu_util, "compute_pids", lambda g: [])
    monkeypatch.setattr(gpu_util, "kill_pids", lambda p: [])
    ok, used = gpu_util.wait_free(0, "t", natural_s=3.0, after_kill_s=3.0, sleep=lambda s: None)
    assert not ok and used == 70000


def test_wait_free_is_a_noop_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(gpu_util, "memory_used_mib", lambda g: None)
    assert gpu_util.wait_free(0, "t") == (True, None)


# ---- scheduler -------------------------------------------------------------------------------

def test_busy_card_is_taken_out_of_service_and_the_run_still_ends(monkeypatch):
    """GPU 1 never frees: its jobs stay pending, GPU 0 and the CPU side finish, no hang."""
    monkeypatch.setattr(gpu_util, "wait_free",
                        lambda g, why, **kw: (g != 1, 70000 if g == 1 else 0))
    jobs = [_job("a", kind="gpu", gpu=0), _job("b", kind="gpu", gpu=1),
            _job("c", deps=["b"]), _job("d", deps=["a"])]
    sch = Scheduler(jobs, 2)
    done = threading.Event()
    rc = {}

    def go():
        rc["v"] = sch.run()
        done.set()
    threading.Thread(target=go, daemon=True).start()
    assert done.wait(timeout=60), "scheduler hung with a card out of service"
    assert sch.state["a"] == "done" and sch.state["d"] == "done"
    assert sch.state["b"] == "pending" and sch.state["c"] == "pending"
    assert 1 in sch.dead_gpus and rc["v"] == 1


def test_gpu_is_checked_before_and_after_every_gpu_job(monkeypatch):
    calls = []
    monkeypatch.setattr(gpu_util, "wait_free",
                        lambda g, why, **kw: calls.append((g, why)) or (True, 0))
    sch = Scheduler([_job("a", kind="gpu", gpu=0), _job("b", kind="gpu", gpu=0, deps=["a"])], 1)
    assert sch.run() == 0
    assert [w for _, w in calls] == ["before a", "after a", "before b", "after b"]


def test_heavy_cpu_jobs_never_exceed_the_cap(monkeypatch):
    live, peak, lock = [0], [0], threading.Lock()

    def fake_run(cmd, cwd, env, log_file, timeout_s):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.15)
        with lock:
            live[0] -= 1
        return 0, False
    monkeypatch.setattr(sched_mod, "run_isolated", fake_run)
    jobs = [_job(f"deq:{i}", job_name="dequant") for i in range(6)]
    sch = Scheduler(jobs, 1)
    assert sch.run() == 0
    assert peak[0] <= sched_mod.HEAVY_CONCURRENCY and peak[0] >= 1


def test_gpu_jobs_see_only_their_card_and_cpu_jobs_see_none(monkeypatch):
    seen = {}

    def fake_run(cmd, cwd, env, log_file, timeout_s):
        seen[cmd[-1]] = env["CUDA_VISIBLE_DEVICES"]
        return 0, False
    monkeypatch.setattr(sched_mod, "run_isolated", fake_run)
    monkeypatch.setattr(gpu_util, "wait_free", lambda g, why, **kw: (True, 0))
    g1, c = _job("g1", kind="gpu", gpu=1), _job("c1")
    g1["cmd"], c["cmd"] = ["x", "gpu-job"], ["x", "cpu-job"]
    assert Scheduler([g1, c], 2).run() == 0
    assert seen == {"gpu-job": "1", "cpu-job": ""}


def test_disk_guard_covers_downloads_and_gguf_conversion(tmp_path, monkeypatch):
    assert "gguf:" in sched_mod.DISK_GUARDED and "dl:" in sched_mod.DISK_GUARDED
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "model_cards.csv").write_text(
        "model,total_params\nq32,32000000000\n", encoding="utf-8")
    j = _job("gguf:q32")
    j["key"] = "q32"
    assert sched_mod.expected_out_bytes(j) == 64_000_000_000     # bf16: 2 bytes per parameter


# ---- process isolation -----------------------------------------------------------------------

def test_timeout_kills_the_job_and_reports_it(tmp_path):
    with open(tmp_path / "log.txt", "w") as lf:
        t0 = time.monotonic()
        rc, timed_out = run_isolated([sys.executable, "-c", "import time; time.sleep(60)"],
                                     tmp_path, dict(os.environ), lf, timeout_s=1.0)
    assert timed_out and rc != 0 and time.monotonic() - t0 < 30


@pytest.mark.skipif(os.name != "posix", reason="process groups / SIGKILL of a group are POSIX-only")
def test_orphaned_grandchild_dies_with_the_job(tmp_path):
    pidfile = tmp_path / "child.pid"
    code = ("import subprocess, sys; "
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
            f"open(r'{pidfile}', 'w').write(str(p.pid))")  # parent exits at once, child lingers
    with open(tmp_path / "log.txt", "w") as lf:
        rc, _ = run_isolated([sys.executable, "-c", code], tmp_path, dict(os.environ), lf, 30.0)
    pid = int(pidfile.read_text())
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # raises when the process no longer exists
