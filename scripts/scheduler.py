"""DAG executor (plan §7): one worker thread per GPU (each pins CUDA_VISIBLE_DEVICES=i and
pulls jobs assigned to that GPU in listed order whose deps are done) + a CPU pool
(min(16, cores), RAM/disk-heavy jobs capped). Every job runs in its own process group with a
timeout; the GPU is checked free (and leftovers killed) before and after each GPU job.
--dry-run prints the topological order and
per-GPU timeline. Disk guard before dequant/KL jobs: free space >= 2x expected output,
else wait. Honors .done markers (resume). Logs to logs/sched.log; SIGINT drains.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import gpu as gpu_util  # noqa: E402
from scripts.common import logging_setup, paths  # noqa: E402
from scripts.exp import jobspec  # noqa: E402

DEQUANT_PREFIX = "deq"
DISK_POLL_S = 30.0
# RAM/disk-heavy CPU jobs (by record_run job name): at most this many at once, so a 32B
# conversion + dequant + quantize cannot exhaust memory or saturate the disk together.
HEAVY_CPU_JOBS = {"gguf_convert", "gguf_quantize", "dequant", "produce", "head", "download"}
HEAVY_CONCURRENCY = 2
# Jobs whose output size is checked against free disk before they start (id prefixes).
DISK_GUARDED = (DEQUANT_PREFIX, "kl", "gguf:", "dl:")
JOB_TIMEOUT_S = 4 * 3600  # a hung job must not stall its GPU forever (longest real job ~1-2 h)


def _kill_group(p: subprocess.Popen) -> None:
    """SIGKILL the job's whole process group (orphaned vLLM engines / llama-servers included)."""
    if os.name != "posix":
        if p.poll() is None:
            p.kill()
        return
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass  # group already gone


def run_isolated(cmd: list[str], cwd: Path, env: dict, log_file, timeout_s: float) -> tuple[int, bool]:
    """Run `cmd` in its own session so that, once it exits (or times out), every process it left
    behind is killed with the group. Ctrl-C on the scheduler therefore drains instead of killing
    running jobs. Returns (returncode, timed_out)."""
    kw = {"start_new_session": True} if os.name == "posix" else {}
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log_file, stderr=subprocess.STDOUT, **kw)
    timed_out = False
    try:
        p.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(p)
        p.wait()
    finally:
        _kill_group(p)  # leftovers of a finished or failed job
    return p.returncode, timed_out


def marker_of(job: dict) -> Path:
    return paths.done_marker(job["exp"], job["job_name"], job["key"])


def expected_out_bytes(job: dict) -> int:
    """Rough output size for the disk guard: dequant / BF16 GGUF conversion / HF download →
    bf16 (2 bytes/param of the model); KL → 8 GiB."""
    if job["id"].startswith((DEQUANT_PREFIX, "gguf:", "dl:")):
        model = job["key"].split("__")[0]
        cards = paths.RESULTS / "model_cards.csv"
        if cards.exists():
            import csv as _csv
            with open(cards, encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    if row.get("model") == model and row.get("total_params"):
                        return int(float(row["total_params"]) * 2)
    return 8 * (1 << 30)


class Scheduler:
    def __init__(self, jobs: list[dict], n_gpus: int, check_gpus: bool = True):
        self.check_gpus = check_gpus
        # per GPU: (memory MiB, compute PIDs) before any job ran = what is not ours
        self.gpu_baseline: dict[int, tuple[int | None, frozenset[int]]] = {}
        self.jobs = jobs
        self.by_id = {j["id"]: j for j in jobs}
        self.state: dict[str, str] = {j["id"]: "pending" for j in jobs}
        self.lock = threading.RLock()  # reentrant: _ready → _mark nests under cond
        self.cond = threading.Condition(self.lock)
        self.n_gpus = n_gpus
        self.stop = threading.Event()
        self.heavy_slots = threading.BoundedSemaphore(HEAVY_CONCURRENCY)
        self.dead_gpus: set[int] = set()  # cards that would not come back to idle
        self.log = logging_setup.setup("sched")
        self.failures: list[str] = []

    def resume_done(self) -> int:
        n = 0
        for j in self.jobs:
            if not marker_of(j).exists():
                continue
            missing = [o for o in j.get("outputs", []) if not (paths.ROOT / o).exists()]
            if missing:  # marker without its declared outputs (deleted by hand): rerun
                self.log.warning("%s: .done marker present but output missing %s; rerunning",
                                 j["id"], missing)
                continue
            self.state[j["id"]] = "done"
            n += 1
        return n

    def _mark(self, jid: str, st: str) -> None:
        with self.cond:
            self.state[jid] = st
            self.cond.notify_all()

    def _ready(self, jid: str) -> bool:
        j = self.by_id[jid]
        for d in j["deps"]:
            st = self.state.get(d, "missing")
            if st == "done":
                continue
            # soft_deps (attempted:) can be failed/skipped but still allow job to run
            if d.startswith("attempted:"):
                actual_dep = d[len("attempted:"):]
                st = self.state.get(actual_dep, "missing")
                if st in ("done", "failed", "skipped", "missing"):
                    if st == "missing":
                        self.log.warning("attempted:%s is missing, treating as satisfied", actual_dep)
                    continue
                # attempted dep not run yet; wait
                return False
            if st in ("failed", "skipped", "missing"):
                self._mark(jid, "skipped")
                self.log.warning("skip %s: dep %s is %s", jid, d, st)
                return False
            return False
        return True

    def _stuck(self) -> bool:
        """Nothing is running and nothing pending is runnable: what remains waits on work that
        will never finish (e.g. the jobs of a card taken out of service). Ends the run instead
        of waiting forever; the summary lists those jobs as still pending."""
        with self.cond:
            if any(s == "running" for s in self.state.values()):
                return False
            for i, s in self.state.items():
                j = self.by_id[i]
                if s != "pending":
                    continue
                if j["kind"] == "gpu" and j["gpu_affinity"] in self.dead_gpus:
                    continue  # no worker will ever pick it up
                if self._ready(i):
                    return False
            return True

    def _wait_disk(self, job: dict) -> bool:
        """Wait for the disk guard; False if a stop was requested while waiting."""
        need = 2 * expected_out_bytes(job)
        while not self.stop.is_set():
            free = shutil.disk_usage(paths.ROOT).free
            if free >= need:
                return True
            self.log.warning("disk guard %s: free %.1f GiB < needed %.1f GiB, waiting",
                             job["id"], free / 2**30, need / 2**30)
            time.sleep(DISK_POLL_S)
        return False

    def _run(self, job: dict) -> None:
        jid = job["id"]
        if jid.startswith(DISK_GUARDED):
            if not self._wait_disk(job):
                self._mark(jid, "pending")  # stop requested; job stays pending for resume
                return
        env = dict(os.environ)
        # a GPU job sees exactly its card; a CPU job sees none, so it can never grab memory on a
        # card a pinned job is using (it would fail loudly instead of silently contending)
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu_affinity"]) if job["kind"] == "gpu" else ""
        heavy = job["kind"] == "cpu" and job.get("job_name") in HEAVY_CPU_JOBS
        paths.JOBS.mkdir(parents=True, exist_ok=True)
        log_path = paths.JOBS / f"{job['exp']}__{job['job_name']}__{job['key']}.log"
        if heavy:
            self.heavy_slots.acquire()
        try:
            self.log.info("start %s on %s (log %s)", jid,
                          f"gpu{job['gpu_affinity']}" if job["kind"] == "gpu" else "cpu",
                          log_path.name)
            with open(log_path, "a", encoding="utf-8") as lf:
                rc, timed_out = run_isolated(job["cmd"], paths.ROOT, env, lf, JOB_TIMEOUT_S)
        finally:
            if heavy:
                self.heavy_slots.release()
        # a job that finished while the scheduler drains still counts: its process was not
        # interrupted (own session), so rc is the job's real outcome
        if rc != 0:
            why = f"timeout after {JOB_TIMEOUT_S // 3600} h" if timed_out else f"rc={rc}"
            self.failures.append(f"{jid} ({why}, log {log_path})")
            self._mark(jid, "failed")
            self.log.error("FAIL %s %s", jid, why)
        else:
            self._mark(jid, "done")
            self.log.info("done %s", jid)

    def _gpu_idle(self, gpu: int, when: str) -> bool:
        """The card must hold (almost) nothing before the next job starts and after the last
        one ends. Leftover compute processes are killed; a card that still will not come back is
        taken out of service (its remaining jobs stay pending) instead of OOM-ing every job."""
        if not self.check_gpus:
            return True
        base_mib, base_pids = self.gpu_baseline.get(gpu, (None, frozenset()))
        ok, used = gpu_util.wait_free(gpu, when, baseline_mib=base_mib or 0, keep_pids=base_pids)
        if used is not None:
            self.log.info("gpu%s %s: %s MiB in use", gpu, when, used)
        if not ok:
            self.dead_gpus.add(gpu)
            self.log.error("gpu%s still holds %s MiB %s after killing leftovers: taking it out "
                           "of service; its remaining jobs stay pending — free the card and "
                           "rerun the scheduler", gpu, used, when)
        return ok

    def _gpu_worker(self, gpu: int) -> None:
        mine = [j for j in self.jobs if j["kind"] == "gpu" and j["gpu_affinity"] == gpu]
        while not self.stop.is_set():
            nxt = None
            with self.cond:
                while not self.stop.is_set():
                    nxt = next((j for j in mine if self.state[j["id"]] == "pending"
                                and self._ready(j["id"])), None)
                    if nxt is not None:
                        break
                    if not any(self.state[j["id"]] == "pending" for j in mine):
                        return  # this GPU has nothing left
                    if self._stuck():
                        self.log.error("gpu%s: remaining jobs wait on work that will never "
                                       "run (a failed/out-of-service dependency chain)", gpu)
                        return
                    self.cond.wait(timeout=1.0)
            if nxt is None:
                return
            if not self._gpu_idle(gpu, f"before {nxt['id']}"):
                return  # card unusable: leave its remaining jobs pending
            self._mark(nxt["id"], "running")
            self._run(nxt)
            if not self._gpu_idle(gpu, f"after {nxt['id']}"):
                return

    def _cpu_dispatcher(self, pool: "concurrent.futures.ThreadPoolExecutor") -> None:
        import concurrent.futures
        futures: list = []
        while not self.stop.is_set():
            nxt = []
            with self.cond:
                nxt = [j for j in self.jobs if j["kind"] == "cpu"
                       and self.state[j["id"]] == "pending" and self._ready(j["id"])]
                for j in nxt:
                    self.state[j["id"]] = "running"
            if not nxt:
                with self.cond:
                    if any(self.state[j["id"]] == "pending" for j in self.jobs
                           if j["kind"] == "cpu"):
                        if self._stuck():
                            return  # leftover CPU jobs wait on GPU work that will never run
                        self.cond.wait(timeout=1.0)
                        continue
                    # no cpu job left pending; GPU jobs may still be running
                    if any(self.state[j["id"]] == "running" for j in self.jobs):
                        self.cond.wait(timeout=1.0)
                        continue
                    return
            for j in nxt:  # heavy jobs queue on self.heavy_slots inside _run, not here
                futures.append(pool.submit(self._run, j))
        for f in futures:
            f.result()

    def run(self) -> int:
        import concurrent.futures
        if self.check_gpus:
            for g in range(self.n_gpus):
                self.gpu_baseline[g] = gpu_util.snapshot(g)
                self.log.info("gpu%s baseline before any job: %s MiB, pids %s", g,
                              self.gpu_baseline[g][0], sorted(self.gpu_baseline[g][1]))
        n_cpu = max(4, min(16, os.cpu_count() or 8))
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_cpu) as pool:
            threads = [threading.Thread(target=self._gpu_worker, args=(g,), name=f"gpu{g}")
                       for g in range(self.n_gpus)]
            threads.append(threading.Thread(target=self._cpu_dispatcher, args=(pool,),
                                            name="cpu"))
            for t in threads:
                t.start()
            try:
                for t in threads:
                    t.join()
            except KeyboardInterrupt:
                self.log.warning("SIGINT: draining — no new jobs will start")
                self.stop.set()
                with self.cond:
                    self.cond.notify_all()
                for t in threads:
                    t.join()
        skipped = [i for i, s in self.state.items() if s == "skipped"]
        failed = [i for i, s in self.state.items() if s == "failed"]
        pend = [i for i, s in self.state.items() if s == "pending"]
        self.log.info("summary: %d done, %d failed, %d skipped, %d still pending",
                      len(self.jobs) - len(skipped) - len(failed) - len(pend),
                      len(failed), len(skipped), len(pend))
        return 1 if (failed or skipped or pend) else 0


def dry_run(jobs: list[dict], n_gpus: int) -> None:
    """Topological order + per-GPU timeline estimate."""
    by_id = {j["id"]: j for j in jobs}
    indeg = {i: len(j["deps"]) for i, j in by_id.items()}
    dependents: dict[str, list[str]] = {i: [] for i in by_id}
    for j in jobs:
        for d in j["deps"]:
            dependents[jobspec.dep_id(d)].append(j["id"])
    ready = sorted(i for i, k in indeg.items() if k == 0)
    order: list[str] = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for dep in dependents[i]:
            indeg[dep] -= 1
            if indeg[dep] == 0:
                ready.append(dep)
        ready.sort()
    if len(order) != len(jobs):
        stuck = sorted(set(by_id) - set(order))
        raise RuntimeError(f"dependency cycle involving: {stuck}")
    print(f"{len(jobs)} jobs, topological order:")
    for rank, i in enumerate(order, 1):
        j = by_id[i]
        tgt = f"gpu{j['gpu_affinity']}" if j["kind"] == "gpu" else "cpu"
        print(f"  {rank:4d} [{tgt:5s}] {i:34s} {j['est_min']:6.1f} min")
    print("\nper-GPU timeline estimate (listed order, deps permitting):")
    for g in range(n_gpus):
        mine = [j for j in jobs if j["kind"] == "gpu" and j["gpu_affinity"] == g]
        tot = sum(j["est_min"] for j in mine)
        print(f"  gpu{g}: {len(mine)} jobs, {tot / 60:.1f} h -> "
              + " ".join(j["id"] for j in mine))
    cpu = [j for j in jobs if j["kind"] == "cpu"]
    print(f"  cpu pool: {len(cpu)} jobs, {sum(j['est_min'] for j in cpu) / 60:.1f} h total")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--no-gpu-check", action="store_true",
                    help="skip the free-GPU checks (dev machines whose GPU is shared)")
    args = ap.parse_args()
    log = logging_setup.setup("sched")
    q = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    n_gpus = args.gpus or q.get("gpus", 2)
    jobs = q["jobs"]
    if args.dry_run:
        dry_run(jobs, n_gpus)
        return 0
    sch = Scheduler(jobs, n_gpus, check_gpus=not args.no_gpu_check)
    signal.signal(signal.SIGINT, lambda *_: sch.stop.set())
    n = sch.resume_done()
    log.info("resume: %d/%d jobs already done", n, len(jobs))
    return sch.run()


if __name__ == "__main__":
    sys.exit(main())

