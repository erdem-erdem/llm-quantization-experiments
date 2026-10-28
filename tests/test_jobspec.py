"""jobspec: GPU jobs are pinned through record_run --gpu (budget accounting + E4 exclusivity)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp import jobspec  # noqa: E402


def test_gpu_job_gets_gpu_flag_and_keeps_marker_positions():
    cmd = jobspec.record_cmd("E1", "eval", "q8__fp8__none__s0", ["echo", "hi"],
                             validator="a.b:c")
    j = jobspec.job("ev:x", "E1", "gpu", cmd, gpu_affinity=2)
    i = j["cmd"].index("--gpu")
    assert j["cmd"][i + 1] == "2" and j["cmd"].index("--") > i
    assert (j["job_name"], j["key"]) == ("eval", "q8__fp8__none__s0")
    assert j["cmd"][j["cmd"].index("--") + 1:] == ["echo", "hi"]


def test_cpu_job_is_not_pinned():
    j = jobspec.job("q:x", "E1", "cpu", jobspec.record_cmd("E1", "q", "k", ["echo"]))
    assert "--gpu" not in j["cmd"]


def test_full_queue_is_a_valid_dag_with_soft_deps_and_markers():
    from scripts import queue_gen

    jobs = queue_gen.validate(queue_gen.collect("full"), "full")  # raises on dangling deps
    by_id = {j["id"]: j for j in jobs}
    assert any(d.startswith(jobspec.SOFT_PREFIX) for d in by_id["master:build"]["deps"])
    # gate reads the E1 llama.cpp KL rows of every monotonic quant: they must be deps
    gate = by_id["gate:E0"]
    assert {"kl:q8__q8_0__none__s0", "kl:q8__q2_k__tr__s0"} <= set(gate["deps"])
    # gate/theory get .done markers like every other job (no re-run on scheduler restart)
    assert gate["job_name"] == "gate" and by_id["theory:q8"]["job_name"] == "theory"
    assert all(j["job_name"] for j in jobs)


def _full_queue():
    from scripts import queue_gen

    jobs = queue_gen.validate(queue_gen.collect("full"), "full")
    return jobs, {j["id"]: j for j in jobs}


def test_two_gpu_plan_pins_every_gpu_job_to_gpu0_or_gpu1():
    jobs, by_id = _full_queue()
    assert {j["gpu_affinity"] for j in jobs if j["kind"] == "gpu"} == {0, 1}
    assert by_id["ev:q32__bf16__none__s0"]["gpu_affinity"] == 0     # the big model owns GPU 0
    assert by_id["ev:q14__bf16__none__s0"]["gpu_affinity"] == 1
    assert by_id["demo:hook"]["gpu_affinity"] == 0 and by_id["e5map"]["gpu_affinity"] == 1


def test_spec_cuts_for_two_gpus_are_applied():
    _, by_id = _full_queue()
    # cut 1: q32 q8_0/q3_k_m are produced + KL'd + speed-benched, but not evaluated/dequantized
    for rk in ("q32__q8_0__none__s0", "q32__q3_k_m__tr__s0"):
        assert f"ev:{rk}" not in by_id and f"deq:{rk}" not in by_id
        assert f"q:{rk}" in by_id and f"kl:{rk}" in by_id
    assert "ev:q32__q4_k_m__tr__s0" in by_id
    # cut 3: Belebele-en only for q4/q8/q32
    tasks = lambda j: j["cmd"][j["cmd"].index("--tasks") + 1].split(",")  # noqa: E731
    assert "belebele_en" in tasks(by_id["ev:q4__gptq__tr__s0"])
    assert "belebele_en" not in tasks(by_id["ev:q14__gptq__tr__s0"])
    # cut 2 + model swap: E2 runs q17 + q4, second seed only on q4
    assert "q2:q4__gptq__en__s1" in by_id and "q2:q17__gptq__en__s1" not in by_id
    assert not any(i.startswith(("ev2:q8", "q2:q8")) for i in by_id)
    #   q17 never had Belebele-en in E1, so E2 adds it for the reused (tr, s0) cell
    assert "belebele_en" in tasks(by_id["ev2:q17__gptq__tr__s0"])
    assert "belebele_en" not in tasks(by_id["ev2:q4__gptq__tr__s0"])
    # cut 5: no concurrency 64; cut 4: no 8B transfer stage
    serve = by_id["serve:bf16"]["cmd"]
    assert serve[serve.index("--concs") + 1] == "1,4,16"
    assert not any(i.startswith(("prod5:q8", "ev5:q8", "kl5:q8", "klref5:q8")) for i in by_id)
