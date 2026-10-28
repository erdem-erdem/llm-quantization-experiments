"""E4/E6 tool-output parsers (formats taken from llama-bench / vllm bench sources)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import paths  # noqa: E402
from scripts.e4 import bench, theory  # noqa: E402


def test_parse_llama_bench_json():
    out = json.dumps([
        {"n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ts": 9000.5},
        {"n_prompt": 0, "n_gen": 256, "n_depth": 0, "avg_ts": 210.25}])
    assert bench.parse_llama_bench(out) == (9000.5, 210.25)
    with pytest.raises(RuntimeError, match="lacks"):
        bench.parse_llama_bench(json.dumps([{"n_prompt": 512, "n_gen": 0, "avg_ts": 1.0}]))


def test_vllm_latency_json_median():
    assert bench._latency_s({"avg_latency": 1.0, "percentiles": {"50": 0.9}}) == 0.9
    assert bench._latency_s({"avg_latency": 1.5}) == 1.5


def test_theory_subtracts_embedding_table_only(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results")
    rk = "q8__q4_k_m__tr__s0"
    dump = paths.tensor_types_csv(rk)
    dump.parent.mkdir(parents=True)
    pd.DataFrame([{"name": "token_embd.weight", "n_bytes": 100},
                  {"name": "output.weight", "n_bytes": 80},
                  {"name": "blk.0.attn_q.weight", "n_bytes": 820}]).to_csv(dump, index=False)
    sizes = theory._size_bytes(rk)
    assert sizes["weight_bytes"] == 1000 and sizes["embed_bytes"] == 100  # lm_head stays


def test_wait_healthy_fails_fast_when_server_dies():
    import subprocess
    import sys
    import time

    from scripts.e4 import bench

    proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    proc.wait()
    t0 = time.monotonic()
    try:
        bench._wait_healthy(59999, timeout=600, proc=proc)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "exited with code 3" in str(e)
    assert time.monotonic() - t0 < 5  # did not burn the timeout


def test_head_tensor_rows_untied_uses_output_only_tied_uses_embedding():
    from scripts.e6.lm_head import head_tensor_rows

    untied = [{"name": "token_embd.weight", "n_bytes": "100", "tensor_type": "Q4_K"},
              {"name": "output.weight", "n_bytes": "60", "tensor_type": "Q8_0"},
              {"name": "blk.0.attn_q.weight", "n_bytes": "5", "tensor_type": "Q4_K"}]
    assert [r["name"] for r in head_tensor_rows(untied)] == ["output.weight"]
    tied = [{"name": "token_embd.weight", "n_bytes": "100", "tensor_type": "Q8_0"},
            {"name": "blk.0.attn_q.weight", "n_bytes": "5", "tensor_type": "Q4_K"}]
    assert [r["name"] for r in head_tensor_rows(tied)] == ["token_embd.weight"]
