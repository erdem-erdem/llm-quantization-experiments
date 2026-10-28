"""parse_lm_eval: lm_eval's nested output layout + harness task names → vectors + RunRow."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import paths  # noqa: E402
from scripts.eval import parse_lm_eval  # noqa: E402


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results")
    return tmp_path


def _write_samples(exp: str, run_key: str, harness: str, n: int) -> None:
    # lm_eval: <output_path>/<sanitized model_args>/samples_<harness task>_<timestamp>.jsonl
    d = paths.raw_dir(exp, run_key) / "models__ct__q06__fp8__none__s0"
    d.mkdir(parents=True)
    rows = [{"doc_id": i, "acc": float(i % 2), "acc_norm": float(i % 3 == 0)} for i in range(n)]
    (d / f"samples_{harness}_2026-10-12T00-00-00.000000.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in reversed(rows)), encoding="utf-8")


def test_parse_run_writes_vector_and_row(sandbox):
    rk = "q06__fp8__none__s0"
    _write_samples("E1", rk, "belebele_tur_Latn", 900)
    parse_lm_eval.parse_run("E1", rk, ["belebele_tr"], "vllm",
                            {"size_bytes": 123, "bpw": 8.1})
    vec = [json.loads(l) for l in
           paths.sample_vector_path("E1", rk, "belebele_tr").read_text(encoding="utf-8").splitlines()]
    assert vec[0]["task"] == "belebele_tr" and vec[0]["n"] == 900
    assert [e["doc_id"] for e in vec[1:]] == list(range(900))  # sorted by doc_id
    row = json.loads(paths.runs_jsonl("E1").read_text(encoding="utf-8").splitlines()[0])
    assert row["metric"] == "acc" and row["n"] == 900 and row["size_bytes"] == 123
    assert abs(row["value"] - 0.5) < 1e-9
    # a rerun must not duplicate the runs.jsonl row
    parse_lm_eval.parse_run("E1", rk, ["belebele_tr"], "vllm", {})
    assert len(paths.runs_jsonl("E1").read_text(encoding="utf-8").splitlines()) == 1


def test_parse_run_limit_and_wrong_n(sandbox):
    rk = "q06__fp8__none__s0"
    _write_samples("E1", rk, "arc_tr", 50)
    with pytest.raises(RuntimeError, match="n=50"):
        parse_lm_eval.parse_run("E1", rk, ["arc_tr"], "vllm", {})
    parse_lm_eval.parse_run("E1", rk, ["arc_tr"], "vllm", {}, limit=50)  # smoke --limit 50
