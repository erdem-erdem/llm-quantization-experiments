"""Path constants. All path logic lives here; nothing else builds paths by string concat."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root

DATA = ROOT / "data"
CALIB = DATA / "calib"
OWN = DATA / "own"
MODELS = ROOT / "models"
MODELS_HF = MODELS / "hf"
MODELS_GGUF = MODELS / "gguf"
MODELS_CT = MODELS / "ct"
MODELS_TMP_DEQ = MODELS / "tmp_deq"
TASKS = ROOT / "tasks"
REFS = ROOT / "refs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"
JOBS = LOGS / "jobs"
CONFIGS = ROOT / "configs"
ENV = ROOT / "env"


def run_dir(exp: str) -> Path:
    return RESULTS / exp


def samples_dir(exp: str) -> Path:
    return run_dir(exp) / "samples"


def raw_dir(exp: str, run_key: str) -> Path:
    return run_dir(exp) / "raw" / run_key


def run_meta_path(exp: str, run_key: str, job: str) -> Path:
    return run_dir(exp) / "run_meta" / f"{run_key}__{job}.json"


def model_hf_path(model: str) -> Path:
    return MODELS_HF / model


def gguf_path(model: str, quant: str, calib: str, seed: str) -> Path:
    return MODELS_GGUF / f"{model}__{quant}__{calib}__{seed}.gguf"


def tmp_deq_path(run_key: str) -> Path:
    return MODELS_TMP_DEQ / run_key


def ct_path(run_key: str) -> Path:
    return MODELS_CT / run_key


def runs_jsonl(exp: str) -> Path:
    return run_dir(exp) / "runs.jsonl"


def sample_vector_path(exp: str, run_key: str, task: str) -> Path:
    return samples_dir(exp) / f"{run_key}__{task}.jsonl"


def done_marker(exp: str, job: str, key: str) -> Path:
    return JOBS / f"{exp}__{job}__{key}.done"


def tensor_types_csv(run_key: str) -> Path:
    return RESULTS / "tensor_types" / f"{run_key}.csv"


def master_csv() -> Path:
    return RESULTS / "master.csv"


def budget_csv() -> Path:
    return LOGS / "gpu_budget.csv"