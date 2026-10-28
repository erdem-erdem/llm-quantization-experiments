"""Pydantic schemas for result rows, sample vectors and run metadata (plan §5.3)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class RunRow(BaseModel):
    """One line of results/<exp>/runs.jsonl."""
    exp: str
    model: str
    quant: str
    calib: str
    seed: str
    task: str
    metric: str
    value: float
    delta: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    flip_c2w: float | None = None
    flip_w2c: float | None = None
    n: int
    bpw: float | None = None
    bpw_body: float | None = None
    size_bytes: int | None = None
    engine: str
    run_id: str
    chance: bool | None = None
    reason: str | None = None   # present when value is NA / not measurable


class SampleHeader(BaseModel):
    """First line of a sample vector file."""
    task: str
    n: int
    metric: str
    dataset_sha: str
    run_key: str


class SampleEntry(BaseModel):
    doc_id: int
    correct: Literal[0, 1]


class RunMeta(BaseModel):
    """results/<exp>/run_meta/<run_key>__<job>.json"""
    exp: str
    job: str
    run_key: str
    git_commit: str
    pip_freeze_sha256: str
    llamacpp_commit: str | None = None
    gpu_name: str | None = None
    gpu_memory_total: str | None = None
    gpu_power_limit: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    seed: str | None = None
    cmd: list[str]
    start: str
    end: str
    gpu_minutes: float
    returncode: int
    status: Literal["ok", "fail"]
    extras: dict[str, Any] = Field(default_factory=dict)

    @field_validator("returncode")
    @classmethod
    def rc_binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"returncode must be 0 or 1, got {v}")
        return v


def validate_runs_row(obj: dict[str, Any]) -> RunRow:
    return RunRow.model_validate(obj)


# Allowed values, for cross-checking rows against configs.
VALID_TASK_METRICS = {
    "belebele_tr": ("acc",),
    "belebele_en": ("acc",),
    "arc_tr": ("acc_norm",),
    "arc_en": ("acc_norm",),
    "gpqa_tr": ("acc",),
    "kl_havadis": ("mean_kld", "p99_kld", "same_top_p", "ppl"),
    "tr_arc_avg": ("delta",),
    "en_avg": ("delta",),
}