"""Static GPU assignment (one place): every emitter asks here instead of hard-coding ids.

Source of truth: configs/experiments.yaml `gpus` and `gpu_plan` (2 x 80 GB H100 profile:
GPU 0 = the big models, GPU 1 = everything else). A job sees exactly one GPU
(CUDA_VISIBLE_DEVICES), so per-GPU serialization in the scheduler means no two jobs ever share a
card; the plan only decides which card carries which model/role.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config  # noqa: E402


def n_gpus() -> int:
    return int(config.load("experiments")["gpus"])


def gpu_of_model(model: str) -> int:
    """GPU that carries a model's production/eval/KL chain."""
    return _checked(config.load("experiments")["gpu_plan"]["models"][model], f"model {model}")


def gpu_of_role(role: str) -> int:
    """GPU for a cross-model experiment role (e0, e2, e5, e6_kv, e6_head, demo)."""
    return _checked(config.load("experiments")["gpu_plan"]["roles"][role], f"role {role}")


def _checked(gpu: int, what: str) -> int:
    if not 0 <= int(gpu) < n_gpus():
        raise ValueError(f"gpu_plan assigns {what} to GPU {gpu}, but only {n_gpus()} GPUs exist")
    return int(gpu)
