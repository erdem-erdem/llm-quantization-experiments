"""YAML config loading."""
from __future__ import annotations

import functools
from typing import Any

import yaml

from . import paths


@functools.lru_cache(maxsize=None)
def load(name: str) -> dict[str, Any]:
    with open(paths.CONFIGS / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)