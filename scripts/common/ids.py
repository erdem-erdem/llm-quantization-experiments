"""Identifiers (plan §5.2)."""
from __future__ import annotations

MODELS = ["q06", "q17", "q4", "q8", "q14", "q32"]
HF_REPO = {
    "q06": "Qwen/Qwen3-0.6B",
    "q17": "Qwen/Qwen3-1.7B",
    "q4": "Qwen/Qwen3-4B",
    "q8": "Qwen/Qwen3-8B",
    "q14": "Qwen/Qwen3-14B",
    "q32": "Qwen/Qwen3-32B",
}
TIED = {"q06", "q17", "q4"}  # tie_word_embeddings=True (asserted at download)
QUANTS_E1 = ["fp8", "gptq", "q8_0", "q4_k_m", "q3_k_m", "q2_k"]
GGUF_QUANTS = {"q8_0", "q4_k_m", "q3_k_m", "q2_k"}
LLMC_QUANTS = {"fp8", "gptq", "rtn"}
CALIBS = ["none", "en", "tr", "mix"]
SEEDS = ["s0", "s1"]


def run_key(model: str, quant: str, calib: str, seed: str) -> str:
    return f"{model}__{quant}__{calib}__{seed}"


def parse_run_key(key: str) -> tuple[str, str, str, str]:
    m, q, c, s = key.split("__")
    return m, q, c, s


def seed_id(seed: str) -> int:
    """s0 -> 0, s1 -> 1."""
    return {"s0": 0, "s1": 1}[seed]


def seed_int(seed: str, offset: int = 0) -> int:
    """Numeric seed for rng/torch use."""
    return seed_id(seed) + offset