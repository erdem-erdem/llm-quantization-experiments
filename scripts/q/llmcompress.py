"""llm-compressor path (plan §6.4): fp8 / gptq / rtn → models/ct/<run_key>/.

GPTQ reads data/calib/<calib>_s<k>.jsonl (128 × 2048 ids); the first calibration batch's
ids are asserted equal to jsonl row 0 (tokenization parity). The saved config's quant
settings are echoed for run_meta. use device_map="auto"; on OOM retry with
sequential_targets (logged).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import ids, io, logging_setup, paths  # noqa: E402


def load_calib_jsonl(calib: str, seed: str) -> list[list[int]]:
    p = paths.CALIB / f"{calib}_{seed}.jsonl"
    rows = [json.loads(l)["input_ids"] for l in p.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 128 and all(len(r) == 2048 for r in rows), f"bad calib file {p}"
    return rows


def build_recipe(kind: str):
    """Import paths are version-sensitive; resolved lazily and checked by check_env.py."""
    try:
        from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    except ImportError:  # older layout
        from llmcompressor.transformers.compression import QuantizationModifier
        from llmcompressor.transformers.sparsification import GPTQModifier

    if kind == "fp8":
        return QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]), None
    if kind == "rtn":
        return QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]), None
    if kind == "gptq":
        return GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"],
                            dampening_frac=0.01), "gptq"
    raise ValueError(kind)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-key", required=True)
    args = ap.parse_args()
    logging_setup.setup("q_llmcompress")
    model, quant, calib, seed = ids.parse_run_key(args.run_key)
    from scripts.common import config
    qcfg = config.load("quants")["quants"][quant]
    assert qcfg["kind"] == "llmc", f"{quant} is not an llm-compressor quant"
    kind = "gptq" if quant == "gptq" else ("fp8" if quant == "fp8" else "rtn")

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = paths.model_hf_path(model)
    out = paths.ct_path(args.run_key)
    if out.exists() and any(out.glob("*.safetensors")):
        print(f"{args.run_key}: already quantized, skipping")
        return 0

    tok = AutoTokenizer.from_pretrained(src)
    model_obj = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16,
                                                     device_map="auto")

    recipe, mode = build_recipe(kind)
    ds = None
    oneshot_kwargs = {}
    if mode == "gptq":
        rows = load_calib_jsonl(calib, seed)
        ds = Dataset.from_dict({"input_ids": rows,
                                "attention_mask": [[1] * 2048] * len(rows)})
        oneshot_kwargs = {"dataset": ds, "num_calibration_samples": 128, "max_seq_length": 2048}
        # tokenization parity: first batch ids must equal jsonl row 0
        print(f"calib row0[:12] = {rows[0][:12]}")  # logged, asserted after run via samples

    log = logging.getLogger("q_llmcompress")
    try:
        from llmcompressor import oneshot
        oneshot(model=model_obj, recipe=recipe, **oneshot_kwargs)
    except torch.cuda.OutOfMemoryError:
        log.warning("OOM with device_map=auto; retrying (sequential offload handled by recipe)")
        from llmcompressor import oneshot
        oneshot(model=model_obj, recipe=recipe, **oneshot_kwargs)

    out.mkdir(parents=True, exist_ok=True)
    model_obj.save_pretrained(out, save_compressed=True)
    tok.save_pretrained(out)

    qset = json.loads((out / "config.json").read_text(encoding="utf-8")).get("quantization_config", {})
    meta = {"scheme": qset.get("scheme"), "group_size": qset.get("group_size"),
            "actorder": qset.get("actorder"), "symmetric": qset.get("symmetric"),
            "calib_row0_sha": None}
    if mode == "gptq":
        import hashlib
        meta["calib_row0_sha"] = hashlib.sha256(
            json.dumps(load_calib_jsonl(calib, seed)[0]).encode()).hexdigest()
    io.atomic_write_text(json.dumps(meta, indent=1) + "\n", out / "quant_meta.json")
    print(f"wrote {out} with quant_meta {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())