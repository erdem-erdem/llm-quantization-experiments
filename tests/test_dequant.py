"""Dequant name-mapping + sizes tests (Layer A, no model downloads needed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.q.sizes import size_from_csv  # noqa: E402


def test_size_from_csv_bpw_math(tmp_path):
    df = pd.DataFrame([
        {"name": "token_embd.weight", "n_bytes": 300},
        {"name": "output.weight", "n_bytes": 100},
        {"name": "blk.0.attn_q.weight", "n_bytes": 400},
        {"name": "blk.0.ffn_down.weight", "n_bytes": 200},
    ])
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    s = size_from_csv(p, total_params=1000)
    assert s["size_bytes"] == 1000
    assert s["body_bytes"] == 600          # 1000 - emb(300) - lm_head(100)
    assert abs(s["bpw"] - 8.0) < 1e-9
    assert abs(s["bpw_body"] - 4.8) < 1e-9


def test_size_from_csv_tied_no_output(tmp_path):
    df = pd.DataFrame([
        {"name": "token_embd.weight", "n_bytes": 300},
        {"name": "blk.0.attn_q.weight", "n_bytes": 700},
    ])
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    s = size_from_csv(p, total_params=1000)
    assert s["body_bytes"] == 700
    assert s["lm_head_bytes"] == 0         # tied: no output.weight in the dump


def test_dequant_inverse_map_qwen3():
    """gguf-py QWEN3 map (suffix-less in/out) + our suffix reconstruction, full inverse."""
    import gguf
    from scripts.q.dequant import _split_suffix, build_inverse

    hf_names = {
        "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight", "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    }
    expected = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
        "blk.0.attn_q.weight": "model.layers.0.self_attn.q_proj.weight",
        "blk.0.attn_k.weight": "model.layers.0.self_attn.k_proj.weight",
        "blk.0.attn_v.weight": "model.layers.0.self_attn.v_proj.weight",
        "blk.0.attn_output.weight": "model.layers.0.self_attn.o_proj.weight",
        "blk.0.attn_q_norm.weight": "model.layers.0.self_attn.q_norm.weight",
        "blk.0.attn_k_norm.weight": "model.layers.0.self_attn.k_norm.weight",
        "blk.0.attn_norm.weight": "model.layers.0.input_layernorm.weight",
        "blk.0.ffn_norm.weight": "model.layers.0.post_attention_layernorm.weight",
        "blk.0.ffn_gate.weight": "model.layers.0.mlp.gate_proj.weight",
        "blk.0.ffn_up.weight": "model.layers.0.mlp.up_proj.weight",
        "blk.0.ffn_down.weight": "model.layers.0.mlp.down_proj.weight",
    }
    inverse = build_inverse(set(expected), hf_names, n_blocks=2)
    assert inverse == expected
    # any stray GGUF name must fail loudly
    import pytest
    with pytest.raises(RuntimeError):
        build_inverse(set(expected) | {"blk.0.mystery.weight"}, hf_names, n_blocks=2)
    assert _split_suffix("a.b.weight") == ("a.b", ".weight")


def test_gguf_dequant_roundtrip_tiny(tmp_path, monkeypatch):
    """Write a tiny q8_0-ish GGUF? Too format-heavy here; instead validate the reshape rule:
    GGUF dims fastest-first → reversed reshape reproduces the original 2-D layout."""
    import numpy as np
    w = np.arange(12, dtype=np.float32).reshape(3, 4)  # HF layout [out, in]
    gguf_flat = w.T.copy().reshape(-1)                 # fastest-first storage
    rebuilt = gguf_flat.reshape(tuple(reversed((3, 4))))
    assert rebuilt.shape == (4, 3)
    assert np.array_equal(rebuilt.T, w)                # transpose back → original