"""HF-side KL divergence (plan §6.6): reference vs test model on the same token windows.

kl_hf(ref_model, test_model, token_ids, ctx=2048, score_from=1024): non-overlapping ctx
windows; logits at all positions; score positions [score_from, ctx-1) (llama.cpp's
second-half rule); KL = Σ_v p_ref (log p_ref − log p_test) in fp32, mean over scored
positions; plus top-1 agreement. Reference log-probs cached on GPU as fp32
[n_windows, ctx, V]. Entry modes: --ref/--test dirs (E0 path check) and the
in-process API sweep_cell(model, cache, fn) for E5.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import io, logging_setup, paths  # noqa: E402


def _windows(token_ids: list[int], ctx: int) -> np.ndarray:
    n = len(token_ids) // ctx
    return np.array(token_ids[: n * ctx], dtype=np.int64).reshape(n, ctx)


class RefCache:
    """Cached fp32 reference log-probs for a fixed (ref_model, token_ids, ctx, score_from)."""

    def __init__(self, model, tok_ids: np.ndarray, ctx: int = 2048, 
                 score_from: int = 1024, device: str = "cuda"):
        self.device = device
        self.ctx = ctx
        self.score_from = score_from
        self.ids = torch.as_tensor(tok_ids, device=device)
        self.tok_ids = tok_ids
        self.logprobs = self._compute(model, self.ids)

    @torch.no_grad()
    def _compute(self, model, ids: torch.Tensor) -> torch.Tensor:
        n_windows = ids.shape[0]
        scored_positions = self.ctx - 1 - self.score_from
        V = model.config.vocab_size
        # Pre-allocate tensor for only scored positions
        lp_cached = torch.zeros((n_windows, scored_positions, V), 
                                dtype=torch.float32, device=self.device)
        for i, w in enumerate(ids):
            logits = model(input_ids=w[None, :]).logits[0].float()
            lp = torch.log_softmax(logits, dim=-1)  # [ctx, V]
            # Store only the scored positions
            lp_cached[i] = lp[self.score_from:self.ctx - 1]
        return lp_cached  # [n_win, scored_positions, V]


@torch.no_grad()
def kl_hf(ref_model, test_model, token_ids: list[int], ctx: int = 2048,
          score_from: int = 1024, device: str = "cuda") -> dict:
    ids = _windows(token_ids, ctx)
    cache = RefCache(ref_model, ids, ctx=ctx, score_from=score_from, device=device)
    return score_against(cache, test_model, ctx=ctx, score_from=score_from)


@torch.no_grad()
def score_against(cache: RefCache, test_model, ctx: int = 2048, score_from: int = 1024) -> dict:
    """KL of test_model against the cached reference on the scored positions."""
    ids = cache.ids.to(test_model.device if hasattr(test_model, "device") else cache.device)
    kls, tops = [], []
    for i, w in enumerate(ids):
        logits = test_model(input_ids=w[None, :]).logits[0].float()
        # llama.cpp scores positions [ctx/2, ctx-1): the last logit has no in-window target
        lp_test = torch.log_softmax(logits, dim=-1)[score_from:ctx - 1]     # [S, V]
        lp_ref = cache.logprobs[i].to(lp_test.device)   # [S, V] (already scored positions)
        kl = (lp_ref.exp() * (lp_ref - lp_test)).sum(-1)                # [S]
        kls.append(kl.cpu().numpy())  # per-token KL: p99 is over tokens like llama.cpp's 99.0%
        tops.append((lp_ref.argmax(-1) == lp_test.argmax(-1)).float().cpu().numpy())
    kls_all, tops_all = np.concatenate(kls), np.concatenate(tops)
    return {"mean_kld": float(kls_all.mean()), "p99_kld": float(np.percentile(kls_all, 99)),
            "same_top_p": 100.0 * float(tops_all.mean()),
            "n_windows": int(ids.shape[0]), "scored_positions": int(ctx - 1 - score_from)}


def sweep_cell(model, cache: RefCache, ctx: int = 2048, score_from: int = 1024) -> dict:
    """E5 in-process API: score a quantized (already loaded) model against a cached ref."""
    return score_against(cache, model, ctx=ctx, score_from=score_from)


def _load(path: Path):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                device_map="cuda:0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="HF dir of the bf16 reference")
    ap.add_argument("--test", required=True, help="HF dir of the dequantized model")
    ap.add_argument("--kl-text", default=None, help="default data/havadis_kl64k.txt")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--run-key", required=True)
    ap.add_argument("--cleanup", action="store_true",
                    help="delete models/tmp_deq/<run-key> afterwards (its eval kept it for this job)")
    args = ap.parse_args()
    logging_setup.setup("kl_hf")

    kl_text = Path(args.kl_text) if args.kl_text else paths.DATA / "havadis_kl64k.txt"
    from scripts.common.tokenize import load_qwen_tokenizer
    tok = load_qwen_tokenizer()
    token_ids = tok(kl_text.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]

    ref = _load(Path(args.ref))
    test = _load(Path(args.test))
    vals = kl_hf(ref, test, token_ids)

    from datetime import datetime
    from scripts.common import ids
    from scripts.common.run_meta import build_meta
    model, quant, calib, seed = ids.parse_run_key(args.run_key)
    meta = build_meta(exp=args.exp, job="kl_hf", run_key=args.run_key,
                      cmd=[args.ref, args.test], t0=datetime.now(), t1=datetime.now(),
                      returncode=0, n_gpus=1, seed=seed, extras={"metrics": vals})
    io.atomic_write_text(json.dumps(meta, indent=1) + "\n",
                         paths.run_meta_path(args.exp, args.run_key, "kl_hf"))
    from scripts.common.schemas import RunRow
    for metric in ("mean_kld", "p99_kld", "same_top_p"):
        io.jsonl_append(paths.runs_jsonl(args.exp), RunRow(
            exp=args.exp, model=model, quant=quant, calib=calib, seed=seed,
            task="kl_havadis", metric=metric, value=vals[metric], n=vals["n_windows"],
            engine="hf", run_id=f"{args.run_key}__kl_hf").model_dump())
    print(f"kl_hf {args.run_key}: {vals}")
    if args.cleanup:
        from scripts.q.dequant_validate import cleanup
        cleanup(args.run_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())