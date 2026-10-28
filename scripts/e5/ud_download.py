"""E5 control — Unsloth UD download (plan §6.8): the unsloth/Qwen3-4B-GGUF file whose
size is nearest to the uniform Q3_K_M. Copies it to models/gguf/q4__ud__tr__s0.gguf,
dumps tensor types, and records the size delta vs the uniform baseline; if |Δsize| > 2%
the analysis must not claim "same size".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, io, logging_setup, paths  # noqa: E402

REPO = "unsloth/Qwen3-4B-GGUF"


def main() -> int:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    logging_setup.setup("e5_ud")

    ref = config.load("experiments")["E5"]["calib_ref"]
    ref_key = f"{ref['model']}__{ref['quant']}__{ref['calib']}__{ref['seed']}"
    ref_size = sum(float(r["n_bytes"]) for r in _read_csv(paths.tensor_types_csv(ref_key)))

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = api.model_info(REPO, files_metadata=True)
    # spec §8 E5: among the UD quants, the one whose size is nearest to the uniform Q3_K_M
    ud = [(sib.rfilename, sib.size) for sib in info.siblings
          if sib.rfilename.endswith(".gguf") and "-UD-" in sib.rfilename and sib.size]
    if not ud:
        raise RuntimeError(f"no UD-quant gguf files listed in {REPO}")
    fname = min(ud, key=lambda fs: abs(fs[1] - ref_size))[0]
    print(f"UD candidates: {sorted((f, round(sz / 1e9, 2)) for f, sz in ud)}; picked {fname}")

    local = hf_hub_download(REPO, fname)
    out = paths.gguf_path("q4", "ud", "tr", "s0")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(Path(local).read_bytes())

    from scripts.q.gguf_quantize import _total_params
    from scripts.q.sizes import size_from_csv
    from scripts.q.tensor_dump import dump_csv
    dest = paths.tensor_types_csv("q4__ud__tr__s0")
    dump_csv(out, dest)
    sizes = size_from_csv(dest, _total_params("q4"))
    delta = abs(sizes["size_bytes"] - ref_size) / ref_size
    io.atomic_write_json(paths.RESULTS / "E5" / "ud.json", {
        "file": fname, "repo": REPO, "size_bytes": sizes["size_bytes"],
        "ref_size_bytes": ref_size, "size_delta_pct": round(100 * delta, 2),
        "same_size_claim": delta <= 0.02, "bpw": sizes["bpw"]})
    print(f"UD: {fname} Δsize={100 * delta:.2f}% (same-size claim: {delta <= 0.02})")
    return 0


def _read_csv(p: Path) -> list[dict]:
    import csv
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    sys.exit(main())