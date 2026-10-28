"""T0: BF16 GGUF conversion — convert_hf_to_gguf.py per model, largest first, CPU-only.

Usage: python scripts/t0/gguf_convert.py --all | --models q8,q4
Records the llama.cpp commit into the log; the llama.cpp root is resolved from
env/llamacpp_root.txt (one path line) or --llamacpp.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import ids, logging_setup, paths  # noqa: E402
from scripts.common.run_meta import llamacpp_commit  # noqa: E402
from scripts.common.shell import run  # noqa: E402

SIZE_ORDER = ["q32", "q14", "q8", "q4", "q17", "q06"]  # largest first (CPU, background)


def convert(model: str, llamacpp: Path) -> Path:
    src = paths.model_hf_path(model)
    if not (src / "config.json").exists():
        raise FileNotFoundError(f"{src}: run scripts/t0/download_models.py first")
    out = paths.MODELS_GGUF / f"{model}-bf16.gguf"
    if out.exists():
        print(f"{model}: {out.name} already exists, skipping")
        return out
    paths.MODELS_GGUF.mkdir(parents=True, exist_ok=True)
    script = llamacpp / "convert_hf_to_gguf.py"
    if not script.exists():
        raise FileNotFoundError(f"{script}: set --llamacpp or env/llamacpp_root.txt")
    log = paths.LOGS / f"gguf_convert_{model}.log"
    run([sys.executable, str(script), str(src), "--outtype", "bf16", "--outfile", str(out)],
        log_path=log, check=True)
    print(f"{model}: wrote {out} (llama.cpp commit {llamacpp_commit()})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--llamacpp", default="", help="path to llama.cpp checkout")
    args = ap.parse_args()
    logging_setup.setup("t0_gguf_convert")

    root_file = paths.ENV / "llamacpp_root.txt"
    if args.llamacpp:
        llamacpp = Path(args.llamacpp)
    elif root_file.exists():
        llamacpp = Path(root_file.read_text(encoding="utf-8").strip())
    else:
        llamacpp = Path("llama.cpp")

    models = SIZE_ORDER if args.all else [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        ap.error("pass --all or --models q8,q4")
    for m in models:
        if m not in ids.MODELS:
            raise ValueError(f"unknown model {m}")
        convert(m, llamacpp)
    return 0


if __name__ == "__main__":
    sys.exit(main())