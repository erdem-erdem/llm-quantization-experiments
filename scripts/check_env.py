"""Environment checker (plan §6.2): runs on the remote, prints a table, appends findings
to NOTES.md under '## CLI deviations' and writes configs/cli_overrides.json for any
missing llama.cpp flags. Every check degrades to a recorded deviation, never a crash —
but the summary says PASS/WARN/FAIL per line and the exit code is 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date
from importlib.metadata import version as pkg_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import config, logging_setup, paths  # noqa: E402

PKGS = ["torch", "vllm", "lm_eval", "llmcompressor", "gguf", "transformers",
        "datasets", "huggingface_hub", "numpy", "pandas", "pydantic"]
LLAMA_FLAGS = {
    "llama-perplexity": ["--kl-divergence-base", "--kl-divergence"],
    "llama-quantize": ["--imatrix", "--tensor-type", "--output-tensor-type"],
    "llama-imatrix": ["--chunks", "-c", "-ngl"],
    "llama-server": ["--jinja", "-ctk", "-ctv", "-fa"],
    "llama-bench": ["-fa", "-p", "-n", "-r", "-o", "-d", "-ctk", "-ctv"],
}
GPU_NAME_HINT = "H100"
MIN_GPU_MIB = 70_000   # q32 BF16 (~65 GB) + KV must fit one card: 80 GB class only


def _run(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, str(e)


def check_gpus(rows: list[tuple[str, str, str]]) -> bool:
    rc, out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader"])
    if rc != 0:
        rows.append(("GPUs", "FAIL", f"nvidia-smi failed: {out.strip()[:100]}"))
        return False
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    names = [l.split(",")[0].strip() for l in lines]
    drivers = {l.split(",")[1].strip() for l in lines if "," in l}
    expected = int(config.load("experiments")["gpus"])  # the queue's gpu_plan size (2)
    mem = []
    for l in lines:
        parts = [p.strip() for p in l.split(",")]
        try:
            mem.append(int(parts[2].split()[0]))
        except (IndexError, ValueError):
            mem.append(0)
    ok = (len(lines) >= expected and all(GPU_NAME_HINT in n for n in names)
          and all(m >= MIN_GPU_MIB for m in mem))
    rows.append(("GPUs", "PASS" if ok else "FAIL",
                 f"{len(lines)}x {names[0] if names else '?'} ({min(mem, default=0)} MiB min), "
                 f"driver {drivers or '?'}; plan needs {expected} x 80 GB H100"))
    return ok


def check_disk(rows: list[tuple[str, str, str]]) -> bool:
    free = shutil.disk_usage(paths.ROOT).free
    tb = free / 2**40
    ok = tb >= 1.0
    rows.append(("Disk free", "PASS" if ok else "WARN",
                 f"{tb:.2f} TiB (warn < 1.0, expect >= 1.2)"))
    return True


def check_python(rows: list[tuple[str, str, str]]) -> bool:
    rows.append(("Python", "PASS", sys.version.split()[0]))
    ok = True
    for pkg in PKGS:
        try:
            rows.append((pkg, "PASS", pkg_version(pkg)))
        except Exception:
            rows.append((pkg, "FAIL" if pkg in ("torch", "vllm", "lm_eval", "gguf")
                         else "WARN", "not installed"))
            ok = False
    return ok


def check_llamacpp(llamacpp: str, rows: list[tuple[str, str, str]]) -> bool:
    ok = True
    want = next((l.strip() for l in
                 Path("env/llamacpp_commit.txt").read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")), "")
    rc, out = _run(["git", "-C", llamacpp, "rev-parse", "HEAD"])
    got = out.strip() if rc == 0 else "?"
    ok &= got == want
    rows.append(("llama.cpp commit", "PASS" if ok else "FAIL",
                 f"{got[:12]} vs pinned {want[:12]}"))
    rc, out = _run([sys.executable, "-c", "import gguf; print(gguf.__version__)"])
    ver = out.strip().splitlines()[-1] if (rc == 0 and out.strip()) else ""
    rows.append(("gguf-py (pip)", "PASS" if rc == 0 and ver else "WARN",
                 ver[:40] if ver else "import failed"))
    return ok


def check_cli_flags(llamacpp: str, rows: list[tuple[str, str, str]]) -> dict:
    """--help scans; missing flags recorded in cli_overrides.json (wrappers fail loudly)."""
    overrides: dict[str, list[str]] = {}
    for binary, flags in LLAMA_FLAGS.items():
        exe = Path(llamacpp) / "build" / "bin" / binary
        if not exe.exists():
            exe = Path(llamacpp) / binary
        rc, out = _run([str(exe), "--help"], timeout=30)
        if rc != 0:
            rows.append((binary, "FAIL", "binary not runnable"))
            overrides[binary] = flags
            continue
        missing = [f for f in flags if f not in out]
        if missing:
            overrides[binary] = missing
            rows.append((binary, "WARN", f"missing flags: {missing} (recorded)"))
        else:
            rows.append((binary, "PASS", f"all {len(flags)} flags present"))
    for sub in ("latency", "serve"):
        rc, out = _run(["vllm", "bench", sub, "--help"], timeout=120)
        rows.append((f"vllm bench {sub}", "PASS" if rc == 0 else "FAIL",
                     "subcommand ok" if rc == 0 else "unavailable"))
    # lm-eval >= 0.4.10 has subcommands: `--model ... vllm` is only documented under `run --help`
    rc, out = _run([sys.executable, "-m", "lm_eval", "run", "--help"], timeout=120)
    if rc != 0:  # older lm-eval has no subcommands
        rc, out = _run([sys.executable, "-m", "lm_eval", "--help"], timeout=120)
    ok_lm = "--model" in out and "vllm" in out
    rows.append(("lm_eval --model vllm", "PASS" if ok_lm else "FAIL",
                 "backend listed" if ok_lm else "vllm backend missing"))
    if not ok_lm:
        overrides["lm_eval"] = ["--model", "vllm"]
    rc, out = _run(["hf", "auth", "whoami"], timeout=30)
    if rc != 0:  # older huggingface_hub ships the CLI as a module, not the `hf` script
        rc, out = _run([sys.executable, "-m",
                        "huggingface_hub.commands.huggingface_cli", "auth", "whoami"],
                       timeout=30)
    rows.append(("hf auth", "PASS" if rc == 0 else "WARN", out.strip().splitlines()[0]
                 if out.strip() else "not logged in"))
    return overrides


def check_nvcc(rows: list[tuple[str, str, str]]) -> None:
    rc, out = _run(["nvcc", "--version"])
    line = [l for l in out.splitlines() if "release" in l.lower()]
    rows.append(("nvcc", "PASS" if rc == 0 else "WARN",
                 line[0].strip()[-20:] if line else "not found (llama.cpp build needs it)"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llamacpp", default="llama.cpp")
    ap.add_argument("--no-notes", action="store_true", help="do not append to NOTES.md")
    args = ap.parse_args()
    logging_setup.setup("check_env")

    rows: list[tuple[str, str, str]] = []
    ok = check_gpus(rows)
    check_nvcc(rows)
    ok &= check_python(rows)
    ok &= check_llamacpp(args.llamacpp, rows)
    check_disk(rows)
    overrides = check_cli_flags(args.llamacpp, rows)

    print(f"{'check':22s} {'status':6s} detail")
    for name, status, detail in rows:
        print(f"{name:22s} {status:6s} {detail}")
    paths.CONFIGS.mkdir(exist_ok=True)
    (paths.CONFIGS / "cli_overrides.json").write_text(
        json.dumps({"generated": str(date.today()), "missing_flags": overrides},
                   indent=1), encoding="utf-8")
    if not args.no_notes:
        with open(paths.ROOT / "NOTES.md", "a", encoding="utf-8") as f:
            f.write(f"\n## CLI deviations (check_env {date.today()})\n\n")
            if overrides:
                for tool, flags in overrides.items():
                    f.write(f"- {tool}: missing {flags} — adapt parsers before running\n")
            else:
                f.write("- none: every asserted CLI flag present\n")
    n_fail = sum(1 for _, s, _ in rows if s == "FAIL")
    notes_note = "+ NOTES.md" if not args.no_notes else ""
    print(f"\ncheck_env: {n_fail} FAIL, exit {1 if n_fail else 0} "
          f"(deviations recorded in configs/cli_overrides.json {notes_note})")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())