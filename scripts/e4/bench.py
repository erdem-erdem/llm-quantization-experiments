"""E4 benchmark runner (plan §6.8): one measurement cell per invocation.

Cells (each writes RunRows to results/E4/runs.jsonl + raw log):
  --llama-bench --run-key <rk>   llama-bench -fa 1 -p 512 -n 256 -r 3 -o json (GGUF/bf16-GGUF)
  --vllm-latency --run-key <rk>  vllm bench latency ×2 (output-len 1 → TTFT; 256 → decode)
  --vllm-serve --model <path> --concs 1,4,16,64   starts vllm serve, bench serve per concurrency
Every cell logs nvidia-smi dmon to results/E4/raw/ and requires an idle GPU (record_run
enforces exclusivity for job name "speed").
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import config, ids, io, logging_setup, paths  # noqa: E402
from scripts.common.schemas import RunRow  # noqa: E402


def _dmon(exp: str, run_id: str):
    """Start nvidia-smi dmon in the background; returns a stop() callable."""
    raw = paths.raw_dir(exp, run_id)
    raw.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["nvidia-smi", "dmon", "-s", "pcu", "-d", "1"],
                            stdout=open(raw / "dmon.log", "w", encoding="utf-8"),
                            stderr=subprocess.DEVNULL)
    return lambda: proc.terminate()


def _emit(exp: str, run_key: str, task: str, metric: str, value: float,
          engine: str, run_id: str, extras_note: str = "") -> None:
    model, quant, calib, seed = ids.parse_run_key(run_key)
    io.jsonl_append(paths.runs_jsonl(exp), RunRow(
        exp=exp, model=model, quant=quant, calib=calib, seed=seed,
        task=task, metric=metric, value=value, n=1, engine=engine,
        run_id=run_id, reason=extras_note or None).model_dump())


def parse_llama_bench(stdout: str) -> tuple[float, float]:
    """(prefill, decode) tok/s from `llama-bench -o json`: one object per test with n_prompt,
    n_gen and avg_ts (pp = n_prompt>0 & n_gen==0, tg = n_gen>0 & n_prompt==0)."""
    cells = json.loads(stdout)
    pp = [float(c["avg_ts"]) for c in cells if c["n_prompt"] > 0 and c["n_gen"] == 0]
    tg = [float(c["avg_ts"]) for c in cells if c["n_gen"] > 0 and c["n_prompt"] == 0]
    if not pp or not tg:
        raise RuntimeError(f"llama-bench JSON lacks a pp or tg test: "
                           f"{[(c.get('n_prompt'), c.get('n_gen')) for c in cells]}")
    return pp[0], tg[0]


def llama_cell(exp: str, run_key: str, llamacpp: str) -> None:
    model, quant, calib, seed = ids.parse_run_key(run_key)
    e4 = config.load("experiments")["E4"]
    gguf = paths.MODELS_GGUF / f"{model}-bf16.gguf" if quant == "bf16" \
        else paths.gguf_path(model, quant, calib, seed)
    if not gguf.exists():
        raise FileNotFoundError(gguf)
    run_id = f"{run_key}__llama"
    cmd = [str(Path(llamacpp) / "llama-bench"), "-m", str(gguf), "-ngl", "99", "-fa", "1",
           "-p", str(e4["prefill"]), "-n", str(e4["decode"]), "-r", str(e4["repeats"]),
           "-o", "json"]
    stop = _dmon(exp, run_id)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600)
    finally:
        stop()
    (paths.raw_dir(exp, run_id) / "llama_bench.json").write_text(out.stdout, encoding="utf-8")
    prefill, decode = parse_llama_bench(out.stdout)
    _emit(exp, run_key, "speed", "prefill_tok_s", prefill, "llama.cpp", run_id)
    _emit(exp, run_key, "speed", "decode_tok_s", decode, "llama.cpp", run_id)
    print(f"llama-bench {run_key}: prefill {prefill:.1f} tok/s, decode {decode:.1f} tok/s")


def vllm_cell(exp: str, run_key: str, model_path: Path) -> None:
    """Two latency runs: output-len 1 → TTFT; output-len 256 → total latency."""
    e4 = config.load("experiments")["E4"]
    run_id = f"{run_key}__vllm"
    res = {}
    for tag, out_len in (("ttft", 1), ("lat", e4["decode"])):
        jout = paths.raw_dir(exp, run_id) / f"latency_{tag}.json"
        cmd = ["vllm", "bench", "latency", "--model", str(model_path),
               "--input-len", str(e4["prefill"]), "--output-len", str(out_len),
               "--batch-size", "1", "--num-iters", str(e4["repeats"]),
               "--output-json", str(jout)]
        stop = _dmon(exp, run_id)
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600)
        finally:
            stop()
        data = json.loads(jout.read_text(encoding="utf-8"))
        res[tag] = data
    ttft_s = _latency_s(res["ttft"])  # output-len 1: latency = prefill + first token
    lat_s = _latency_s(res["lat"])
    decode_s = lat_s - ttft_s
    decode_tok_s = (e4["decode"] - 1) / decode_s
    for metric, v in (("ttft_s", ttft_s), ("latency_s", lat_s),
                      ("decode_tok_s", decode_tok_s)):
        _emit(exp, run_key, "speed", metric, v, "vllm", run_id)
    print(f"vllm {run_key}: ttft {ttft_s:.3f}s, decode {decode_tok_s:.1f} tok/s")


def _latency_s(d: dict) -> float:
    """Median latency (s) from `vllm bench latency --output-json`:
    {"avg_latency", "latencies", "percentiles": {"10", "25", "50", ...}}."""
    p50 = d.get("percentiles", {}).get("50")
    return float(p50) if p50 is not None else _pick(d, ("avg_latency",))


def _pick(d: dict, keys: tuple[str, ...]) -> float:
    for k in keys:
        if k in d:
            return float(d[k])
    raise RuntimeError(f"none of {keys} in bench JSON keys {sorted(d)}")


def serve_cell(exp: str, model_path: Path, engine_tag: str, concs: list[int],
               port: int) -> None:
    """32B concurrency sweep: vllm serve + vllm bench serve per concurrency.

    A dead server (OOM, bad checkpoint) writes one NA row with a reason instead of
    crashing the job; the server's output goes to logs/jobs/ for diagnosis."""
    e4 = config.load("experiments")["E4"]
    # the served quant, as in the sp: job ids (gptq calibrated on tr, bf16/fp8 uncalibrated)
    run_key = f"q32__{engine_tag}__{'tr' if engine_tag == 'gptq' else 'none'}__s0"
    log_path = paths.LOGS / "jobs" / f"E4__serve_{engine_tag}__server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        server = subprocess.Popen(
            ["vllm", "serve", str(model_path), "--port", str(port), "--max-model-len", "2048"],
            stdout=logf, stderr=subprocess.STDOUT)
        try:
            try:
                _wait_healthy(port, timeout=1800, proc=server)
            except RuntimeError as e:
                _emit(exp, run_key, "serve", "output_tok_s", float("nan"), engine_tag,
                      f"serve__{engine_tag}__c{concs[0]}",
                      extras_note=f"vllm serve failed to start: {e} (see {log_path.name})")
                print(f"serve {engine_tag}: NA row written ({e})")
                return
            for c in concs:
                run_id = f"serve__{engine_tag}__c{c}"
                rdir = paths.raw_dir(exp, run_id)
                jout = rdir / "bench_serve.json"
                # `vllm bench serve` has no --output-json; results go through --save-result
                cmd = ["vllm", "bench", "serve", "--model", str(model_path),
                       "--port", str(port), "--dataset-name", "random",
                       "--max-concurrency", str(c), "--num-prompts", str(4 * c),
                       "--random-input-len", str(e4["prefill"]),
                       "--random-output-len", str(e4["decode"]),
                       "--save-result", "--result-dir", str(rdir),
                       "--result-filename", jout.name]
                stop = _dmon(exp, run_id)
                try:
                    subprocess.run(cmd, capture_output=True, text=True, check=True,
                                   timeout=3600)
                finally:
                    stop()
                data = json.loads(jout.read_text(encoding="utf-8"))
                _emit(exp, run_key, "serve", "output_tok_s",
                      float(data["output_throughput"]), engine_tag, run_id)
                _emit(exp, run_key, "serve", "ttft_s",
                      _pick(data, ("median_ttft_ms", "ttft_median_ms")) / 1000.0,
                      engine_tag, run_id)
                _emit(exp, run_key, "serve", "tpot_s",
                      float(data["median_tpot_ms"]) / 1000.0, engine_tag, run_id)
                print(f"serve {engine_tag} c={c}: {data['output_throughput']:.1f} tok/s out")
        finally:
            server.terminate()
            server.wait(timeout=60)


def _wait_healthy(port: int, timeout: float, proc: subprocess.Popen | None = None) -> None:
    """Poll /health; fail fast when the server process has exited."""
    import urllib.request
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(10)
    raise RuntimeError(f"vllm serve not healthy after {timeout}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-bench", metavar="RUN_KEY")
    ap.add_argument("--vllm-latency", metavar="RUN_KEY")
    ap.add_argument("--vllm-serve", metavar="ENGINE",
                    help="one of bf16/fp8/gptq for the q32 sweep")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--concs", default="1,4,16,64")
    ap.add_argument("--port", type=int, default=8465)
    ap.add_argument("--exp", default="E4")
    ap.add_argument("--llamacpp", default="llama.cpp")
    args = ap.parse_args()
    logging_setup.setup("e4_bench")

    if args.llama_bench:
        llama_cell(args.exp, args.llama_bench, args.llamacpp)
    elif args.vllm_latency:
        vllm_cell(args.exp, args.vllm_latency, Path(args.model_path))
    elif args.vllm_serve:
        serve_cell(args.exp, Path(args.model_path), args.vllm_serve,
                   [int(c) for c in args.concs.split(",")], args.port)
    else:
        ap.error("choose --llama-bench, --vllm-latency or --vllm-serve")
    return 0


if __name__ == "__main__":
    sys.exit(main())