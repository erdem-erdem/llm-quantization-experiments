"""Shared llama-server orchestration for E6 + demos (plan §6.8).

start_server(): llama-server with CUDA layers, flash-attn on, --jinja; waits for /health.
complete(): POST /v1/chat/completions with temperature=0 and enable_thinking=False
(chat_template_kwargs, /no_think fallback). Server is killed on context exit.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from scripts.common import paths

NEXT_PORT = [8460]


def _free_port(gpu_id: int | None = None) -> int:
    """Derive port from GPU ID or find a free port."""
    import socket
    if gpu_id is not None:
        # Port 8000 + 100*gpu_id to reduce collisions
        port = 8000 + gpu_id * 100
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        if result != 0:
            return port
    # Fallback to auto-increment if specified port is busy
    while True:
        NEXT_PORT[0] += 1
        port = NEXT_PORT[0]
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        if result != 0:
            return port


class LlamaServer:
    def __init__(self, gguf: Path, ctx: int, *, cache_types: tuple[str, str] | None = None,
                 llamacpp: str = "llama.cpp", extra_args: list[str] | None = None,
                 port: int | None = None, gpu_id: int | None = None):
        import os as _os
        if gpu_id is None:
            # Try to get GPU ID from CUDA_VISIBLE_DEVICES
            cuda_devices = _os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
            if cuda_devices[0].strip().isdigit():
                gpu_id = int(cuda_devices[0].strip())
        self.port = port or _free_port(gpu_id)
        self.gpu_id = gpu_id
        self.cmd = [str(Path(llamacpp) / "llama-server"), "-m", str(gguf),
                    "-c", str(ctx), "-ngl", "99", "-fa", "on", "-np", "1", "--jinja",
                    "--port", str(self.port)]
        if cache_types:
            self.cmd += ["--cache-type-k", cache_types[0], "--cache-type-v", cache_types[1]]
        if extra_args:
            self.cmd += extra_args
        self.proc: subprocess.Popen | None = None
        self.logfile = paths.LOGS / f"llama_server_p{self.port}.log"

    def __enter__(self) -> "LlamaServer":
        self.logfile.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.logfile, "w", encoding="utf-8")
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        _wait_healthy(self.port, timeout=900, proc=self.proc)
        return self

    def __exit__(self, *exc) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()

    def tokenize(self, text: str) -> list[int]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/tokenize",
            data=json.dumps({"content": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode("utf-8"))["tokens"]

    def complete(self, prompt: str, max_tokens: int, *, system: str | None = None) -> str:
        body: dict = {"messages": ([{"role": "system", "content": system}] if system else [])
                      + [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens, "temperature": 0,
                      "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    out = json.loads(r.read().decode("utf-8"))
                return out["choices"][0]["message"]["content"]
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
        raise RuntimeError("unreachable")


def _wait_healthy(port: int, timeout: float, proc: subprocess.Popen | None = None) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        # Check if process died
        if proc and proc.poll() is not None:
            raise RuntimeError(f"llama-server process died (exit code {proc.returncode})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"llama-server not healthy after {timeout}s")


def kv_cache_mib_from_log(logfile: Path) -> float | None:
    """'llama_kv_cache: size = X MiB' from the server log (plan §6.8)."""
    text = logfile.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"llama_kv_cache:.*?size\s*=\s*([0-9.]+)\s*MiB", text)
    return float(m.group(1)) if m else None


def peak_gpu_mib(func, gpu_id: int | None = None) -> float:
    """Poll nvidia-smi for a specific GPU while func() runs; return peak used MiB."""
    import subprocess as sp
    import os as _os
    peak = 0.0
    # Determine GPU ID if not provided
    if gpu_id is None:
        cuda_devices = _os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        if cuda_devices[0].strip().isdigit():
            gpu_id = int(cuda_devices[0].strip())
        else:
            gpu_id = 0
    # Query only the specified GPU
    cmd = ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=memory.used",
           "--format=csv,noheader,nounits", "-l", "1"]
    proc = sp.Popen(cmd, stdout=sp.PIPE, text=True)
    try:
        try:
            func()
        finally:
            proc.terminate()
        for line in proc.stdout:
            if line.strip().isdigit():
                peak = max(peak, float(line.strip()))
    finally:
        proc.terminate()
    return peak