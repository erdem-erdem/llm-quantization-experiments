#!/usr/bin/env bash
# Remote setup (docs/RUNBOOK.md step 2): Ubuntu 22.04, H100 80GB GPUs. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/../.."

sudo apt-get update
sudo apt-get install -y build-essential cmake git git-lfs python3-venv python3-dev tmux

# llama.cpp is built with CUDA (-DGGML_CUDA=ON): it needs the CUDA toolkit (nvcc), not just
# the driver. Stop here with a clear message instead of failing inside cmake.
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. Install the CUDA toolkit matching the driver (see" >&2
  echo "       https://developer.nvidia.com/cuda-downloads) and put nvcc on PATH, then re-run." >&2
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r env/requirements.base.txt

bash scripts/remote/build_llamacpp.sh

# HF token: either export HF_TOKEN beforehand or log in interactively.
# huggingface_hub >= 1.0 ships `hf` (huggingface-cli was removed); fall back for older hubs.
if command -v hf >/dev/null 2>&1; then HF_LOGIN=(hf auth login)
else HF_LOGIN=(huggingface-cli login); fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN not set; run: ${HF_LOGIN[*]}"
  "${HF_LOGIN[@]}" || true
else
  "${HF_LOGIN[@]}" --token "$HF_TOKEN" || true
fi

echo "setup_remote done — next: python scripts/check_env.py"