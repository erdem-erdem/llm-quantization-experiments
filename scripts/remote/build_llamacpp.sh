#!/usr/bin/env bash
# Build llama.cpp at the pinned commit with CUDA (sm_90 for H100); install its gguf-py.
set -euo pipefail
cd "$(dirname "$0")/../.."

COMMIT=$(grep -v '^#' env/llamacpp_commit.txt | head -n1 | tr -d '[:space:]')
if [[ -z "$COMMIT" || "$COMMIT" == "PIN-ON-REMOTE" ]]; then
  # First setup on the remote: pin current master (requires re-running E0 if bumped later)
  COMMIT=$(git ls-remote https://github.com/ggml-org/llama.cpp HEAD | cut -f1)
  echo "$COMMIT" > env/llamacpp_commit.txt
  echo "pinned env/llamacpp_commit.txt to current master: $COMMIT"
fi
if [[ -d llama.cpp/.git ]]; then
  git -C llama.cpp fetch origin
  git -C llama.cpp checkout "$COMMIT"
else
  git clone https://github.com/ggml-org/llama.cpp llama.cpp
  git -C llama.cpp checkout "$COMMIT"
fi
git -C llama.cpp rev-parse HEAD > env/llamacpp_built_commit.txt

cmake -B llama.cpp/build -S llama.cpp -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DLLAMA_CURL=OFF
cmake --build llama.cpp/build --config Release -j"$(nproc)"

# the Python wrappers call llama.cpp/<binary>; cmake leaves them in build/bin
for exe in llama.cpp/build/bin/llama-*; do
  [[ -x "$exe" && -f "$exe" ]] && ln -sf "build/bin/$(basename "$exe")" "llama.cpp/$(basename "$exe")"
done

pip install ./llama.cpp/gguf-py
echo "llama.cpp built at $(cat env/llamacpp_built_commit.txt)"