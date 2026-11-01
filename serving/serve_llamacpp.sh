#!/usr/bin/env bash
# The brain: llama.cpp llama-server, gpt-oss-20B MXFP4, OpenAI-compatible on :8000.
#
# gpt-oss-20B MXFP4 runs natively on this RX 6950 XT (gfx1030) via llama.cpp/ROCm:
# weights stay ~13 GB (no bf16 blowup) and it does ~90 t/s single-stream here.
# This is the serving engine for the project. (vLLM has no gfx1030 MXFP4 kernels;
# that path is abandoned.)
#
# Build llama.cpp once (HIP backend for gfx1030):
#   sudo pacman -S --needed rocm-hip-sdk cmake ninja git
#   git clone https://github.com/ggml-org/llama.cpp ~/src/llama.cpp && cd ~/src/llama.cpp
#   HIPCXX="$(hipconfig -l)/clang" cmake -B build -G Ninja \
#       -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
set -euo pipefail

LLAMA=${LLAMA:-$HOME/src/llama.cpp/build/bin/llama-server}
MODEL=${MODEL:-ggml-org/gpt-oss-20b-GGUF}   # HF repo (auto-downloaded + cached) or a local .gguf path
PARALLEL=${PARALLEL:-8}                      # concurrent slots == max concurrent agents
CTX_TOTAL=${CTX_TOTAL:-32768}                # total KV, split across slots: 32768/8 = 4K per agent
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

# -hf for a HF repo id, -m for a local file.
case "$MODEL" in
  */*.gguf|/*|./*) model_arg=(-m "$MODEL") ;;
  *)               model_arg=(-hf "$MODEL") ;;
esac

export HSA_OVERRIDE_GFX_VERSION=10.3.0        # no-op on native gfx1030; harmless

args=(
  "${model_arg[@]}"
  --alias gpt-oss-20b
  --host "$HOST" --port "$PORT"
  -ngl 99                   # all layers on the GPU
  -fa on                    # flash attention: required for quantized KV, big concurrency win
  --cont-batching           # continuous batching: new requests join the running batch
  --parallel "$PARALLEL"    # N slots; per-agent context = CTX_TOTAL / PARALLEL
  -c "$CTX_TOTAL"
  --kv-unified              # one shared KV pool across slots: short requests leave room for long ones
  --jinja                   # gpt-oss harmony chat template (reasoning channels + tool format)
  --reasoning-format auto   # reasoning goes to reasoning_content, kept out of the JSON body
  --metrics                 # Prometheus /metrics on this same port (see watch_metrics.sh)
)
exec "$LLAMA" "${args[@]}" "$@"
