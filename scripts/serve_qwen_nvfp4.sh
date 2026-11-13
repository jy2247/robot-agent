#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

model_env="${MODEL_ENV:-/data/jingyu/H-PEEK/.venv}"
model_path="${MODEL_PATH:-/data/jingyu/Qwen3.8-27B-NVFP4}"
cache_root="${ROBOT_CACHE:-/data/jingyu/.cache/robot-agent}"

test -x "$model_env/bin/vllm"
test -s "$model_path/model.safetensors.index.json"
test -s "$model_path/hf_quant_config.json"
"$model_env/bin/python" scripts/check_nvfp4_runtime.py "$model_path"

mkdir -p "$cache_root/huggingface" "$cache_root/triton" "$cache_root/torch" \
  "$cache_root/cuda" "$cache_root/flashinfer" "$cache_root/vllm"
# Keep first-run compilation and FlashInfer JIT artifacts out of $HOME.  This
# matters on shared machines where the root filesystem is often space-limited.
export PATH="$model_env/bin:$PATH"
export HF_HOME="$cache_root/huggingface"
export TRITON_CACHE_DIR="$cache_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/torch"
export CUDA_CACHE_PATH="$cache_root/cuda"
export FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer"
export VLLM_CACHE_ROOT="$cache_root/vllm"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "$model_env/bin/vllm" serve "$model_path" \
  --served-model-name Qwen/Qwen3.8-27B \
  --host 127.0.0.1 \
  --port "${PORT:-8000}" \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 24576 \
  --max-num-batched-tokens 20480 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":24}' \
  --mm-processor-kwargs '{"min_pixels":65536,"max_pixels":802816,"do_resize":false}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}' \
  "$@"
