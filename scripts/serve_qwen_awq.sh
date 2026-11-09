#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

model_env="${MODEL_ENV:-/data/jingyu/H-PEEK/.venv}"
model_path="${MODEL_PATH:-/data/jingyu/Qwen3.8-27B-AWQ}"
cache_root="${ROBOT_CACHE:-/data/jingyu/.cache/robot-agent-awq}"
tp_size="${TP_SIZE:-2}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

test -x "$model_env/bin/vllm"
test -s "$model_path/model.safetensors.index.json"
"$model_env/bin/python" scripts/prepare_qwen_awq_checkpoint.py "$model_path"
"$model_env/bin/python" scripts/patch_vllm_qwen35_awq.py
"$model_env/bin/python" scripts/check_awq_runtime.py "$model_path" \
  --tensor-parallel-size "$tp_size"

mkdir -p "$cache_root/huggingface" "$cache_root/triton" "$cache_root/torch" \
  "$cache_root/cuda" "$cache_root/flashinfer"
export HF_HOME="$cache_root/huggingface"
export TRITON_CACHE_DIR="$cache_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/torch"
export CUDA_CACHE_PATH="$cache_root/cuda"
export FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "$model_env/bin/vllm" serve "$model_path" \
  --served-model-name Qwen/Qwen3.8-27B \
  --host 127.0.0.1 \
  --port "${PORT:-8000}" \
  --tensor-parallel-size "$tp_size" \
  --disable-custom-all-reduce \
  --enforce-eager \
  --quantization awq_marlin \
  --dtype half \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-8192}" \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.88}" \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --generation-config vllm \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt '{"image":16}' \
  --mm-processor-kwargs '{"min_pixels":65536,"max_pixels":802816,"do_resize":false}' \
  "$@"
