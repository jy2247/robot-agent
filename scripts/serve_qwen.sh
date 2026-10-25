#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tmp .cache/triton .cache/torch .cache/cuda .cache/xdg .cache/flashinfer
export PATH="$PWD/.venv-model/bin:$PATH"
: "${TMPDIR:=$PWD/.tmp}"
export TMPDIR
: "${TRITON_CACHE_DIR:=$PWD/.cache/triton}"
: "${TORCHINDUCTOR_CACHE_DIR:=$PWD/.cache/torch}"
: "${CUDA_CACHE_PATH:=$PWD/.cache/cuda}"
: "${XDG_CACHE_HOME:=$PWD/.cache/xdg}"
: "${XDG_CONFIG_HOME:=$PWD/.cache/xdg-config}"
: "${FLASHINFER_WORKSPACE_BASE:=$PWD/.cache/flashinfer}"
if [[ -z "${MM_LIMIT+x}" ]]; then
  MM_LIMIT='{"image":8}'
fi
export TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR CUDA_CACHE_PATH XDG_CACHE_HOME XDG_CONFIG_HOME FLASHINFER_WORKSPACE_BASE
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
export CUDA_VISIBLE_DEVICES
MODEL_PATH="${MODEL_PATH:-checkpoints/Qwen3.8-27B}"
[[ -f "$MODEL_PATH/model.safetensors.index.json" ]] || { echo 'Download the checkpoint first' >&2; exit 1; }
exec .venv-model/bin/vllm serve "$MODEL_PATH" \
  --served-model-name Qwen/Qwen3.8-27B --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size "${TP_SIZE:-4}" --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-8192}" --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --gpu-memory-utilization 0.85 --enable-prefix-caching --enable-chunked-prefill \
  --async-scheduling --generation-config vllm \
  --limit-mm-per-prompt "$MM_LIMIT" "$@"
