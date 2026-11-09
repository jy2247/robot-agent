#!/usr/bin/env bash
set -euo pipefail

# Download a complete resumable snapshot for later offline serving.
repo_id="${HF_MODEL_REPO:-mattbucci/Qwen3.8-27B-AWQ}"
model_dir="${1:-${MODEL_PATH:-/data/jingyu/Qwen3.8-27B-AWQ}}"
hf_bin="${HF_BIN:-/data/jingyu/H-PEEK/.venv/bin/hf}"
model_env="${MODEL_ENV:-/data/jingyu/H-PEEK/.venv}"

mkdir -p "$model_dir"
"$hf_bin" download "$repo_id" \
  --local-dir "$model_dir" \
  --max-workers "${HF_MAX_WORKERS:-2}"
"$model_env/bin/python" "$(dirname "$0")/prepare_qwen_awq_checkpoint.py" "$model_dir"
