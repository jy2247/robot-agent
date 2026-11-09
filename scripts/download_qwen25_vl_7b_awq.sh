#!/usr/bin/env bash
set -euo pipefail

repo_id="${HF_MODEL_REPO:-Qwen/Qwen2.5-VL-7B-Instruct-AWQ}"
model_dir="${1:-${MODEL_PATH:-/data/jingyu/Qwen2.5-VL-7B-Instruct-AWQ}}"
hf_bin="${HF_BIN:-/data/jingyu/H-PEEK/.venv/bin/hf}"

mkdir -p "$model_dir"
exec "$hf_bin" download "$repo_id" \
  --local-dir "$model_dir" \
  --max-workers "${HF_MAX_WORKERS:-2}"
