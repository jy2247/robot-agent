#!/usr/bin/env bash
set -euo pipefail

# Download the public NVFP4 checkpoint from Hugging Face. The hf client resumes
# incomplete downloads automatically when this script is run again.
repo_id="${HF_MODEL_REPO:-nvidia/Qwen3.8-27B-NVFP4}"
model_dir="${1:-${MODEL_PATH:-/data/jingyu/Qwen3.8-27B-NVFP4}}"
hf_bin="${HF_BIN:-hf}"
revision="${HF_REVISION:-main}"
workers="${HF_MAX_WORKERS:-4}"
hf_home="${HF_HOME:-/data/jingyu/.cache/huggingface}"

if ! command -v "$hf_bin" >/dev/null 2>&1; then
  cat >&2 <<EOF
找不到 Hugging Face CLI: $hf_bin
请先安装：python -m pip install -U huggingface_hub
也可以通过 HF_BIN 指定完整路径，例如：
  HF_BIN=/path/to/venv/bin/hf bash $0
EOF
  exit 127
fi

mkdir -p "$model_dir" "$hf_home"
export HF_HOME="$hf_home"

echo "仓库: $repo_id"
echo "版本: $revision"
echo "目录: $model_dir"

"$hf_bin" download "$repo_id" \
  --revision "$revision" \
  --local-dir "$model_dir" \
  --max-workers "$workers"

test -s "$model_dir/model.safetensors.index.json"
test -s "$model_dir/hf_quant_config.json"

echo "下载完成: $model_dir"
