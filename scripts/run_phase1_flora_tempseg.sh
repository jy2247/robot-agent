#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

model_env="${MODEL_ENV:-/data/jingyu/H-PEEK/.venv}"
flora_repo="${FLORA_REPO:-/data/jingyu/Flora-TempSeg}"
data_root="${G1_CHAIN_ROOT:-$PWD/data/g1-pipette-chain-237-241}"
run_dir="${RUN_DIR:-$PWD/outputs/benchmark/phase1-flora-tempseg}"
gpu="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$run_dir"

export HF_HOME="${HF_HOME:-/data/jingyu/.cache/huggingface}"
export FLORA_MODEL_CACHE="${FLORA_MODEL_CACHE:-/data/jingyu/.cache/flora/models}"
export PYTHONPATH="$flora_repo/.deps:$flora_repo${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$gpu"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader >"$run_dir/gpu-before.csv"

"$model_env/bin/python" scripts/run_flora_tempseg.py \
  --flora-repo "$flora_repo" \
  --video "$data_root/videos/chunk-000/observation.images.rgb/episode_000237.mp4" \
  --output "$run_dir/result.json" \
  --overwrite 2>&1 | tee "$run_dir/run.log"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader >"$run_dir/gpu-after.csv"

echo "result=$run_dir/result.json"
