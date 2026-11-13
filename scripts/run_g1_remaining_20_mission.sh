#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

gpu_id="${GPU_ID:?Set GPU_ID to the single allocated GPU index}"
clients="${MISSION_CLIENTS:-1}"
dataset_root="${DATASET_ROOT:-/data/jliu3292/robot-agent-data/g1-pipette-chains-10}"
output_root="${OUTPUT_ROOT:-/data/jliu3292/robot-agent-results/g1-remaining-20-causal-ablation}"
model_env="${MODEL_ENV:-/data/jliu3292/robot-agent-vllm}"
model_path="${MODEL_PATH:-/data/jliu3292/models/Qwen3.8-27B-NVFP4}"
server_log="$output_root/vllm.log"

if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must name exactly one GPU index" >&2
  exit 2
fi
if [[ "$clients" != "1" && "$clients" != "2" ]]; then
  echo "MISSION_CLIENTS must be 1 or 2" >&2
  exit 2
fi

missing=0
for episode_number in $(seq 287 386); do
  episode=$(printf '%06d' "$episode_number")
  for view in observation.images.rgb observation.images.wrist_left; do
    file="$dataset_root/videos/chunk-000/$view/episode_${episode}.mp4"
    [[ -s "$file" ]] || missing=$((missing + 1))
  done
done
if (( missing > 0 )); then
  echo "$missing required videos are missing." >&2
  echo "Run scripts/download_g1_remaining_20.sh before requesting the GPU." >&2
  exit 1
fi

mkdir -p "$output_root"
if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "port 8000 already has a model server; refusing to replace it" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
MODEL_ENV="$model_env" MODEL_PATH="$model_path" \
ROBOT_CACHE="${ROBOT_CACHE:-/data/jliu3292/.cache/robot-agent}" \
PORT=8000 bash scripts/serve_qwen_nvfp4.sh >"$server_log" 2>&1 &
server_pid=$!
evaluator_pids=()
cleanup() {
  if (( ${#evaluator_pids[@]} > 0 )); then
    kill "${evaluator_pids[@]}" 2>/dev/null || true
    wait "${evaluator_pids[@]}" 2>/dev/null || true
  fi
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "waiting_for_vllm pid=$server_pid gpu=$gpu_id"
for _ in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM exited during startup; inspect $server_log" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null

run_shard() {
  local test_start="$1"
  local first_episode=$((237 + (test_start - 1) * 5))
  local test_end=$((test_start + 4))
  local shard="$output_root/tests-$(printf '%02d' "$test_start")-$(printf '%02d' "$test_end")"
  if [[ -s "$shard/ablation_summary.json" ]]; then
    echo "skip_completed=$shard"
    return
  fi
  mkdir -p "$shard"
  PYTHONPATH=src "$model_env/bin/python" scripts/evaluate_temporal_decoder_ablation.py \
    --config configs/monitor.toml \
    --dataset-root "$dataset_root" \
    --first-episode "$first_episode" \
    --chains 5 \
    --assessment-hz 2.5 \
    --output-dir "$shard" \
    --quiet >"$shard/mission.log" 2>&1
  echo "completed_tests=${test_start}-${test_end}"
}

starts=(11 16 21 26)
if [[ "$clients" == "1" ]]; then
  for start in "${starts[@]}"; do
    run_shard "$start"
  done
else
  for offset in 0 2; do
    run_shard "${starts[$offset]}" &
    first_pid=$!
    run_shard "${starts[$((offset + 1))]}" &
    second_pid=$!
    evaluator_pids=("$first_pid" "$second_pid")
    wait "$first_pid"
    wait "$second_pid"
    evaluator_pids=()
  done
fi

summaries=(
  "$output_root/tests-11-15/ablation_summary.json"
  "$output_root/tests-16-20/ablation_summary.json"
  "$output_root/tests-21-25/ablation_summary.json"
  "$output_root/tests-26-30/ablation_summary.json"
)
"$model_env/bin/python" scripts/aggregate_causal_ablation_shards.py \
  --output "$output_root/remaining-20-summary.json" "${summaries[@]}"
echo "mission_complete=$output_root/remaining-20-summary.json"
