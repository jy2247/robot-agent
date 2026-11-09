#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

model_env="${MODEL_ENV:-/data/jingyu/H-PEEK/.venv}"
model_path="${MODEL_PATH:-/data/jingyu/Qwen2.5-VL-7B-Instruct-AWQ}"
model_name="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
data_root="${G1_CHAIN_ROOT:-$PWD/data/g1-pipette-chain-237-241}"
run_dir="${RUN_DIR:-$PWD/outputs/benchmark/phase1-qwen25-vl-7b-awq-single-gpu}"
base_url="${ROBOT_VLM_BASE_URL:-http://127.0.0.1:8000/v1}"
server_root="${base_url%/v1}"
server_pid=""

mkdir -p "$run_dir"

stop_server() {
  if [[ -n "$server_pid" && "${KEEP_SERVER:-0}" != "1" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap stop_server EXIT INT TERM

if ! curl -fsS "$server_root/health" >/dev/null 2>&1; then
  echo "Starting Qwen2.5-VL-7B AWQ on GPU ${CUDA_VISIBLE_DEVICES:-0}; log=$run_dir/vllm.log"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    MODEL_ENV="$model_env" MODEL_PATH="$model_path" \
    bash scripts/serve_qwen25_vl_7b_awq.sh >"$run_dir/vllm.log" 2>&1 &
  server_pid=$!
  ready=0
  for _ in $(seq 1 "${SERVER_READY_TIMEOUT_S:-900}"); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Qwen2.5-VL-7B vLLM exited before becoming ready" >&2
      tail -n 120 "$run_dir/vllm.log" >&2
      exit 1
    fi
    if curl -fsS "$server_root/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != "1" ]]; then
    echo "Qwen2.5-VL-7B vLLM did not become ready before timeout" >&2
    exit 1
  fi
fi

curl -fsS "$server_root/metrics" >"$run_dir/metrics-before.txt" || true

ROBOT_VLM_MODEL="$model_name" PYTHONPATH=src "$model_env/bin/python" \
  scripts/benchmark_single_vote.py \
  --camera "$data_root/videos/chunk-000/observation.images.rgb/episode_000241.mp4" \
  --camera "$data_root/videos/chunk-000/observation.images.wrist_left/episode_000241.mp4" \
  --rotate-180-camera 1 \
  --phase step5_pipette_return \
  --timeout-s "${WARMUP_TIMEOUT_S:-120}" \
  --max-assessments 1 \
  --output-dir "$run_dir/warmup"

ROBOT_VLM_MODEL="$model_name" PYTHONPATH=src "$model_env/bin/python" \
  scripts/benchmark_single_vote.py \
  --camera "$data_root/videos/chunk-000/observation.images.rgb/episode_000237.mp4" \
  --camera "$data_root/videos/chunk-000/observation.images.wrist_left/episode_000237.mp4" \
  --rotate-180-camera 1 \
  --phase step1_pipette_pick \
  --timeout-s "${BENCHMARK_TIMEOUT_S:-120}" \
  --output-dir "$run_dir/phase1"

curl -fsS "$server_root/metrics" >"$run_dir/metrics-after.txt" || true
echo "result=$run_dir/phase1/summary.json"
