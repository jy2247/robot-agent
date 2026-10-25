#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

OUT_DIR="${1:-outputs/random-phase-batch-seed-20261020}"
SEED="${SEED:-20261020}"
mkdir -p "$OUT_DIR" outputs/one-click .tmp

echo "[1/4] Sampling ten chains and downloading their source videos..."
PYTHONPATH=src:. .venv-model/bin/python scripts/random_phase_batch.py \
  --out-dir "$OUT_DIR" --seed "$SEED" --prepare-only

STARTED_SERVER=0
SERVER_PID=""
cleanup() {
  if [[ "$STARTED_SERVER" == 1 && -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if ! curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "[2/4] Starting local Qwen3.8-27B vLLM server..."
  bash scripts/serve_qwen.sh > outputs/one-click/vllm-random-batch.log 2>&1 &
  SERVER_PID=$!
  STARTED_SERVER=1
  for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "vLLM exited during startup; see outputs/one-click/vllm-random-batch.log" >&2
      exit 1
    fi
    sleep 1
  done
  curl -fsS http://127.0.0.1:8000/v1/models >/dev/null || {
    echo "Timed out waiting for vLLM" >&2
    exit 1
  }
else
  echo "[2/4] Reusing the running local vLLM server."
fi

echo "[3/4] Running raw temporal inference and rendering ten MP4 files..."
PYTHONPATH=src:. .venv-model/bin/python scripts/random_phase_batch.py \
  --out-dir "$OUT_DIR" --seed "$SEED" --run-only
echo "[4/4] Done: $OUT_DIR"
