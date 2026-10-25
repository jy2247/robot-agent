#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: ./run_agent.sh VIDEO_OR_EPISODE_DIR [--phase PHASE] [--output FILE]" >&2
  exit 2
fi

mkdir -p outputs/one-click .tmp
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
  echo "[1/3] Starting local Qwen3.8-27B vLLM server..."
  bash scripts/serve_qwen.sh > outputs/one-click/vllm.log 2>&1 &
  SERVER_PID=$!
  STARTED_SERVER=1
  for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "vLLM exited during startup; see outputs/one-click/vllm.log" >&2
      exit 1
    fi
    sleep 1
  done
  if ! curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "Timed out waiting for vLLM; see outputs/one-click/vllm.log" >&2
    exit 1
  fi
else
  echo "[1/3] Reusing the running local vLLM server."
fi

echo "[2/3] Running temporal inference and rendering MP4..."
PYTHONPATH=src:. .venv-model/bin/python scripts/infer_video.py "$@"
echo "[3/3] Done. The paths above contain the MP4 and JSON outputs."
