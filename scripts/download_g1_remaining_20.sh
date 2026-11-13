#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

hf_bin="${HF_BIN:-/data/jliu3292/robot-agent-vllm/bin/hf}"
destination="${1:-/data/jliu3292/robot-agent-data/g1-pipette-chains-10}"
repo="jren313/g1-pipette-2view-teleop0925-5task-eerel"

files=()
for episode_number in $(seq 287 386); do
  episode=$(printf '%06d' "$episode_number")
  files+=("videos/chunk-000/observation.images.rgb/episode_${episode}.mp4")
  files+=("videos/chunk-000/observation.images.wrist_left/episode_${episode}.mp4")
done

"$hf_bin" download "$repo" "${files[@]}" \
  --repo-type dataset \
  --local-dir "$destination" \
  --max-workers "${HF_MAX_WORKERS:-4}"

missing=0
for file in "${files[@]}"; do
  if [[ ! -s "$destination/$file" ]]; then
    echo "missing=$destination/$file" >&2
    missing=$((missing + 1))
  fi
done
if (( missing > 0 )); then
  echo "download incomplete: $missing files missing" >&2
  exit 1
fi
echo "remaining_20_videos_ready=200 destination=$destination"
