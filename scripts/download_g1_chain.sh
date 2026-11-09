#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

hf_bin="${HF_BIN:-hf}"
destination="${1:-$PWD/data/g1-pipette-chain-237-241}"
repo="jren313/g1-pipette-2view-teleop0925-5task-eerel"

files=(meta/info.json meta/tasks.jsonl meta/episodes.jsonl
       meta/source_provenance.jsonl meta/source_notes.json)
for episode in 000237 000238 000239 000240 000241; do
  files+=("videos/chunk-000/observation.images.rgb/episode_${episode}.mp4")
  files+=("videos/chunk-000/observation.images.wrist_left/episode_${episode}.mp4")
done

"$hf_bin" download "$repo" "${files[@]}" \
  --repo-type dataset \
  --local-dir "$destination" \
  --max-workers "${HF_MAX_WORKERS:-2}"

echo "chain=$destination"
