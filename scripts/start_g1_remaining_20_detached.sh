#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

gpu_id="${GPU_ID:?Set GPU_ID to the single allocated GPU index}"
session="${MISSION_SESSION:-g1-remaining-20}"
output_root="${OUTPUT_ROOT:-/data/jliu3292/robot-agent-results/g1-remaining-20-causal-ablation}"
supervisor_log="$output_root/supervisor.log"

if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must name exactly one GPU index" >&2
  exit 2
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi
mkdir -p "$output_root"

tmux new-session -d -s "$session" -c "$PWD" \
  "GPU_ID='$gpu_id' MISSION_CLIENTS='${MISSION_CLIENTS:-1}' OUTPUT_ROOT='$output_root' \
   bash -lc 'set +e
   echo mission_started_at=\"\$(date --iso-8601=seconds)\" >>\"$supervisor_log\"
   scripts/run_g1_remaining_20_mission.sh >>\"$supervisor_log\" 2>&1
   mission_rc=\$?
   echo mission_finished_at=\"\$(date --iso-8601=seconds)\" rc=\$mission_rc >>\"$supervisor_log\"
   echo final_gpu_release_check >>\"$supervisor_log\"
   nvidia-smi -i \"\$GPU_ID\" --query-gpu=index,memory.used,memory.free,utilization.gpu \
     --format=csv,noheader >>\"$supervisor_log\" 2>&1
   exit \$mission_rc'"

echo "detached_session=$session"
echo "supervisor_log=$supervisor_log"
echo "monitor=tmux attach -t $session"
