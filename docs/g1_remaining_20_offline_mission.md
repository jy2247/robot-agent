# Remaining 20 five-stage tests: one-GPU offline mission

## Scope

Run strictly causal five-vote evidence collection and temporal-decoder ablations
for full tests 11--30 (episodes 287--386). Tests 1--10 are excluded. The mission
uses one NVIDIA B200 and writes four independently resumable five-test shards,
followed by a combined 20-test summary.

## Before allocating a GPU

Download the 200 missing videos. This uses no GPU and is expected to add about
3.4 GB based on the first 100 downloaded videos.

```bash
cd /home/jliu3292/robot-agent
HF_MAX_WORKERS=4 scripts/download_g1_remaining_20.sh
```

## GPU request

Request one NVIDIA B200 192 GB for **90 minutes**. The expected run is 52--60
minutes with one client; the extra allocation time covers cold model startup,
JIT compilation, filesystem variability, and safe shutdown. CPU video decoding
benefits from at least 8 CPU cores. Request 16 GB host RAM and roughly 8 GB of
free disk space for videos, evidence JSONL, and logs.

The completed five-test run measured 914.15 seconds for 601 causal windows, or
1.521 seconds per window. Tests 11--30 contain approximately 1,820 windows:

- Model work: about 46.1 minutes
- Input preparation: about 4--5 minutes
- Cold server startup/JIT and aggregation: about 3--8 minutes
- Expected total: about 52--60 minutes

Video download time is excluded and should be completed before GPU allocation.

## Launch

Safest, reproducible mode:

```bash
cd /home/jliu3292/robot-agent
GPU_ID=0 MISSION_CLIENTS=1 scripts/run_g1_remaining_20_mission.sh
```

The mission refuses to start if any required video is absent or if port 8000 is
already occupied by a model server. Completed five-test shards are skipped on a
restart.

## Concurrency decision

Do not launch a second vLLM server on the same GPU. The server reserves 85% of
the 183,359 MiB B200 (about 156 GiB), leaving insufficient safe headroom for a
second copy with the same settings.

Two evaluator clients may share the one server:

```bash
GPU_ID=0 MISSION_CLIENTS=2 scripts/run_g1_remaining_20_mission.sh
```

Each client submits five concurrent votes, while vLLM permits eight active
sequences. Two clients therefore submit ten requests and two will queue. The
theoretical throughput improvement is capped near 1.6x, and this mode has not
yet been benchmarked end-to-end. Allow 40--55 minutes and still request the same
90-minute allocation. Use one client when reproducibility and lowest contention
matter more than possible wall-time savings.

## Outputs

- Combined result: `/data/jliu3292/robot-agent-results/g1-remaining-20-causal-ablation/remaining-20-summary.json`
- Per-shard evidence and results: `tests-11-15`, `tests-16-20`, `tests-21-25`,
  and `tests-26-30` under the same directory
- Server log: `vllm.log`
