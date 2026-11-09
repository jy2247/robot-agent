# Single-vote benchmark status

## A5000 AWQ result

The complete `mattbucci/Qwen3.8-27B-AWQ` snapshot is stored at
`/data/jingyu/Qwen3.8-27B-AWQ`. It was tested with vLLM 0.19.1 on two RTX A5000 GPUs
(SM 8.6), tensor parallel size 2, AWQ-Marlin, FP16 activations, an 8192-token context,
one running sequence, NCCL all-reduce and eager execution.

The first phase uses the two real episode-237 camera videos, eight chronological frames per
camera, 448×256 images, thinking disabled, temperature zero and a 32-token output limit. Video
decoding and JPEG preparation occur before the timed request loop.

| Metric | Result |
| --- | ---: |
| Complete responses | 9 / 9 |
| Backend errors | 0 |
| Prompt tokens, median | 2,229 |
| Completion tokens, median | 13 |
| Full-response latency p50 | 2,063.9 ms |
| Full-response latency p95 | 2,156.6 ms |
| Full-response latency mean | 2,065.4 ms |
| Sequential single-vote rate | **0.485 Hz** |
| First visual success | source time 8.75 s |
| Accepted after two confirmations | source time 9.75 s |

The separate phase-5 warmup request took 2,804.6 ms. The benchmark timeout was raised to 120 s
only to prevent cold-start truncation; the production config remains 8 s. The output is at
`outputs/benchmark/phase1-awq-a5000-measured/`, including `vllm.log`, metrics before/after,
the warmup record, all nine raw predictions and `phase1/summary.json`.

## Compatibility fixes

The checkpoint correctly stores the vision tower, `lm_head`, and two Gated DeltaNet scalar
projections in FP16, but its initial `modules_to_not_convert` list was empty. The local preparation
script records those exclusions explicitly. This checkpoint also nests vision keys below
`model.language_model.visual`; vLLM 0.19.1 lacks that exact prefix mapping, so the startup script
applies a narrow loader patch before serving.

CUDA graph profiling reached a custom all-reduce `invalid argument` on this A5000 topology.
The validated baseline therefore disables custom all-reduce and uses eager execution. This is a
correctness-first baseline; later optimization can independently test CUDA graphs with NCCL after
the end-to-end result is preserved.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_ENV=/data/jingyu/H-PEEK/.venv \
MODEL_PATH=/data/jingyu/Qwen3.8-27B-AWQ \
bash scripts/run_phase1_awq.sh
```

The report records raw responses, parsed decisions, token usage, full-response p50/p95 and
`1000 / p50_ms` sequential Hz using fresh 16-frame windows. Do not substitute TTFT or offline
video preparation time for full verdict latency.

## NVFP4 comparison

The original NVFP4 checkpoint cannot start on RTX A5000: ModelOpt requires compute capability
8.9, while this host is 8.6. Its failed diagnostic remains in `outputs/diagnostics/nvfp4_runtime.json`.
