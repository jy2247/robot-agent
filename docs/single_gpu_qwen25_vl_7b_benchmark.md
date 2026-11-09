# Single-GPU Qwen2.5-VL-7B AWQ benchmark

## Setup

- Model: `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`
- Hardware: one RTX A5000 (SM 8.6), GPU 0
- Runtime: vLLM 0.19.1, AWQ-Marlin, FP16 activations, CUDA graphs
- Input: episode 237, two cameras × eight frames, 448×256 source JPEGs
- Agent: one vote per window, thinking off, temperature 0, maximum 32 output tokens
- Model weights loaded: 6.57 GiB

## Result

| Metric | 7B single GPU | 27B two GPUs |
| --- | ---: | ---: |
| Complete responses | 9 / 9 | 9 / 9 |
| Backend errors | 0 | 0 |
| Full-response p50 | 819.3 ms | 2,063.9 ms |
| Full-response p95 | 1,011.0 ms | 2,156.6 ms |
| Full-response mean | 844.0 ms | 2,065.4 ms |
| Sequential single-vote rate | **1.221 Hz** | 0.485 Hz |
| First success | 9.75 s | 8.75 s |
| Two-window terminal acceptance | not reached | 9.75 s |

The 7B model is 2.52× faster by median sequential rate. It marked only the final available window as
success, so the agent correctly did not accept the terminal state without a second fresh confirmation.
The 27B model marked both 8.75 s and 9.75 s as success. This one-video result measures latency and
exposes a possible one-window detection delay; it is not yet an accuracy benchmark across all phases.

Artifacts are under
`outputs/benchmark/phase1-qwen25-vl-7b-awq-single-gpu-measured/`. The phase summary contains all
measurement parameters and points to the per-window raw predictions.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_ENV=/data/jingyu/H-PEEK/.venv \
MODEL_PATH=/data/jingyu/Qwen2.5-VL-7B-Instruct-AWQ \
bash scripts/run_phase1_qwen25_vl_7b_awq.sh
```
