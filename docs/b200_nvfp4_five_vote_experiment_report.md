# B200 NVFP4 robot-monitor experiment report

Date: 2026-11-11

## Objective

Bring the Qwen3.8-27B visual monitor up on this machine, verify the native NVFP4
kernel on a B200, add the five-vote design described in
`18_vlm_monitor_latency_engineering_zh.md`, and measure complete-assessment
throughput over real videos rather than a single request.

## Runtime established

| Component | Value |
|---|---|
| GPU | NVIDIA B200, 183,359 MiB, compute capability 10.0 |
| Model | `nvidia/Qwen3.8-27B-NVFP4`, 20.42 GiB checkpoint |
| Model path | `/data/jliu3292/models/Qwen3.8-27B-NVFP4` |
| Python environment | `/data/jliu3292/robot-agent-vllm` |
| PyTorch | 2.13.0+cu130 |
| vLLM | 0.31.0 |
| FlashInfer | 0.7.0.post1 |
| Model endpoint | `http://127.0.0.1:8000/v1` |
| Monitor adapter | port 8081; restart with `--votes 5` to activate five-vote mode |

vLLM selected `FlashInferCuteDslNvFp4LinearKernel` for NVFP4 GEMM and compiled
the SM100 kernels successfully. Model weights, compilation caches, FlashInfer JIT
artifacts and the runtime are on `/data`; the launch script was changed to keep
vLLM caches off the nearly full root filesystem.

## Five-vote implementation

Five-vote mode sends five identical OpenAI-compatible requests concurrently.
Each request has its own existing eight-second timeout and no retry. Thinking is
disabled and sampling uses temperature 0.7. The server has prefix caching enabled
and permits eight concurrent sequences.

Invalid, timed-out or transport-failed votes abstain without lowering the
agreement threshold. Four or five agreeing terminal votes accept immediately.
Three agreeing terminal votes require the same result on the next fresh window.
A non-terminal result or a changed terminal result clears the pending streak.

The mode is selected with `--votes 5`. Single-vote mode remains available with
`--votes 1`. Benchmark output now records all individual votes, individual errors,
aggregate status, token/cache usage and complete five-vote assessment latency.

## Benchmark method

The available G1 recordings contain two cameras (`rgb` and `wrist_left`), so the
tests use two cameras rather than the production target of three. Every assessment
contains eight chronological frames per camera, spaced by 0.25 seconds, encoded as
448x256 JPEGs. The wrist camera is rotated 180 degrees.

Video decoding and JPEG preparation happen before the timed request loop and are
excluded from inference latency. Assessments run sequentially, while the five model
requests inside each assessment run concurrently. The offline loop does not sleep:
`offline_request_loop_hz` is measured sustained capacity.

`assessment_hz_requested` controls the spacing of source-video windows. It is not
the achieved inference rate. For example, requesting 5 Hz produces windows every
0.2 seconds of video, but the engine processes them at its measured capacity.

## Results

| Run | Source video | Source rate | Windows | p50 | p95 | Wall time | Sustained rate | Prefix-cache ratio | Vote errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Single vote, full phase | phase 1, 10.43 s | 1 Hz | 9 | 293.8 ms | 342.5 ms | 2.629 s | 3.423 Hz | 7.88% | n/a |
| Five votes, full phase | phase 1, 10.43 s | 1 Hz | 9 | 319.6 ms | 501.8 ms | 3.116 s | 2.888 Hz | 70.89% | 0/45 |
| Five votes, denser replay | phase 1, 10.43 s | 2.5 Hz | 22 | 349.6 ms | 409.7 ms | 7.868 s | 2.796 Hz | 59.93% | 1/110 |
| Five votes, capacity run | unseen phase 2, 9.17 s | 5 Hz | 38 | 330.0 ms | 425.6 ms | 13.116 s | **2.897 Hz** | 56.99% | 1/190 |

The phase-2 capacity run is the best estimate of current maximum throughput because
it used a video not processed by the earlier phase-1 runs and collected 38 complete
assessments. Its latency-derived rate is 3.030 Hz (`1 / p50`), while the directly
measured sustained rate is 2.897 Hz. The latter is the number to use for capacity
planning because it includes the complete serial request loop and aggregation.

The five-vote capacity run made 190 model requests. One vote failed and abstained;
all 38 aggregate assessments still completed. Five concurrent votes cost only a
small multiple of single-vote latency because identical requests within an
assessment reuse vLLM's prefix cache.

## Operating-rate conclusion

The measured raw ceiling is approximately **2.9 five-vote assessments per second**.
A 2.5 Hz live cadence gives each assessment 400 ms. The capacity run's p50 is below
that budget, but its p95 is 425.6 ms, so a one-assessment-in-flight scheduler will
occasionally skip a 2.5 Hz tick. A 2.8-2.9 Hz live target has too little latency
headroom even though it can be reached by the offline loop.

The recommended initial live setting is therefore **2.5 Hz**, with the documented
skip-if-busy scheduling rule and explicit monitoring of skipped ticks. This is a
throughput recommendation, not an accuracy result.

## Interpretation limits

- The dataset has two cameras; production calls for three cameras and 24 images.
- The benchmark is offline and unpaced. Camera capture, transfer and live JPEG
  preparation are outside its inference timing.
- Phase-1 five-vote tests followed earlier phase-1 processing, so their cache ratios
  include cross-run reuse. The unseen phase-2 run reduces this concern.
- Prefix-cache reuse among the five identical votes is intentional and represents
  the deployed five-vote design.
- Only phases 1 and 2 were exercised for throughput. Status correctness and task
  accuracy were not scored against ground-truth transition timestamps.
- The first one-window smoke test took about 868 ms; after kernel compilation and
  warm-up, the full-video single-vote p50 fell to 293.8 ms. Warm numbers describe
  steady-state service, not cold startup.

## Verification

The repository test suite passes 13 tests, including five-vote acceptance and error
abstention behavior. Mock replay also passes. The real benchmarks produced valid
aggregate responses for every assessment.

Raw summaries and per-window predictions are stored in:

- `outputs/benchmark/b200-single-full/`
- `outputs/benchmark/b200-five-full/`
- `outputs/benchmark/b200-five-full-2.5hz/`
- `outputs/benchmark/b200-five-max-phase2/`

## Remaining validation

Before fixing 2.5 Hz as the production rate, run a paced live-camera test with all
three cameras, measure capture-to-verdict latency and skipped ticks, and replay all
five phases against labeled transition times. That test should separately report
TTFT, complete five-vote latency, accepted-decision latency, vote abstentions and
prefix-cache hit rate.
