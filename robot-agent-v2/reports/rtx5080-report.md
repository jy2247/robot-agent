# RTX 5080: Qwen3-VL-8B-Instruct-FP8 dual critics

Measured on 2026-11-24. The checkpoint **loads and runs entirely on the RTX 5080**, with no CPU weight offloading. Shared vision improves throughput by **10.0%**, but this configuration's verification results are poor on the tested chain: **0/5 accepted phase completions** and four accepted safety-stop windows on a source recording labeled successful. This run validates execution and encoder sharing; it does not validate the model for robot control.

## Measured performance

| Measurement | Shared encoder | Unshared encoder |
|---|---:|---:|
| Valid dual-critic assessments | 50/50 | 50/50 |
| Sustained dual-critic throughput | **1.205 Hz** | 1.095 Hz |
| Full pair latency, p50 | **824.1 ms** | 907.1 ms |
| Full pair latency, p95 | **853.5 ms** | 958.2 ms |
| Full pair latency, mean | 829.8 ms | 913.0 ms |
| Phase TTFT, p50 | 487.8 ms | 412.5 ms |
| Safety TTFT, p50 | 488.5 ms | 569.5 ms |
| Phase full response, p50 | 800.6 ms | 862.3 ms |
| Safety full response, p50 | 806.2 ms | 889.7 ms |
| Vision encoder time, p50 | **76.7 ms** | 154.2 ms |
| Encoded images across all pairs | **800** | 1,600 |
| Encoder calls across all pairs | 50 | 100 |
| Peak PyTorch allocated GPU memory | 13.15 GiB | 12.93 GiB |
| Timed loop wall time | 41.495 s | 45.656 s |

Hz is `50 / timed loop wall seconds`, not reciprocal median latency. Each assessment completes both critic requests. Per-critic timings are measured while the pair is scheduled together; they are not independent single-critic capacity measurements. Scheduling can improve combined latency while increasing one critic's TTFT. GPU allocation excludes desktop and other non-PyTorch memory.

## Workload and controls

- Exact checkpoint: `Qwen/Qwen3-VL-8B-Instruct-FP8`, revision `9cdc6310a8cb770ce18efaf4e9935334512aee45`.
- RTX 5080, compute capability 12.0, 16 GB; driver 595.99.02. Python 3.12.3, PyTorch 2.13.0+cu130, vLLM 0.31.0, Transformers 5.17.0, B12X 1.3.0.
- FP8 model weights, BF16 activations and KV cache. B12X block-FP8 kernels, FlashAttention, eager execution, no CUDA graphs, no language prefix cache.
- One loaded model, two separate rubric prompts, same image UUIDs within a shared pair. Fresh UUIDs in every window prevent encoder reuse across assessments. The baseline assigns different image UUIDs to each critic.
- Instrumentation wraps the actual worker `embed_multimodal` method; counts are not inferred from HTTP or preprocessing-cache hits. GPU synchronization for encoder timing is included in both runs.
- Real [G1 pipette dataset](https://huggingface.co/datasets/jren313/g1-pipette-2view-teleop0925-5task-eerel), episodes 237–241, consecutive stages of session `20261007_1319`, original episode `episode_0002`.
- 51.083-second continuous sequence; two real camera views, eight chronological frames per camera, 0.25-second spacing, 448×256. No synthetic third camera. Left view rotated 180 degrees and letterboxed. The playback videos are re-encoded with high-quality H.264 before JPEG sampling.
- Windows end at 1.75, 2.75, …, 50.75 seconds: video sampling is 1 Hz. The loop does not sleep. No repeated static benchmark inputs.
- Temperature zero, thinking disabled, 32 output tokens per critic, no retries, eight-second deadline checked between engine steps. A running GPU kernel cannot be preempted. The initial compilation exceeded this budget during setup, before the reported run; no timed assessment exceeded it.
- Preparation and model loading are excluded. Per-window image decoding/preprocessing, inference, encoder counter RPCs, and state aggregation are included. TTFT begins at pair submission and includes request preprocessing.

## Verifier outputs

The replay supplies expected phases using recorded stage boundaries, emulating FC input. It intentionally continues after stop verdicts to evaluate all five stages; this is not a robot controller.

| Expected phase | First success vote | Accepted success |
|---|---:|---|
| Pipette pick | None | No |
| Tube pick | None | No |
| Pipette aiming | None | No |
| Tube return | None | No |
| Pipette return | 49.75 s | No; 50.75 s returned in_progress |

Final episode result: **INCOMPLETE_AT_END**. Accepted-decision latency is unavailable because no terminal completion was accepted; it must not be inferred from single-vote latency.

The safety critic accepted `TUBE_ON_UNINTENDED_SURFACE` at 12.75, 27.75, 28.75, and 32.75 seconds. These conflict with the dataset's successful-source label and require frame-level review. The label alone is not independent safety ground truth, so this is not a measured false-positive rate.

Between shared and unshared runs:

- Phase statuses agree on **50/50** windows.
- Safety raw verdicts agree on **50/50** windows.
- Phase raw verdicts agree on **45/50** windows; the remaining five differ only in confidence. Deterministic sampling does not guarantee identical floating-point results under different scheduling/batching.

This evidence points to a verifier/model limitation in this configuration rather than changed categorical decisions caused by sharing. It does not establish parity with the previous 27B model, and a single chain cannot characterize accuracy or safety sensitivity.

## Local deployment and artifacts

- Viewer: http://127.0.0.1:8765/ — synchronized videos, stage seeking, telemetry, JSON links, and an annotated MP4.
- Adapter: http://127.0.0.1:8082/health — currently configured for this dataset's **two** cameras. `serve.py --cameras 3` supports the production input shape, but three-camera throughput has not been measured.
- Annotated video: `../outputs/rtx5080/critic-comparison.mp4`.
- Versionable evidence: [shared summary](rtx5080/summary.json), [shared predictions](rtx5080/predictions.jsonl), [unshared summary](rtx5080-unshared/summary.json), [comparison](comparison.json).
- Runtime logs and intermediate startup diagnostics are under `../outputs/rtx5080/`.

User-level transient services `robot-agent-v2` and `robot-agent-v2-viewer` are running; they are not enabled to start at reboot. Stop them with `systemctl --user stop robot-agent-v2 robot-agent-v2-viewer`. Stopping the model service releases its GPU memory.

The server initially lacked a compiler and Python development headers. Setup unpacks GCC/G++ 13 and Python headers into `.toolchain/` without sudo. The selected B12X backend targets the 5080's SM120 architecture. The original repository's implementation is unchanged; V2 lives in its own folder and imports existing state machines.

Validation: 18 original unit tests, mock replay, four V2 tests, real HTTP inference (1,055 ms including transport), duplicate rejection, episode reset, and browser video decoding/seeking/synchronized playback/telemetry/rewind checks passed. Legacy per-phase adapter timing labels in the first benchmark files were corrected using their already-recorded `v2_metrics`; verdicts and measured timing values were not changed.
