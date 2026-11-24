# Robot agent V2 — shared vision, dual critics

当前 RTX 5080 推荐实验启动方式：`bash robot-agent-v2/serve_5080.sh`（在仓库根目录执行）。该配置明确启用 **3 vote、并发双 critic、共享视觉编码、KV prefix cache、decode-only CUDA graphs**；API 配置端口为 8082。同一双相机 50 窗口实测 **1.346 Hz、p50 724.5 ms、p95 763 ms**。阶段接受仍为 2/5，速度优化没有修复视觉误判。优化方法与今日完整结果见 [RTX 5080 inference / throughput README](../README_RTX5080.md)。2026-11-24 文档检查时本地 API 未运行；这些数值来自已保存的 benchmark。

V2 runs **Qwen/Qwen3-VL-8B-Instruct-FP8** on one RTX 5080. One checkpoint serves two prompt-based critics; these are not newly trained classification heads. The phase critic returns `status/confidence`; the safety critic returns `pipette_support/tube_support`. Their existing parsers and confirmation state machines are reused from `../src/robot_agent`.

Each assessment submits both critics to one vLLM engine. Multi-vote scheduling supports concurrent submission, sequential pairs, and a first-pair-then-concurrent staged control. Matching image UUIDs let vLLM share encoder outputs within the pair. UUIDs are fresh for every window. KV prefix caching is enabled by default; the archived original baseline explicitly ran with it disabled. Worker instrumentation counts real encoder calls and encoded images: 16 encoded images per two-camera pair demonstrates sharing; the unshared comparison uses distinct UUIDs and should encode 32. A single encoder call may itself batch multiple images, so call count alone does not prove reuse.

Original baseline (legacy prompts): **1.205 shared dual-critic assessments/s**, versus **1.095 Hz** unshared (50 real windows each). Full pair p50/p95: **824/854 ms**. The model loaded, but **0/5 phases were accepted** on this chain. See [the full report](reports/rtx5080-report.md) before interpreting this as a usable verifier.

The adjusted implementation adds chronological camera labels, function-based target identity, constrained safety JSON, and an accepted-decision FC replay fixture. See [adjustment results](reports/adjustment-report.md). The default remains one vote per critic and two consecutive confirmations. Explicit 3/5-vote experiments are also supported.

## Setup and execution

From the repository root:

```bash
bash robot-agent-v2/setup.sh
robot-agent-v2/.venv-vllm/bin/python robot-agent-v2/benchmark.py --schedule recorded --annotations robot-agent-v2/annotations/chain-237-241.json
robot-agent-v2/.venv-vllm/bin/python robot-agent-v2/build_viewer.py robot-agent-v2/outputs/rtx5080-adjusted
python3 robot-agent-v2/viewer_server.py
```

Open http://127.0.0.1:8765 for synchronized camera replay, phase/safety verdicts, raw JSON, latency, and accepted outcomes.

Unshared encoder comparison:

```bash
robot-agent-v2/.venv-vllm/bin/python robot-agent-v2/benchmark.py --prompt-mode legacy --no-prefix-cache --schedule recorded --no-share --output robot-agent-v2/outputs/rtx5080-unshared
```

The local adapter defaults to the production three-camera input shape. For this dataset's two cameras:

```bash
robot-agent-v2/.venv-vllm/bin/python robot-agent-v2/serve.py --cameras 2 --port 8082
curl http://127.0.0.1:8082/health
```

The generic `serve.py` default remains one vote; `serve_5080.sh` is the explicit validated 3-vote profile.

POST `/observe` accepts the existing observation schema and returns both critic results; POST `/reset` with `{}` starts a new episode. The adapter serializes assessments and owns one episode. FC remains responsible for motion and phase order.

## Measurement and interpretation

- Input: actual episodes 237–241 of [G1 pipette](https://huggingface.co/datasets/jren313/g1-pipette-2view-teleop0925-5task-eerel). Provenance identifies consecutive stages of `20261007_1319/episode_0002`.
- Two available cameras, eight oldest-to-newest frames each, 0.25 s apart, 448×256. The left camera is rotated 180°. No duplicated third camera or synthetic end hold.
- Videos are letterboxed and joined for continuous cross-stage windows; this adds a high-quality H.264 re-encode. Frames remain real, temporally changing inputs.
- With `--schedule recorded`, replay supplies `expected_phase` from recorded boundaries as a diagnostic fixture. It continues through all five stages even if a critic asks to stop; it does not control a robot or claim autonomous execution success.
- Default `--schedule accepted` advances only on accepted phase success and stops on accepted failure. Run it into a separate directory with `--output robot-agent-v2/outputs/rtx5080-fc`. Recorded motion cannot wait for a late verdict; no artificial hold frames are added.
- `--annotations` loads completion intervals independently of recorded cuts. Missing intervals produce unknown transition delays; current visual checkpoints are not precise human-annotated onset times.
- `--prompt-mode grounded` constrains safety decoding to the original two-field schema; it does not map invalid labels after generation or guarantee visual accuracy. Legacy/labels modes remain unconstrained for diagnostic comparisons.
- Default single-vote decoding is deterministic; no thinking, max 32 tokens per response, no retry; eight-second deadlines cancel unfinished requests between engine steps. An already-running GPU kernel cannot be preempted. Two consecutive confident terminal windows are required.
- `assessment_hz_requested` is video sampling rate. `sustained_dual_assessment_hz` is completed pairs divided by actual inference-loop wall time. They are different quantities.
- Model loading, warmup, video decoding, and JPEG preparation are excluded from throughput; per-window image processing and the two decodes are included.
- TTFT and full response are separate per-critic measurements (per-vote timings are retained in multi-vote mode); combined pair latency and confirmation delay are recorded separately. Offline acceptance compute time is not live capture-to-decision latency.
- A backend or parse error never becomes a visual failure. Missing accepted phases at the end produce `INCOMPLETE_AT_END`.
- A single successful-source chain does not establish safety accuracy or false-negative rates. Three-camera capacity requires a separate real three-camera recording.

## Multi-vote and KV prefix cache experiments

```bash
robot-agent-v2/.venv-vllm/bin/python robot-agent-v2/benchmark.py --votes 3 --prefix-cache --vote-execution sequential --schedule recorded --output robot-agent-v2/outputs/rtx5080-vote3-kv --annotations robot-agent-v2/annotations/chain-237-241.json
```

`--votes 3` or `--votes 5` applies to both critics. These are explicitly requested experimental multi-request modes, distinct from the production one-request-per-critic path. Default multi-vote sampling is temperature 0.7, top-p 0.9, seeds 0…N−1. Use `--vote-temperature 0` for deterministic repetition. The whole vote group has one eight-second deadline, no retries, and each response still has exactly two fields and a 32-token maximum. The two-window confirmation requirement is preserved.

Phase aggregation needs a strict majority of the configured vote count, each terminal vote with confidence ≥0.70; reported terminal confidence is their mean, not an inflated probability. Without a confident terminal majority it abstains as in_progress with confidence 0; inspect individual votes for the model's original confidences. Invalid votes never shrink the denominator. Safety uses a per-object strict majority; disagreement becomes unknown, not safe. An insufficient valid-vote majority is a backend error, not a physical failure.

Vision UUIDs remain fresh across real windows, but are shared between critics and all votes within one window. `--prefix-cache` additionally enables vLLM automatic KV prefix caching. Each critic reuses its own identical prompt prefix in later votes, including the same window's image tokens. Different critic instructions usually prevent sharing the full language KV across critics. Changing windows invalidate image-prefix reuse through fresh image UUIDs; matching text-only prefixes may still hit. Cached prefixes do not reuse generated verdicts. Cache hit token counts and per-vote TTFT/full latency are saved, separately from vision image-cache hits and entire-group latency. KV prefix caching is now enabled by default, including for one vote; use `--no-prefix-cache` only for controlled ablations.

`--prompt-mode support_contact` is a safety-only ablation: it clarifies palm/finger support, visible unintended-surface contact, and unknown under unresolved support. Phase instructions and thresholds are unchanged. It is not a hardcoded suppression of safety failures. See [vote/cache comparison](reports/vote-cache-report.md) for measured results and limitations.

## Runtime audit and scheduling

[RTX 5080 inference / throughput report](../README_RTX5080.md) documents the measured effects of shared vision, KV caching, concurrent scheduling, and CUDA graphs. The B200 figures in the root README are historical measurements, not this RTX 5080's capacity.

- `--vote-execution concurrent` (default): submit all phase/safety votes to vLLM together, with up to `2 × votes` live sequences. The engine retains a single shared vision encoder and fresh window UUIDs.
- `--vote-execution sequential`: retained control used in the earlier V2 vote/cache report.
- `--vote-execution staged`: complete the first pair, then submit remaining votes concurrently, a cache-priming control.
- `--cuda-graphs`: optional full decode-only CUDA graphs, bounded to the configured concurrent sequence count; no prefill graph or Inductor compilation. Enable only with the verified 5080 profile in the audit report.
- HTTP `/observe` and `/reset` use a nonblocking single-flight guard. A second request returns 429 `monitor_busy` / `skip_this_snapshot`; the caller should skip that snapshot, not retry stale frames. Health remains responsive during inference. Reset requires exactly `{}`.
- Keep short two-field outputs, static task/system prefix before images, one-time real frame/JPEG preparation outside offline timing, structured safety decoding, chronological temporal windows, no retry and model error abstention.
- Five-vote fast acceptance and the historical monotonic decoder are policy changes, not speedups. The first has an explicit offline comparison; the second hardcodes five-vote normalization and must not take phase control away from FC. Neither is silently enabled by a cache/scheduling flag.

## Offline acceptance parameter sweep

`evaluate_acceptance.py` replays saved votes with adjustable confidence thresholds,
success-confirmation counts, duration priors and evidence decay. It preserves the
recorded expected-phase schedule and safety outcomes; it does not run inference or
change the service. See [the sweep report](reports/acceptance-sweep/report.md),
[all configurations](reports/acceptance-sweep/sweep.csv), and
[per-phase results](reports/acceptance-sweep/sweep.json).

Across 315 configurations on each of three runs of the same recording, confidence
thresholds 0.50–0.90 still accept 2/5 phases with two consecutive windows. One-window
success confirmation can accept 3/5; phases 2 and 4 have no success votes. Duration
center 2 s, decay 0.85, and score threshold 0.35 do not add accepts while keeping
two-window confirmation. These are sensitivity results, not accuracy or latency
improvements validated on a holdout. The first archived safety stop at 11.75 s
also means later diagnostic accepts are not closed-loop execution outcomes.

```bash
python3 robot-agent-v2/evaluate_acceptance.py \
  robot-agent-v2/outputs/rtx5080-graphs3 \
  --output robot-agent-v2/outputs/acceptance-sweep
```

## Checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m robot_agent --backend mock replay
python3 -m unittest discover -s robot-agent-v2/tests -v
```

The configured experimental profile uses `serve_5080.sh`; the viewer is configured to open the optimized three-vote run and link older comparisons. At the 2026-11-24 documentation check, the `robot-agent-v2` user service was inactive and port 8082 refused connections. Check live status with `systemctl --user status robot-agent-v2 robot-agent-v2-viewer`; saved benchmark Hz does not establish service availability. If running under these user services, stop both with `systemctl --user stop robot-agent-v2 robot-agent-v2-viewer`. The transient services are not enabled at reboot.
