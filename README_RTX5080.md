# RTX 5080 inference and throughput report

Results recorded on **2026-11-24** for Robot Agent V2.

The latest measured rate for the configured three-vote profile is **1.346 complete
assessments/s (Hz)** on one RTX 5080. Each assessment runs **three phase votes and
three safety votes** on the same fresh camera window. Full assessment latency is
**724.5 ms p50**, **763.0 ms p95**, and **742.8 ms mean**. All 50 windows completed
without backend errors; all 300 individual votes were valid.

This profile combines shared vision encoding, KV prefix caching, concurrent vote
submission, and decode-only CUDA graphs. It is **3.46× faster** than the earlier
sequential three-vote run without KV caching. Both runs accepted **2/5 phases**;
the speed improvements have not resolved the visual verification problems.

**Live status at documentation check:** the local `robot-agent-v2` systemd user
service was inactive and `http://127.0.0.1:8082/health` refused the connection.
The 1.346 Hz figure is a saved benchmark result, not a live throughput reading.
This report was checked against existing artifacts; no new GPU benchmark was run.

## What improved inference and throughput

| Change | How it helps | Measured effect on this RTX 5080 |
|---|---|---|
| One FP8 checkpoint on the GPU | One loaded Qwen3-VL-8B model serves both critic prompts; the recorded deployment uses no CPU weight offloading. B12X provides the block-FP8 linear kernels. | The optimized run loaded successfully and reached 13.03 GiB peak PyTorch allocation. No same-model BF16 comparison was measured. |
| Shared vision encoding | Both critics and all votes reuse the same window's image embeddings. Each new window receives fresh image UUIDs. | Legacy single-vote comparison: **1.095 → 1.205 Hz (+10.0%)**. Encoder p50 fell **154.2 → 76.7 ms**; encoded images fell **1,600 → 800** across 50 windows. |
| KV prefix caching | Reuses each critic's matching language-model prefix, including that window's image tokens for later votes. The two critics have different instructions, so they do not share one complete language KV prefix. | Sequential three-vote comparison: **0.389 → 0.582 Hz (+49.6%)**; assessment p50 **2,552.7 → 1,698.6 ms**. |
| Concurrent vote submission | Submits all six requests together to one vLLM engine with `max_num_seqs=6`, allowing their decoding to be batched. | With KV caching and graphs off: **0.582 → 1.098 Hz (+88.5%)**; p50 **1,698.6 → 910.6 ms**. |
| Decode-only CUDA graphs | Replays captured decode execution for the configured sequence counts, reducing repeated launch overhead. Prefill graphs and Inductor compilation remain disabled. | Concurrent three-vote comparison: **1.098 → 1.346 Hz (+22.6%)**; p50 **910.6 → 724.5 ms (−20.4%)**. |

Across the complete three-vote optimization sequence, throughput rose **245.8%**
and p50 latency fell **71.6%**. These are measured changes on the same recording,
with the same vote count and grounded prompts. They are not averages over repeated
benchmark trials or a claim about another GPU.

The optimized three-vote run encoded **16 images per window**, for **800 images**
total, with **4,000 image reuse hits**. Its language KV cache reused
**593,232 / 821,310 prompt tokens (72.2%)**. Image reuse and KV token reuse are
separate counters. Neither cache reuses the final verdict or treats an older
window as fresh evidence.

Other choices keep the workload bounded: thinking is off, each response has only
two fields and at most 32 output tokens, the static rubric precedes changing
images, and there are no retries. Offline video decoding and JPEG preparation
happen before timing. These are workload controls; no isolated speedup is claimed
for them in today's ablations.

The project-local compiler, Python headers, venv tool path, and CUDA toolkit/linker
setup enabled the recorded runtime to initialize. Earlier graph startup failures
are retained in `robot-agent-v2/reports/graph-startup-*`; failed initialization
attempts have **no inference Hz**.

## Today's RTX 5080 results

All rows below use 50 real, changing windows from the same two-camera chain.
“Votes” means votes **per critic**; Hz counts completed windows after both critics
and every configured vote have finished. Each listed run had 50/50 valid windows.

| Configuration and saved summary | Votes | KV cache | Scheduling | CUDA graphs | Assessment Hz | p50 / p95 (ms) | Accepted phases |
|---|---:|---|---|---|---:|---:|---:|
| [Legacy prompts, unshared vision](robot-agent-v2/reports/rtx5080-unshared/summary.json) | 1 | Off | One pair | Off | 1.095 | 907.1 / 958.2 | 0/5 |
| [Legacy prompts, shared vision](robot-agent-v2/reports/rtx5080/summary.json) | 1 | Off | One pair | Off | 1.205 | 824.1 / 853.5 | 0/5 |
| [Grounded prompts, shared vision](robot-agent-v2/reports/rtx5080-adjusted/summary.json) | 1 | Off | One pair | Off | **1.071** | 930.4 / 962.9 | 2/5 |
| [Three votes, no KV cache](robot-agent-v2/reports/rtx5080-vote3/summary.json) | 3 | Off | Sequential pairs | Off | 0.389 | 2,552.7 / 2,649.2 | 2/5 |
| [Five votes, no KV cache](robot-agent-v2/reports/rtx5080-vote5/summary.json) | 5 | Off | Sequential pairs | Off | 0.241 | 4,120.8 / 4,242.2 | 2/5 |
| [Three votes + KV cache](robot-agent-v2/reports/rtx5080-vote3-kv/summary.json) | 3 | On | Sequential pairs | Off | 0.582 | 1,698.6 / 1,793.0 | 2/5 |
| [First pair, then remaining votes together](robot-agent-v2/reports/rtx5080-staged3/summary.json) | 3 | On | Staged | Off | 0.764 | 1,301.1 / 1,376.1 | 2/5 |
| [All three votes together](robot-agent-v2/reports/rtx5080-concurrent3/summary.json) | 3 | On | Concurrent | Off | 1.098 | 910.6 / 962.0 | 2/5 |
| [Configured three-vote profile](robot-agent-v2/reports/rtx5080-graphs3/summary.json) | 3 | On | Concurrent | On | **1.346** | **724.5 / 763.0** | 2/5 |
| [All five votes together + graphs](robot-agent-v2/reports/rtx5080-concurrent5/summary.json) | 5 | On | Concurrent | On | **1.311** | 762.2 / 801.2 | 2/5 |

The five-vote concurrent run also enabled CUDA graphs, despite its directory name.
It was slightly slower than the three-vote profile and accepted no additional
phases. Simply adding sequential votes increased latency without improving phase
acceptance. The staged cache-priming control also remained slower than submitting
all votes together.

The **1.071 Hz** row is the measured grounded single-vote baseline with KV caching
and graphs disabled. The generic `serve.py` now defaults to one vote and KV caching
enabled, so that row does not measure its exact current defaults. An optimized
single-vote + KV + graphs result has not been established by these artifacts.

## What “inference Hz” measures

For the configured three-vote profile:

```text
sustained assessment Hz = completed assessments / timed loop wall seconds
                        = 50 / 37.1498956
                        = 1.3458988 Hz
```

This is about **743 ms of loop wall time per complete assessment on average**.
It is not tokens/s, camera FPS, the reciprocal of p50 latency, or the rate of
accepted phase transitions. The video sampling interval was **1 second (1 Hz)**;
the benchmark loop did not sleep to match playback.

| Timing quantity | Optimized three-vote result | Meaning |
|---|---:|---|
| Phase TTFT, p50 | 330.1 ms | First phase vote's time from submission to first delivered token, including request preprocessing. |
| Safety TTFT, p50 | 526.5 ms | First safety vote's corresponding time. |
| Individual phase vote full response, p50 | 687.5 ms | Median over phase votes while scheduled with the other requests. |
| Individual safety vote full response, p50 | 695.3 ms | Median over safety votes under that same load. |
| Full assessment, p50 / p95 | 724.5 / 763.0 ms | Entire six-request assessment, including preprocessing and aggregation. |
| Accepted pipette-pick decision compute | 1,467.1 ms | Sum of assessment compute across its two confirming windows. |
| Accepted pipette-return decision compute | 1,447.1 ms | Sum of assessment compute across its two confirming windows. |

Individual vote timings under batching do not measure isolated single-request
capacity. Each accepted completion required **1 second of source-time
confirmation** between fresh windows. That interval and accumulated compute are
different from live camera-to-decision latency, which was not measured.

The timed loop includes per-window image processing, encoder execution, prefill,
all decodes, encoder instrumentation, state aggregation, and loop bookkeeping.
Model loading, warmup, and video/JPEG preparation are excluded. The optimized run
recorded **35.54 s model initialization**, including its graph setup, outside the
reported steady-state throughput. Encoder timing uses GPU synchronization, so
the figures describe the instrumented benchmark.

## Hardware and workload

- **GPU:** one NVIDIA GeForce RTX 5080, 16 GB, SM120; recorded driver 595.99.02.
- **Checkpoint:** `Qwen/Qwen3-VL-8B-Instruct-FP8`, revision
  `9cdc6310a8cb770ce18efaf4e9935334512aee45`; FP8 weights, BF16 activations and KV.
- **Recorded software:** Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0,
  vLLM 0.31.0, Transformers 5.17.0, B12X linear backend.
- **Input:** real G1 pipette episodes 237–241, a 51.083 s continuous sequence;
  **2 cameras × 8 chronological frames**, 0.25 s spacing, **448×256** per image.
  The left view is rotated 180 degrees and letterboxed. No synthetic third view
  or repeated static images are used.
- **Replay:** windows end at 1.75, 2.75, …, 50.75 s. `--schedule recorded` supplies
  the expected phase from recorded stage boundaries and continues for diagnosis.
- **Three-vote decoding:** temperature 0.7, top-p 0.9, seeds 0/1/2, thinking off,
  at most 32 tokens per response, no retry, one 8 s deadline for the entire group.
  Deadline checks occur between engine steps; a running GPU kernel cannot be
  preempted. Default single-vote decoding is deterministic at temperature zero.

The three-vote profile is an explicit experimental configuration. The production
contract remains deterministic single-vote inference with **3 cameras × 8
frames**. Today's two-camera numbers do not establish three-camera throughput.
Historical B200 and A5000 measurements in the main README also use different
hardware/models and should not be presented as RTX 5080 capacity.

## Is the Qwen prompt the same as on B200?

**The physical phase criteria are the same, but the final RTX 5080 prompt is not
identical.** The current profile uses `prompt_mode=grounded`, which starts from
the original phase prompt and adds instructions and frame labels.

| Aspect | Historical B200 throughput benchmark | Final RTX 5080 three-vote profile |
|---|---|---|
| Phase task | Verify the controller-supplied `expected_phase` using its active success/failure rules. | Same task and physical rules. |
| Base phase prompt | Original monitor instructions plus `ACTIVE RULES`, followed by camera-major chronological images. | Retains that base and adds explicit target identity/tracking instructions. |
| Frame labels | One overall camera/frame-order description. | Adds camera number, source timestamp, and age or `NEWEST` before every image. |
| Object identity | Identify by function and initial location; track the same objects. | Reinforces tracking the actual manipulated tube, avoiding another tube in the rack, and distinguishing projection overlap from physical support. |
| Final question | Short request for a two-field verdict. | Repeats the active success/failure criteria and asks whether the phase is physically complete at the newest time. |
| Critics in the throughput measurement | Phase critic only. | Phase critic plus a separate object-support safety critic. |
| Output | Phase: `status/confidence`. | Same phase fields; safety separately returns `pipette_support/tube_support`, with constrained JSON decoding. |
| Checkpoint | `nvidia/Qwen3.8-27B-NVFP4`. | `Qwen/Qwen3-VL-8B-Instruct-FP8`. |

The base phase prompt builder is [context.py](src/robot_agent/context.py), with
[monitor instructions](src/robot_agent/prompts/monitor.txt) and the shared
[phase criteria](configs/monitor.toml). The local V2 implementation adds its
grounded instructions in `robot-agent-v2/prompting.py`; `--prompt-mode legacy`
bypasses those additions and returns the original conversation structure. That
option does not restore the B200 checkpoint, its model-specific chat template,
or its phase-only workload.

The B200 five-vote acceptance policy also differed: four agreeing terminal votes
could accept immediately, while three required confirmation in the next window.
The current three-vote profile requires a confident majority in two consecutive
windows. This is an acceptance-policy difference, separate from prompt wording.
See the [historical B200 experiment](docs/b200_nvfp4_five_vote_experiment_report.md).

These runs change the checkpoint, prompt formatting, critic workload, and vote
policy as well as the GPU. They therefore cannot establish that the GPU or prompt
alone caused the missing phase-2/4 completion votes. The RTX 5080 legacy and
grounded runs provide a same-model comparison: grounded prompting improved
accepted phases from 0/5 to 2/5 on this chain, but did not recover phases 2 or 4.

## Verification results and remaining limitations

Grounded prompts added chronological labels, function-based object identity, and
explicit physical completion criteria. Constrained safety JSON eliminated the
invalid support enums seen in the unconstrained run. On this development chain,
phase acceptance improved from **0/5 to 2/5**, while the longer prompts reduced the
early single-vote throughput from **1.205 to 1.071 Hz**. This was a verification
tradeoff, not a prompt-length speedup.

The optimized three-vote run accepted pipette pick at **9.75 s** and pipette return
at **50.75 s**. The episode result remained **`INCOMPLETE_AT_END`**.

### Did phases 2, 3, and 4 receive any success votes?

**Phase 3 received strong success votes; phases 2 and 4 received none.** The final
three-vote run (`rtx5080-graphs3`) recorded these individual phase-critic results:

| Expected phase | Windows | Success votes / total | Remaining votes | Accepted completion |
|---|---:|---:|---|---|
| **2 — Tube pick** | 9 | **0/27** | All 27 were `in_progress`. | No |
| **3 — Pipette aiming** | 8 | **6/24** | The other 18 were `in_progress`. | No |
| **4 — Tube return** | 9 | **0/27** | All 27 were `in_progress`. | No |

At **20.75 s** and **26.75 s**, all three phase-3 votes returned `success`, with
confidences **1.0, 0.99, and 1.0** in each window. Both windows passed the
within-window majority rule. They were not consecutive: intervening windows
returned `in_progress`, resetting confirmation, and the recorded expected phase
changed to phase 4 after the second success window. Acceptance requires at least
two of three success votes with confidence **≥0.70**, followed by another
successful majority in the **next consecutive fresh window**. Phase 3 therefore
had positive completion judgments but did not satisfy temporal confirmation.

Phases 2 and 4 had **no positive completion judgments**, including in windows
where the safety critic returned `safe`. Their missing acceptance was not caused
solely by safety stops or low-confidence success votes. Lowering the confidence
threshold alone cannot turn their `in_progress` votes into `success`. For example,
`{"status":"in_progress","confidence":0.8}` expresses confidence in the
`in_progress` status; it is not an 80% completion probability. The aggregate
`in_progress/confidence=0` denotes abstention from a terminal majority; the
individual model confidences remain in the raw records.

All **78 phase votes** across phases 2–4 were valid, with no backend or parsing
errors. These counts describe completion judgments on this recording, not a
phase-identification accuracy score. FC supplies `expected_phase`; the monitor
checks whether that phase's physical completion criteria are visible. It does
not independently select phase 2, 3, or 4.

Step 3 remains aiming only: the tip must be visibly above and aligned with the
held tube opening. Steps 4/5 still require stable target support after gripper
release. Evidence: [saved per-window and per-vote predictions](robot-agent-v2/reports/rtx5080-graphs3/predictions.jsonl).

### Safety and evaluation limits

Six windows accepted `TUBE_ON_UNINTENDED_SURFACE`, beginning at **11.75 s**.
The source recording is labeled successful, but it lacks independent safety and
completion-onset annotations. These counts are not a measured false-positive rate
or an accuracy score. Later diagnostic accepts are not closed-loop execution
outcomes: a controller honoring the first stop would not continue this replay.

Concurrency and graphs preserved the 2/5 accepted-phase result and six safety-stop
windows. They did not make every raw judgment identical: graph-on versus graph-off
three-vote runs agreed on phase status in 50/50 windows and on both safety-support
labels in 49/50. Caching and batching do not guarantee identical floating-point
outputs or independent evidence from repeated votes.

The separate support-contact prompt experiment increased safety-stop windows from
6 to 7 and was not selected. The
[offline acceptance sweep](robot-agent-v2/reports/acceptance-sweep/report.md)
produced no new inference measurements; lowering confirmation requirements is a
policy change and does not demonstrate a throughput or accuracy improvement.
FC continues to own phase order and motion. Backend/parse errors remain distinct
from visual failure; accepted completion persists until an accepted contradiction.

## Reproduce and inspect

Run from the repository root. With the documented environment installed, start
the experimental two-camera API using:

```bash
bash robot-agent-v2/serve_5080.sh
```

In another terminal, inspect its actual configuration:

```bash
curl http://127.0.0.1:8082/health
```

The wrapper explicitly selects `--cameras 2 --votes 3 --prefix-cache
--vote-execution concurrent --cuda-graphs`. The API owns one episode; send
`POST /reset` with `{}` before reuse. Busy observations return 429; the caller
should skip that snapshot and use fresh evidence on the next cycle.

To reproduce the offline profile, run the benchmark with the model API stopped
so it can own the GPU. Use a new output directory to preserve today's artifacts:

```bash
robot-agent-v2/.venv-vllm/bin/python robot-agent-v2/benchmark.py \
  --votes 3 --prefix-cache --vote-execution concurrent --cuda-graphs \
  --schedule recorded --assessment-hz 1 \
  --annotations robot-agent-v2/annotations/chain-237-241.json \
  --output robot-agent-v2/outputs/rtx5080-graphs3-rerun
```

The table links to saved summaries. The latest
[per-window and per-vote predictions](robot-agent-v2/reports/rtx5080-graphs3/predictions.jsonl)
are archived alongside its summary. The four newer scheduling/graph summaries
and these predictions were copied byte-for-byte from their original local
`robot-agent-v2/outputs/` runs when preparing this report.

For further context, see the [V2 usage README](robot-agent-v2/README.md),
[initial shared-vision report](robot-agent-v2/reports/rtx5080-report.md),
[prompt adjustment report](robot-agent-v2/reports/adjustment-report.md), and
[earlier vote/cache experiments](robot-agent-v2/reports/vote-cache-report.md).

Repository checks:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m robot_agent --backend mock replay
python3 -m unittest discover -s robot-agent-v2/tests -v
```

Documentation verification on 2026-11-24: **18 original tests + 36 V2 tests passed**,
and mock replay passed. All ten performance-table rows were checked against their
saved summaries; the latest raw vote counts, cache counts, safety-stop count, and
median assessment latency were also checked. All local links in this report resolve.
