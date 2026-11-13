# Ten-chain five-vote full-video evaluation

Date: 2026-11-13

## Setup

- Dataset: `jren313/g1-pipette-2view-teleop0925-5task-eerel`
- Chains: dataset episodes 237-286, reconstructed as ten gap-free five-phase timelines
- Cameras: head RGB plus 180-degree-rotated left wrist (the dataset has no third live view)
- Model: `nvidia/Qwen3.8-27B-NVFP4`, served as `Qwen/Qwen3.8-27B`
- Hardware: one NVIDIA B200, physical GPU 0, TP=1
- Agent mode: five concurrent votes, temperature 0.7, 2.5 assessments/s
- Input: 8 chronological frames per camera at 0.25 s spacing, 448x256 JPEG

The evaluator does not expose episode boundaries to the model. It decodes each group of five adjacent
episode MP4s as one continuous frame timeline and advances `expected_phase` only after the monitor
accepts success for the current phase.

## Results

| Metric | Result |
|---|---:|
| Full 1→2→3→4→5 sequence recognized | 7/10 (70%) |
| Successful-chain recall | 7/10 (70%) |
| Transition recall | 35/40 (87.5%) |
| Raw transition MAE, detected transitions only | 5.422 s |
| Raw mean signed transition error | +2.533 s |
| Last-negative/first-positive midpoint MAE | 5.441 s |
| Valid assessments | 1,151/1,151 |
| Backend errors | 0 |
| Individual vote errors | 50/5,755 (0.87%) |
| Sustained five-vote assessment throughput | 2.833 Hz |
| Median of per-chain p50 inference latency | 341.2 ms |
| Worst per-chain p95 inference latency | 444.4 ms |

The source dataset retains only successful recordings. This run can measure successful-chain recall,
but it cannot measure failure precision, failure recall, or specificity. Two transient accepted `fail`
decisions occurred in otherwise successful chains; later accepted success evidence revoked them under
the current agent state rules.

### Per-boundary timing

| Boundary | Detected | Conditional MAE | Mean signed error |
|---|---:|---:|---:|
| Phase 1 → 2 | 10/10 | 0.993 s | -0.960 s |
| Phase 2 → 3 | 9/10 | 6.085 s | -1.000 s |
| Phase 3 → 4 | 9/10 | 6.830 s | +4.937 s |
| Phase 4 → 5 | 7/10 | 9.088 s | +8.974 s |

MAE is explicitly conditional on detecting a boundary. Reporting 5.422 s without the 87.5% transition
recall would hide five missed boundaries.

## Why the timestamp error cascades

The main error is not a constant model delay. A 3/5 success vote requires the same terminal result on
the next fresh window. Some physical phase completions are visible for only one assessment before the
scene enters the next phase. If that single 3/5 window is not confirmed, FC remains on the old phase.
All following phases are then queried late or missed, which explains the increasing positive error at
later boundaries.

The midpoint between the last non-terminal window and first terminal-evidence window only removes
sampling delay. It changed aggregate MAE from 5.422 s to 5.441 s, confirming that timestamp
quantization is not the primary problem.

## Recommended MAE fix

Use a monotonic sequence-aware evidence decoder rather than treating every phase boundary as an
independent hard terminal event:

1. Preserve the five individual vote statuses as per-window evidence.
2. Enforce the known phase order and a conservative minimum dwell time (the dataset documentation says
   every phase is at least 4.8 s; use 4.0 s rather than fitting the ten test chains).
3. After the dwell guard, allow one 3/5 success window to create a boundary candidate. Confirm it from
   subsequent evidence for the next phase instead of requiring another old-phase success window.
4. Localize the accepted boundary to the midpoint between the last old-phase-negative and first
   old-phase-positive evidence timestamps.
5. Keep failure acceptance separate and strict; never convert transport or parse errors into failure.

This specifically addresses missed transient boundaries and prevents one miss from shifting every later
timestamp. It should be evaluated in a new run; applying a phase-specific offset to these same ten test
chains would leak ground truth and would not fix transition recall.

Raw output is stored outside the Git repository at
`/data/jliu3292/robot-agent-results/g1-five-vote-10chains/summary.json`, with one predictions JSONL file
per chain in the same directory.
