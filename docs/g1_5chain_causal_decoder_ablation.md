# Five-chain strictly causal temporal-decoder ablation

## Outcome

The monotonic evidence decoder improved transition recall from 75% to 90% and
full five-phase chain accuracy from 60% to 80% on the five requested test
chains. Its default settings also reduced conditional boundary MAE from 6.387 s
to 6.089 s without increasing false transitions, but mean online commit latency
increased by 0.988 s. On this small set, a 2 s soft-duration center gave the
best tested boundary MAE (5.567 s) at the same recall, false-transition count,
and chain accuracy.

## Protocol

- Dataset: `huggingface/jren313/g1-pipette-2view-telep0925-5task-eerel`
- Tests: exactly five continuous five-task chains, episodes 237--261
- Input: two cameras, eight frames per camera, 0.25 s frame spacing
- Replay cadence: 2.5 Hz (0.4 s)
- VLM and five-vote aggregation: unchanged; all decoder variants replay the
  same stored evidence
- Causality: rows are consumed in timestamp order; a decoder sees only the
  current and prior rows for its current phase
- Ground truth: unavailable during replay and used only for post-replay scoring
- False transition: decoder commit more than 2 s before the ground-truth boundary
- Boundary MAE is conditional on a detected boundary

The fixed evidence bank contains 601 causal windows, 3,005 five-vote phase
assessments, and 15,025 individual votes. Eighty-four individual votes (0.56%)
had parse errors, while all phase-level five-vote assessments completed without
a backend-level error.

## Results

| Decoder | Recall | Boundary MAE (s) | Signed error (s) | Online latency (s) | Perception delay (s) | Decoder delay (s) | False | Full chain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current baseline | 75% | 6.387 | +3.740 | +3.993 | +6.473 | -2.480 | 2 | 60% |
| Monotonic full | 90% | 6.089 | +4.737 | +4.981 | +6.359 | -1.378 | 2 | 80% |
| Decay 0.00 | 90% | 6.089 | +4.737 | +4.981 | +6.359 | -1.378 | 2 | 80% |
| Decay 0.35 | 90% | 6.089 | +4.737 | +4.981 | +6.359 | -1.378 | 2 | 80% |
| Decay 0.85 | 90% | 5.622 | +4.270 | +4.537 | +6.359 | -1.822 | 2 | 80% |
| No duration prior | 90% | 5.919 | +0.870 | +1.115 | +6.359 | -5.244 | 7 | 80% |
| Duration center 2 s | 90% | **5.567** | +4.215 | +4.537 | +6.359 | -1.822 | 2 | 80% |
| Duration center 6 s | 90% | 6.556 | +5.315 | +5.604 | +6.359 | -0.756 | **1** | 80% |
| Threshold 0.35 | 90% | 5.622 | +4.270 | +4.537 | +6.359 | -1.822 | 2 | 80% |
| Threshold 0.55 | 75% | 6.776 | +5.233 | +5.567 | +6.473 | -0.907 | **1** | 60% |

For each detected boundary, the JSON result logs ground-truth time, first
strong VLM evidence at or after ground truth, decoder commit time, estimated
boundary time, perception delay, decoder delay, online latency, and signed and
absolute boundary errors. A negative decoder delay means the decoder committed
from premature evidence before the first qualifying post-boundary VLM evidence;
it is not a claim of negative processing time.

## Interpretation

- The monotonic state machine prevents out-of-order phase jumps and materially
  improves recall and full-chain completion. A genuinely missed current phase
  still blocks later transitions, which is intentionally counted as incomplete.
- Duration gating is important. Removing it looks fast but produces seven false
  transitions versus two for the baseline and default decoder.
- A stronger evidence memory (`decay=0.85`) or lower threshold (`0.35`) improves
  timing on this set without changing recall or false transitions.
- The 2 s duration center is the best measured tradeoff here: 12.8% lower MAE
  than baseline, 15 percentage points higher recall, and 20 percentage points
  higher full-chain accuracy, at a 0.544 s increase in mean online latency.
- Most residual delay is perception delay, especially around visually ambiguous
  middle and return phases. Decoder tuning alone cannot remove that component.

These five chains are enough for the requested controlled comparison, but are
too few to treat the selected parameters as a generalization result. The 2 s
center should be validated on a separate holdout before becoming the production
default. The source dataset retains successful recordings only, so this run
cannot estimate failure precision or failure recall; it evaluates successful
phase recognition, transition timing, and premature-transition behavior.

## Artifacts

- Evaluation driver: `scripts/evaluate_temporal_decoder_ablation.py`
- Decoder: `src/robot_agent/sequence.py`
- Raw summary: `/data/jliu3292/robot-agent-results/g1-five-vote-5chains-causal-ablation/ablation_summary.json`
- Fixed evidence: `/data/jliu3292/robot-agent-results/g1-five-vote-5chains-causal-ablation/chain-*-evidence.jsonl`
