# Monotonic historical-evidence decoder comparison

Date: 2026-11-13

## Question

Does a monotonic phase decoder that accumulates historical five-vote confidence evidence reduce the
cascade where one missed transition delays or prevents every later phase?

Both runs use the same ten reconstructed chains (episodes 237-286), two cameras, Qwen3.8-27B-NVFP4,
one B200, five votes per assessment and a 2.5 Hz video cadence. Because temperature is 0.7 and the
decoder changes later phase prompts after its first divergent transition, the second run necessarily
uses a new stochastic inference sample; it is not an exact paired replay.

## Decoder

For each window, individual vote confidences are converted into masses by dividing each confidence by
five and summing by status. Historical evidence is updated as:

`score[t] = max(0, 0.65 * score[t-1] + success - 0.05 * in_progress - 0.20 * fail)`

For phases 1-4, this is multiplied by a soft duration prior: a logistic curve centered at 4.0 seconds
with scale 0.8 seconds. A transition is accepted when the current success mass and posterior score are
both at least 0.45. Phase order is monotonic. The final phase bypasses the duration prior because it has
no downstream transition to corrupt; this terminal exemption was added after the first decoder pass
showed that late phase-4 detection left too little video to satisfy the soft prior.

## Aggregate comparison

| Metric | Current baseline | Monotonic decoder | Change |
|---|---:|---:|---:|
| Transition recall | 35/40 (87.5%) | 37/40 (92.5%) | +5.0 pp |
| Transition MAE, detected only | 5.422 s | 3.479 s | -1.944 s (-35.8%) |
| Mean signed transition error | +2.533 s | +2.734 s | +0.201 s later |
| Midpoint-estimate MAE, detected only | 5.441 s | 3.441 s | -2.000 s (-36.8%) |
| Full five-phase sequence accuracy | 7/10 (70%) | 9/10 (90%) | +20.0 pp |

The plan improves transition recall, conditional MAE and full-chain accuracy. It does not improve
signed bias: detected transitions remain late on average, and the new decoder is about 0.2 seconds
later overall.

### Per-transition comparison

| Boundary | Baseline recall / MAE / bias | Decoder recall / MAE / bias |
|---|---|---|
| Phase 1 → 2 | 100% / 0.993 s / -0.960 s | 100% / 1.040 s / -1.040 s |
| Phase 2 → 3 | 90% / 6.085 s / -1.000 s | 90% / 3.741 s / +3.622 s |
| Phase 3 → 4 | 90% / 6.830 s / +4.937 s | 90% / 4.770 s / +4.137 s |
| Phase 4 → 5 | 70% / 9.088 s / +8.974 s | 90% / 4.635 s / +4.635 s |

## Interpretation

The historical decoder materially reduces cascade error, especially for the final transition. It does
not repair missing visual evidence: chain 6 still never produced enough tube-pick success evidence, so
the decoder remained on phase 2. This failure requires better visual prompting, views or model quality,
not stronger temporal smoothing.

The 90% result is promising but not a final generalization result. The final-phase exemption was
introduced after inspecting this test run, so it must be validated on the remaining 20 five-step chains
as an untouched holdout. The dataset also contains no retained failure chains, so failure precision and
recall remain unmeasured.

Raw baseline results:
`/data/jliu3292/robot-agent-results/g1-five-vote-10chains/summary.json`

Raw decoder results:
`/data/jliu3292/robot-agent-results/g1-five-vote-10chains-monotonic/summary.json`
