# V2 adjustment evaluation — RTX 5080

The adjusted verifier accepts pipette pick and pipette return in this recorded five-stage diagnostic. It still misses stable acceptance of tube pick, aiming and tube return. Safety judgments remain unreliable: this is an experimental replay deployment, not a validated robot controller.

## Changes

- Label every image with camera, source time and age; retain both cameras × eight chronological frames, 0.25 s spacing, 448×256.
- Identify targets by function and initial support; repeat the active physical success/failure criteria. Step 3 still requires aiming only; returns still require stable support after release.
- Constrain safety decoding to the existing two-field JSON schema and enums. No post-hoc remapping, extra explanation, retry, or extra vote.
- Share vision encoding between critics: 800 images encoded over 50 pairs, 800 within-pair cache hits. No reuse across windows.
- Separate recorded-stage diagnostic replay from the default accepted-decision FC fixture. The latter advances only on accepted success and stops on accepted failure. FC still owns motion; this fixture cannot hold or alter recorded motion.
- Add completion-interval evaluation and separate recorded cuts from accepted times in the viewer. Preserve baseline evidence and links.

## Measured comparison

Same real two-view chain 237–241, same 50 windows, Qwen3-VL-8B-Instruct-FP8, deterministic decoding, max 32 tokens per critic, one vote each, two consecutive confident windows for phase acceptance.

| Metric | Original baseline | Adjusted |
|---|---:|---:|
| Valid dual assessments | 50/50 | 50/50 |
| Accepted phases | 0/5 | 2/5 |
| Sustained pairs/s | 1.205 | 1.071 |
| Pair p50 / p95 | 824 / 854 ms | 930.4 / 962.9 ms |
| Accepted safety-stop windows | 4 | 6 |

Safety-stop window counts are not independent failure events or calibrated false-positive rates. Adjusted stops occur at 11.75, 12.75, 27.75, 28.75, 31.75 and 32.75 s, all TUBE_ON_UNINTENDED_SURFACE. The extra stops are a regression, not evidence of improved safety. Successful-source labels do not independently annotate safety.

Phase 1: first success 8.75 s, accepted 9.75 s. Phase 5: first success 49.75 s, accepted 50.75 s. Each needs 1 s of source-time confirmation. Pair compute is about 0.93 s; two-window accumulated compute is about 1.85 s. These are distinct quantities, neither is a measured live camera-to-decision delay. Phase 3 has an isolated success at 20.75 s but no accepted completion. Final result remains INCOMPLETE_AT_END.

TTFT p50: phase 359.7 ms, safety 584.8 ms. Full-response p50: phase 880.7 ms, safety 910.1 ms. Throughput excludes model loading, warmup and video/JPEG preparation; includes per-window image preprocessing and both decodes. The longer prompts reduce throughput compared with baseline. No three-camera throughput claim is made.

## Diagnostic evidence and limits

Before schema constraints, native vLLM chat rendering and the manual rendering path gave identical raw phase and safety outputs on all seven diagnostic windows. This does not test another quantization or another inference implementation. Labels/grounded prompts recovered stable final return success; newest-frame-only input still missed phases 2–4. A broader postcondition rewrite was rejected because it lost final success and emitted invalid safety enums.

The first full grounded run had five tube_support=holder parse errors, retained in `rtx5080-grounded-unconstrained/`. Schema-constrained rerun has none. Schema validity does not establish visual correctness. Ablations used this same chain and are in-sample development evidence, not held-out accuracy estimates.

`annotations/chain-237-241.json` records assistant-reviewed positive checkpoints, not independent human completion-onset labels. All completion intervals are currently null. Therefore true transition-detection delays remain unknown; recorded cuts (10.433, 19.600, 27.350, 35.967, 51.083 s) are never substituted for ground truth. Supply reviewed onset intervals to `--annotations` for delay intervals.

Next accuracy work should use independently annotated held-out episodes, explicit target tracking/crops or task-specific training, and separately validate support judgments before connecting this verifier to motion control. Do not lower confidence or confirmation thresholds merely to force accepts.

## Artifacts

- [Adjusted full replay](rtx5080-adjusted/summary.json), [per-window results](rtx5080-adjusted/predictions.jsonl).
- [Original baseline](rtx5080/summary.json).
- [Rendering/prompt ablation](diagnostics/predictions.jsonl), [rejected postcondition ablation](diagnostics-postcondition/predictions.jsonl).
- Local viewer: http://127.0.0.1:8765 ; raw videos and annotated critic-comparison.mp4 are in outputs/rtx5080-adjusted.

Accepted-schedule check: 11 windows, [{"phase": "step1_pipette_pick", "accepted_source_s": 9.75, "pair_latency_ms": 952.8647889965214}]; stopped on accepted failure: True. See [FC fixture result](rtx5080-fc/summary.json). This is not a five-stage successful execution.

Verification: original 18 unit tests and mock replay passed; V2 10 unit tests passed. Browser smoke checks passed for both decoded videos, seeking, telemetry, synchronized playback and rewind. Annotated MP4 responds with HTTP 206 for byte-range requests. Local API is bound to 127.0.0.1:8082 and viewer to 127.0.0.1:8765; no robot is connected.
