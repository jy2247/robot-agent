# Why the five-stage replay accepted no phase

Reviewed the actual 448×256 JPEG inputs reconstructed from the same replay videos, including the complete sixteen-image windows ending at 24.75 s and 50.75 s. These are visual-review judgments, not independently labeled ground truth.

## What the frames show versus the recorded verdict

| Source time | Visible evidence | Phase output | Companion support output |
|---|---|---|---|
| 9.75 s | Pipette is held clear of its holder. | in_progress, 0.9 | pipette=hand |
| 18.75 s | Left view shows the handled tube lifted above its rack. | in_progress, 0.9 | tube=rack |
| 24.75 s | Left view shows a held tube with the pipette tip aligned at its opening, throughout the inspected 23.00–24.75 s window. This satisfies the aiming-only intent; transfer is not required. | in_progress, 0.9 | tube=rack |
| 35.75 s | Tube appears seated in the rack with the hand open nearby. Full-window release/stability timing needs closer annotation. | in_progress, 0.8 | tube=rack |
| 49.75 s | Pipette supported in holder, gripper withdrawn. | success, 1.0 | pipette=holder |
| 50.75 s | The complete 49.00–50.75 s window continues to show stable holder support and withdrawn gripper. | in_progress, 0.9 | pipette=holder |

Evidence images:

- [Newest input frames at six assessment times](../outputs/evidence-review/phase-evidence.jpg)
- [All sixteen input images at 24.75 s](../outputs/evidence-review/window-24.75.jpg)
- [All sixteen input images at 50.75 s](../outputs/evidence-review/window-50.75.jpg)

## Established causes

1. **The model did not supply success votes for phases 1–4.** There is nothing for the two-confirmation gate to accept. Several inspected frames contradict the in_progress judgments; this is not evidence that the physical robot failed.
2. **Phase 5 supplied one success, then contradicted it with in_progress.** At 49.75 s pending confirmations became one; at 50.75 s they reset to zero. No accepted completion existed to preserve. Confidence 0.9 on in_progress is confidence in that status, not 90% phase completion.
3. **The replay uses recorded boundaries to advance expected_phase, independently of model acceptance.** At 1 Hz, only the 9.75 s assessment sees late phase-1 evidence before expected_phase changes at 10.75 s. Even a corrected success at 9.75 s alone would not meet two confirmations. This limitation does not explain the missed stable phase-3 and phase-5 evidence.
4. **Encoder sharing did not change categorical results in the measured A/B test.** All 50 phase statuses and all 50 safety outputs matched the unshared baseline. Five phase confidence values differed. This narrows the sharing hypothesis but does not certify every part of the common inference path.

## Still unproven

The cause of the visual misjudgments is not isolated. Possibilities include target identity confusion, multi-image temporal interpretation, prompt phrasing, and the FP8/kernel/model configuration. The safety prompt describes a "small open centrifuge tube" whereas the handled object is a larger open tube; its repeated rack prediction despite a clearly held target suggests possible tracking of a different object. This is a hypothesis, not a demonstrated root cause.

The acceptance threshold should not be lowered to conceal these failures. A useful next controlled diagnostic is to compare the same model on the existing sixteen-image prompt, explicitly labeled camera/time frames, and newest-only views, without changing the saved benchmark or claiming the latter is a production temporal test. Five votes might improve consistency, but cannot be assumed to fix a systematic visual interpretation error.
