# Development rules

## Scope

This repository is a low-latency visual monitor for FC-controlled robot phases. FC owns phase order,
motion, retry and replanning. The monitor verifies only `expected_phase` from a fresh temporal camera
window. One process owns one episode and must be reset before reuse.

## Runtime contract

- Production uses one deterministic model choice and a two-field wire verdict.
- Never add explanations or evidence to the hot-path model output without re-benchmarking latency.
- A backend/parse error is not visual task failure.
- Preserve accepted phase completion until a later accepted contradiction.
- `end_of_episode=true` with missing phases is `INCOMPLETE_AT_END`.
- Step 3 is aiming only: tip visibly above and aligned with the held tube opening. Do not require
  contact, insertion, pressing or liquid transfer.
- Step 4/5 success requires stable target support after gripper release.
- Identify objects by function and initial location, not fixed color or geometry.

## Performance contract

- 3 cameras × 8 chronological frames, 0.25 s spacing, 448×256.
- thinking off, max 32 output tokens, one request, no retry, 8 s timeout.
- Use fresh real images for benchmarks; repeated synthetic images invalidate MM-cache results.
- Distinguish TTFT, full single-vote latency and accepted-decision latency.
- Never report a model Hz when the checkpoint failed to load.

## Verification

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m robot_agent --backend mock replay
```
