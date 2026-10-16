# Development context for Codex

## Goal
Build a temporal VLM observer/critic for Flow Conductor (FC) coordinating VLA execution.
Recognize the current phase, verify completion, report failures and supporting evidence.
Do not confuse this repository with the separate STL/RoboCasa course project.

## Baseline and boundaries
- Python >=3.11, src layout; core currently has no third-party runtime dependencies.
- Keep mock runnable on a CPU without model weights.
- Model serving is a separate process/environment. Use actual server model names from configuration.
- Target 3–5 Hz is an unverified performance objective, not an achieved benchmark.
- FC controls transitions/replanning/actuation; this package only reports observations/recommendations.
- One server instance = one active episode; reset before reuse; no cross-episode memory leakage.
- Treat model output as untrusted: validate fields, fall back to uncertain on errors.
- Do not replace backend failures with mock predictions.
- History storage is not model KV caching. Prefix caching must be measured in the serving engine.
- Do not commit credentials, datasets, recordings, logs, weights or local environment directories.

## Where to work
- Agent state and confirmation: src/robot_agent/agent.py
- Model interface: src/robot_agent/backends/
- Temporal prompt: src/robot_agent/context.py and prompts/observer.txt
- Input/output contract: src/robot_agent/schemas.py
- FC development transport: src/robot_agent/server.py
- Next work: docs/roadmap.md

## Verification
Run from repository root:
```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m robot_agent replay --output outputs/smoke.jsonl
```
For model integration, first inspect available GPU resources and actual installed software.
Ask for the deployed model endpoint or local model path if absent; never invent a checkpoint ID.
Distinguish mock/integration/real robot evaluation in every report.
Update docs when changing the API. Add targeted tests for state and failure behavior.
