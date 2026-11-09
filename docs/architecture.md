# Runtime architecture

The hot path is intentionally small:

```text
FC snapshot (3 cameras × 8 frames)
        ↓ one synchronous request; no queue and no retry
Qwen monitor → {status, confidence}
        ↓ strict parser; transport errors become low-confidence in_progress
two-window terminal confirmer
        ↓
FC response: continue / advance / stop / complete
```

FC supplies `expected_phase`; the phase name is intent, never proof. The system prompt contains only
the active phase's success/failure rubric. Frame timestamps and IDs are not inserted before images,
so the static system/task prefix can be reused by vLLM prefix caching. Frames are camera-major and
oldest-to-newest within each camera.

The model wire schema has only `status` and `confidence`. Phase, episode identity, accepted completion,
terminal state and latency metrics are deterministic controller-side fields. This avoids spending decode
time on explanations or error taxonomies that FC does not need in its real-time loop.

With one vote, stochastic majority voting is replaced by two controls: deterministic decoding
(`temperature=0`) and confirmation of the same high-confidence `success` or `fail` on two consecutive,
distinct windows. `in_progress`, low confidence, phase changes and backend errors reset the pending
terminal streak. Accepted completion persists across later occlusion; two accepted failure observations
for that phase revoke it.

The HTTP adapter is deliberately serial. If a request is in flight, the caller should skip the next tick
rather than queue another snapshot. Production camera ingestion and FC transport remain outside this
repository.
