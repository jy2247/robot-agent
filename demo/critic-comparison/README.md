# Robot safety critic comparison demo

This is the editable, buildless presentation template for the success-versus-failure critic demo.

The generated package contains:

- synchronized browser playback for every camera;
- phase and global-safety state updated from the recorded JSONL assessments;
- support-state, confirmation, latency, and action panels;
- standalone annotated MP4 exports;
- raw predictions, summaries, and a machine-readable manifest.

The committed annotated video exports are available directly here:

- [full five-phase success with confirmed transitions and recorded confidence](exports/success_critic_demo.mp4)
- [VEGA phase 2 dropped-pipette failure](exports/failure_critic_demo.mp4)

The success export uses G1 chain 5 (five adjacent episode clips) and its recorded five-vote
predictions. The timeline advances only on accepted phase completions. Its confidence is the
recorded confidence for the current expected-phase verdict, not a probability over all five phases.
Rebuild that export with `PYTHONPATH=src /data/jliu3292/robot-agent-vllm/bin/python scripts/build_full_phase_demo.py`.

The failure export uses only the original VEGA phase 2 camera recording. Its phase critic stayed
`in_progress`; the global object-state critic detected the drop. Phase 1 is marked "prior / not shown"
because this recording starts in phase 2; phases 3–5 were not reached. The original video is 15.23 s,
followed by a 3 s held final frame. Recorded assessments from the former 10 s hold are replayed in
order across that 3 s hold; displayed time during the hold is edited playback time, not original
inference cadence.
Rebuild with `PYTHONPATH=src /data/jliu3292/robot-agent-vllm/bin/python scripts/build_failure_phase_demo.py`.

Rebuild the package after editing this template:

```bash
PYTHONPATH=src /data/jliu3292/robot-agent-vllm/bin/python scripts/build_critic_demo.py
```

Preview the generated dashboard:

```bash
cd /data/jliu3292/robot-agent-results/critic-demo-success-vs-failure
python3 -m http.server 8765
```

Then open `http://127.0.0.1:8765/`.
