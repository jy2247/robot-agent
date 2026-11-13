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
- [dropped pipette failure during phase 2](exports/failure_critic_demo.mp4)

The success export uses G1 chain 5 (five adjacent episode clips) and its recorded five-vote
predictions. The timeline advances only on accepted phase completions. Its confidence is the
recorded confidence for the current expected-phase verdict, not a probability over all five phases.
Rebuild that export with `PYTHONPATH=src /data/jliu3292/robot-agent-vllm/bin/python scripts/build_full_phase_demo.py`.

The available VEGA failure recording starts in phase 2. Its phase critic stayed `in_progress`;
the global object-state critic detected the drop. No phase 1 VEGA recording or predictions were
available to show a real phase 1-to-2 transition in that failure export.

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
