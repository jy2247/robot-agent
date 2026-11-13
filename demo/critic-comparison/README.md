# Robot safety critic comparison demo

This is the editable, buildless presentation template for the success-versus-failure critic demo.

The generated package contains:

- synchronized browser playback for every camera;
- phase and global-safety state updated from the recorded JSONL assessments;
- support-state, confirmation, latency, and action panels;
- standalone annotated MP4 exports;
- raw predictions, summaries, and a machine-readable manifest.

The committed annotated video exports are available directly here:

- [successful tube pick](exports/success_critic_demo.mp4)
- [dropped pipette failure](exports/failure_critic_demo.mp4)

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
