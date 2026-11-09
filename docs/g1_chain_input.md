# G1 five-phase chain input

Dataset: `jren313/g1-pipette-2view-teleop0925-5task-eerel`.

The 2026-10-07 source recordings contain a continuous five-step workflow, but the published dataset
cuts each chain into five adjacent episodes. Each MP4 contains exactly one phase. Episodes 237–241 are
from source session `20261007_1319`, source recording `episode_0002`, and map as follows:

| Episode | Phase | Frames @ 60 Hz |
|---:|---|---:|
| 237 | `step1_pipette_pick` | 626 |
| 238 | `step2_tube_pick` | 550 |
| 239 | `step3_pipette_aim_at_tube` | 465 |
| 240 | `step4_tube_return` | 517 |
| 241 | `step5_pipette_return` | 907 |

The dataset has two live views: `observation.images.rgb` (1280×720) and
`observation.images.wrist_left` (1080×1080). It does not provide a third live camera. Evaluation must
record `camera_count=2`; production 3-view latency remains a separate benchmark.

Download only this chain and its mapping metadata:

```bash
HF_BIN=/path/to/hf bash scripts/download_g1_chain.sh
```

Dataset task rows are used only to select the phase before inference. They are never inserted into the
visual model request as success evidence.
