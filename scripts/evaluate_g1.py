"""Evaluate the observer on sampled frames from the public G1 LeRobot dataset."""
import argparse
import base64
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import time

from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from robot_agent.agent import ObserverAgent
from robot_agent.backends import create_backend
from robot_agent.config import load_config
from robot_agent.evaluate import evaluate

REPO = "jren313/g1-pipette-2view-teleop0925-5task-eerel"
CAMERAS = ("observation.images.rgb", "observation.images.wrist_left")
PHASES = ("step1_pipette_pick", "step2_tube_pick", "step3_pipette_aim_at_tube",
          "step4_tube_return", "step5_pipette_return")


def image_url(tensor, max_side):
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    if array.dtype != "uint8":
        array = (array.clip(0, 1) * 255).astype("uint8")
    image = Image.fromarray(array)
    image.thumbnail((max_side, max_side))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g1_pipette.toml")
    parser.add_argument("--sample-hz", type=float, default=1.0)
    parser.add_argument("--max-image-side", type=int, default=640)
    parser.add_argument("--history-size", type=int)
    parser.add_argument("--views", choices=("both", "head", "wrist"), default="both")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/g1"))
    args = parser.parse_args()
    if not math.isfinite(args.sample_hz) or not 0 < args.sample_hz <= 60:
        parser.error("--sample-hz must be in (0, 60]")
    if args.max_image_side < 64:
        parser.error("--max-image-side must be at least 64")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    selected_episodes = list(range(args.max_episodes)) if args.max_episodes is not None else None
    dataset = LeRobotDataset(REPO, episodes=selected_episodes)
    camera_keys = set(dataset.meta.camera_keys)
    if not set(CAMERAS) <= camera_keys:
        raise ValueError(f"dataset is missing cameras: {set(CAMERAS) - camera_keys}")
    config = load_config(args.config)
    if args.history_size is not None:
        if args.history_size < 1:
            parser.error("--history-size must be positive")
        config = replace(config, history_size=args.history_size)
    backend = create_backend(config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cameras = CAMERAS if args.views == "both" else (CAMERAS[0],) if args.views == "head" else (CAMERAS[1],)
    predictions, labels = [], []
    episodes = dataset.num_episodes
    stride = max(1, round(dataset.fps / args.sample_hz))
    for episode_index in range(episodes):
        start = int(dataset.episode_data_index["from"][episode_index])
        end = int(dataset.episode_data_index["to"][episode_index])
        task_index = int(dataset[start]["task_index"])
        phase_index = task_index // 3
        if phase_index not in range(len(PHASES)):
            raise ValueError(f"unexpected task_index {task_index} in episode {episode_index}")
        phase = PHASES[phase_index]
        agent = ObserverAgent(config, backend)
        sampled = list(range(start, end, stride))
        if not sampled or sampled[-1] != end - 1:
            sampled.append(end - 1)
        completed_at = max(0, len(sampled) - config.completion_confirmations)
        for offset, global_index in enumerate(sampled):
            started = time.perf_counter()
            frame = dataset[global_index]
            frame_id = f"{episode_index}:{global_index - start}"
            observation = {"episode_id": str(episode_index), "frame_id": frame_id,
                           "timestamp_s": float(frame["timestamp"]),
                           "images": [image_url(frame[key], args.max_image_side) for key in cameras]}
            result = agent.observe(observation)
            e2e_ms = (time.perf_counter() - started) * 1000
            result["metrics"]["end_to_end_latency_ms"] = e2e_ms
            result["metrics"]["over_budget"] = e2e_ms > 1000 / config.target_hz
            predictions.append(result)
            labels.append({"episode_id": str(episode_index), "frame_id": frame_id,
                           "phase": phase,
                           "status": "completed" if offset >= completed_at else "in_progress"})
        print(f"episode {episode_index + 1}/{episodes}: {phase}, {len(sampled)} frames", flush=True)
    for name, rows in (("predictions.jsonl", predictions), ("labels.jsonl", labels)):
        with (args.out_dir / name).open("w") as output:
            for row in rows:
                output.write(json.dumps(row) + "\n")
    report = evaluate(predictions, labels, config.phases)
    report.update({"dataset": REPO, "sample_hz": args.sample_hz,
                   "episodes": episodes, "views": args.views, "image_max_side": args.max_image_side,
                   "history_size": config.history_size, "images_per_frame": len(cameras)})
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
