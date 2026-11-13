"""Enumerate continuous five-stage tests from a LeRobot metadata-only dataset."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


PHASES = (
    "step1_pipette_pick",
    "step2_tube_pick",
    "step3_pipette_aim_at_tube",
    "step4_tube_return",
    "step5_pipette_return",
)
SOURCE_STEP = re.compile(r"^(?P<recording>.+)_step(?P<step>[1-5])$")


def read_jsonl(path):
    return [json.loads(line) for line in path.open()]


def expected_video_paths(info, episode_index):
    chunk = episode_index // info["chunks_size"]
    keys = [key for key, feature in info["features"].items()
            if feature.get("dtype") == "video"]
    return [info["video_path"].format(
        episode_chunk=chunk, video_key=key, episode_index=episode_index)
        for key in keys]


def enumerate_tests(metadata_dir):
    dataset_root = metadata_dir.parent
    info = json.loads((metadata_dir / "info.json").read_text())
    notes = json.loads((metadata_dir / "source_notes.json").read_text())
    episodes = read_jsonl(metadata_dir / "episodes.jsonl")
    provenance = read_jsonl(metadata_dir / "source_provenance.jsonl")
    if len(episodes) != len(provenance):
        raise ValueError("episodes and source provenance have different lengths")
    goals = [notes["task_text"][phase]["goal"] for phase in PHASES]

    tests = []
    rejected_candidates = []
    for start in range(len(episodes) - len(PHASES) + 1):
        episode_group = episodes[start:start + len(PHASES)]
        provenance_group = provenance[start:start + len(PHASES)]
        if [row["tasks"][0] for row in episode_group] != goals:
            continue
        indices = [row["episode_index"] for row in episode_group]
        reasons = []
        if indices != list(range(indices[0], indices[0] + len(PHASES))):
            reasons.append("episode indices are not consecutive")
        sessions = {row["source_session"] for row in provenance_group}
        if len(sessions) != 1:
            reasons.append("source sessions differ")
        parsed = [SOURCE_STEP.fullmatch(row["source_episode"])
                  for row in provenance_group]
        if not all(parsed):
            reasons.append("source episode names do not encode step1..step5")
            recording_ids = []
            source_steps = []
        else:
            recording_ids = [match.group("recording") for match in parsed]
            source_steps = [int(match.group("step")) for match in parsed]
            if len(set(recording_ids)) != 1:
                reasons.append("stages came from different source recordings")
            if source_steps != list(range(1, len(PHASES) + 1)):
                reasons.append("source steps are not ordered step1..step5")
        labels = {row.get("source_label") for row in provenance_group}
        if labels != {"successful"}:
            reasons.append("not every source stage is labelled successful")
        if reasons:
            rejected_candidates.append({"start_episode": indices[0],
                                        "reasons": reasons})
            continue

        stage_rows = []
        elapsed_frames = 0
        transition_times_s = []
        for phase, episode, source in zip(PHASES, episode_group,
                                          provenance_group):
            episode_index = episode["episode_index"]
            stage_rows.append({
                "phase": phase,
                "episode_index": episode_index,
                "frames": episode["length"],
                "duration_s": episode["length"] / info["fps"],
                "source_episode": source["source_episode"],
                "expected_video_paths": expected_video_paths(info, episode_index),
            })
            elapsed_frames += episode["length"]
            if phase != PHASES[-1]:
                transition_times_s.append(elapsed_frames / info["fps"])
        tests.append({
            "test_index": len(tests) + 1,
            "source_session": provenance_group[0]["source_session"],
            "source_recording": recording_ids[0],
            "episode_range": [indices[0], indices[-1]],
            "total_frames": elapsed_frames,
            "duration_s": elapsed_frames / info["fps"],
            "ground_truth_transition_times_s": transition_times_s,
            "stages": stage_rows,
        })

    used_episodes = {stage["episode_index"] for test in tests
                     for stage in test["stages"]}
    expected_paths = [path for test in tests for stage in test["stages"]
                      for path in stage["expected_video_paths"]]
    locally_available = [path for path in expected_paths
                         if (dataset_root / path).is_file()]
    return {
        "dataset_total_episodes": len(episodes),
        "dataset_total_videos": info["total_videos"],
        "full_five_stage_test_count": len(tests),
        "episodes_in_full_tests": len(used_episodes),
        "videos_in_full_tests": len(expected_paths),
        "full_test_videos_available_locally": len(locally_available),
        "full_test_videos_missing_locally": (
            len(expected_paths) - len(locally_available)),
        "episodes_not_in_full_tests": len(episodes) - len(used_episodes),
        "source_labels": dict(Counter(
            row.get("source_label", "missing") for row in provenance)),
        "tests_by_source_session": dict(Counter(
            test["source_session"] for test in tests)),
        "rejected_ordered_candidates": rejected_candidates,
        "tests": tests,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = enumerate_tests(args.metadata_dir)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"tests", "rejected_ordered_candidates"}},
                     indent=2))


if __name__ == "__main__":
    main()
