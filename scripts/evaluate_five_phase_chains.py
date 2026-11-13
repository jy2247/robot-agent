"""Evaluate reconstructed five-phase G1 chains with the five-vote monitor."""
import argparse
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
import statistics
import time

import cv2

from benchmark_single_vote import encode, latency_summary, video_info
from robot_agent.agent import MonitorAgent
from robot_agent.backends import create_backend
from robot_agent.config import load_config
from robot_agent.sequence import MonotonicEvidenceDecoder


CAMERAS = ("observation.images.rgb", "observation.images.wrist_left")


def chain_paths(root, first_episode):
    base = root / "videos/chunk-000"
    return {
        camera: [base / camera / f"episode_{episode:06d}.mp4"
                 for episode in range(first_episode, first_episode + 5)]
        for camera in CAMERAS
    }


def inspect_chain(paths):
    per_camera = {camera: [video_info(path) for path in camera_paths]
                  for camera, camera_paths in paths.items()}
    reference = per_camera[CAMERAS[0]]
    fps = reference[0]["fps"]
    frame_counts = [item["frames"] for item in reference]
    for camera, items in per_camera.items():
        if any(not math.isclose(item["fps"], fps, rel_tol=0, abs_tol=1e-3)
               for item in items):
            raise RuntimeError(f"mixed frame rates in {camera}")
        if [item["frames"] for item in items] != frame_counts:
            raise RuntimeError(f"camera frame-count mismatch in {camera}")
    boundaries = []
    cumulative = 0
    for count in frame_counts[:-1]:
        cumulative += count
        boundaries.append(cumulative / fps)
    return per_camera, fps, frame_counts, boundaries


def prepare_chain_windows(paths, frame_counts, fps, end_times, config):
    """Decode adjacent episode MP4s as one gap-free continuous timeline."""
    indexes_by_window = [
        [round((end_s - offset * config.frame_interval_s) * fps)
         for offset in reversed(range(config.frames_per_camera))]
        for end_s in end_times
    ]
    needed = {index for indexes in indexes_by_window for index in indexes}
    per_camera = []
    for camera_index, camera in enumerate(CAMERAS):
        encoded = {}
        global_offset = 0
        for segment_path, segment_frames in zip(paths[camera], frame_counts):
            capture = cv2.VideoCapture(str(segment_path))
            if not capture.isOpened():
                raise RuntimeError(f"cannot open {segment_path}")
            for local_index in range(segment_frames):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"cannot decode {segment_path} at {local_index}")
                global_index = global_offset + local_index
                if global_index not in needed:
                    continue
                if camera_index == 1:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                encoded[global_index] = encode(
                    frame, config.image_width, config.image_height)
            capture.release()
            global_offset += segment_frames
        missing = needed.difference(encoded)
        if missing:
            raise RuntimeError(f"missing {len(missing)} frames from {camera}")
        per_camera.append([[encoded[index] for index in indexes]
                           for indexes in indexes_by_window])
    return [[url for camera in per_camera for url in camera[window_index]]
            for window_index in range(len(end_times))]


def terminal_evidence(result):
    if result["status"] not in {"success", "fail"}:
        return False
    if result["backend_error_type"] is not None:
        return False
    return result["terminal_confirmation_count"] > 0


def transition_estimate(last_nonterminal_s, first_evidence_s, cadence_s):
    """Estimate an event inside its last-negative/first-positive time bracket."""
    lower = (first_evidence_s - cadence_s if last_nonterminal_s is None
             else last_nonterminal_s)
    return (lower + first_evidence_s) / 2


def evaluate_chain(config, paths, chain_index, first_episode, output_dir, assessment_hz,
                   decoder_name, quiet):
    per_camera, fps, frame_counts, truth = inspect_chain(paths)
    duration_s = sum(frame_counts) / fps
    first_end_s = (config.frames_per_camera - 1) * config.frame_interval_s
    cadence_s = 1 / assessment_hz
    end_times = []
    timestamp = first_end_s
    while timestamp < duration_s:
        end_times.append(timestamp)
        timestamp += cadence_s
    preparation_started = time.perf_counter()
    windows = prepare_chain_windows(
        paths, frame_counts, fps, end_times, config)
    preparation_s = time.perf_counter() - preparation_started

    backend = create_backend(config)
    agent = MonitorAgent(config, backend)
    monotonic_decoder = (MonotonicEvidenceDecoder(config.phases, cadence_s)
                         if decoder_name == "monotonic" else None)
    phase_index = 0
    phase = config.phases[phase_index]
    last_nonterminal_s = None
    candidate_evidence_s = None
    transitions = []
    completion_times = {}
    rows = []
    predictions_path = output_dir / f"chain-{chain_index:02d}-predictions.jsonl"
    run_started = time.perf_counter()
    with predictions_path.open("w") as output:
        for window_index, (source_time_s, images) in enumerate(zip(end_times, windows)):
            is_last = window_index == len(end_times) - 1
            result = agent.observe({
                "episode_id": f"g1-chain-{first_episode:06d}",
                "frame_id": f"window-{window_index:04d}",
                "timestamp_s": source_time_s,
                "expected_phase": phase,
                "images": images,
                "end_of_episode": is_last,
            })
            transition = None
            vote_masses = None
            posterior_score = None
            if monotonic_decoder is not None:
                transition, vote_masses, posterior_score = monotonic_decoder.update(
                    source_time_s, result)
                if transition is not None:
                    completion_times.setdefault(result["phase"], source_time_s)
                    if transition["to_phase"] is not None:
                        transitions.append(transition)
                    phase = monotonic_decoder.phase
            else:
                evidence = terminal_evidence(result)
                if evidence and candidate_evidence_s is None:
                    candidate_evidence_s = source_time_s
                elif not evidence:
                    last_nonterminal_s = source_time_s
                    candidate_evidence_s = None

            if (monotonic_decoder is None and result["terminal_accepted"]
                    and result["status"] == "success"):
                completion_times.setdefault(phase, source_time_s)
                if phase_index < len(config.phases) - 1:
                    first_evidence_s = (candidate_evidence_s if candidate_evidence_s is not None
                                        else source_time_s)
                    transition = {
                        "from_phase": phase,
                        "to_phase": config.phases[phase_index + 1],
                        "accepted_time_s": source_time_s,
                        "first_evidence_time_s": first_evidence_s,
                        "last_nonterminal_time_s": last_nonterminal_s,
                        "bracket_midpoint_time_s": transition_estimate(
                            last_nonterminal_s, first_evidence_s, cadence_s),
                    }
                    transitions.append(transition)
                    phase_index += 1
                    phase = config.phases[phase_index]
                    last_nonterminal_s = source_time_s
                    candidate_evidence_s = None

            row = {
                "assessment_index": window_index,
                "source_time_s": round(source_time_s, 3),
                "expected_phase_before_result": result["phase"],
                "transition": transition,
                "decoder": decoder_name,
                "decoder_vote_masses": vote_masses,
                "decoder_posterior_score": posterior_score,
                "raw_model_output": getattr(backend, "last_raw", None),
                **result,
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            if not quiet or transition is not None or is_last:
                print(json.dumps({
                    "chain": chain_index,
                    "index": window_index,
                    "time_s": round(source_time_s, 2),
                    "phase": result["phase"],
                    "status": result["status"],
                    "agent_accepted": result["terminal_accepted"],
                    "decoder_transition": transition is not None,
                    "next_phase": phase,
                    "latency_ms": round(result["metrics"]["agent_latency_ms"], 1),
                }), flush=True)

    run_wall_s = time.perf_counter() - run_started
    raw_errors = [transition["accepted_time_s"] - target
                  for transition, target in zip(transitions, truth)]
    midpoint_errors = [transition["bracket_midpoint_time_s"] - target
                       for transition, target in zip(transitions, truth)]
    valid = [row for row in rows if row["backend_error_type"] is None]
    inference_ms = [row["metrics"]["inference_and_parse_ms"] for row in valid]
    final_result = rows[-1]
    if monotonic_decoder is not None:
        detected_all = len(monotonic_decoder.completed) == len(config.phases)
    else:
        detected_all = len(transitions) == 4 and all(
            phase in completion_times for phase in config.phases)
    summary = {
        "decoder": decoder_name,
        "chain_index": chain_index,
        "first_episode": first_episode,
        "episode_range": [first_episode, first_episode + 4],
        "source_videos": per_camera,
        "fps": fps,
        "phase_frame_counts": frame_counts,
        "duration_s": duration_s,
        "ground_truth_transition_times_s": truth,
        "predicted_transitions": transitions,
        "raw_signed_errors_s": raw_errors,
        "midpoint_signed_errors_s": midpoint_errors,
        "raw_mae_s": statistics.mean(map(abs, raw_errors)) if raw_errors else None,
        "midpoint_mae_s": (statistics.mean(map(abs, midpoint_errors))
                            if midpoint_errors else None),
        "phase_completion_times_s": completion_times,
        "detected_all_five_phases": detected_all,
        "ground_truth_outcome": "success",
        "predicted_outcome": "success" if detected_all else "failure_or_incomplete",
        "final_overall_status": final_result["overall_status"],
        "final_overall_reason": final_result["overall_reason"],
        "assessment_count": len(rows),
        "valid_inference_count": len(valid),
        "backend_error_count": len(rows) - len(valid),
        "vote_error_count": sum(
            vote["error"] is not None for row in rows for vote in row.get("votes", [])),
        "input_preparation_s": preparation_s,
        "run_wall_s": run_wall_s,
        "offline_request_loop_hz": len(rows) / run_wall_s,
        "inference": latency_summary(inference_ms),
        "predictions": str(predictions_path),
    }
    return summary


def aggregate(chain_summaries, config, assessment_hz):
    raw_errors = [error for chain in chain_summaries
                  for error in chain["raw_signed_errors_s"]]
    midpoint_errors = [error for chain in chain_summaries
                       for error in chain["midpoint_signed_errors_s"]]
    transition_total = 4 * len(chain_summaries)
    phase_metrics = []
    for phase_index in range(4):
        phase_raw = [chain["raw_signed_errors_s"][phase_index]
                     for chain in chain_summaries
                     if len(chain["raw_signed_errors_s"]) > phase_index]
        phase_midpoint = [chain["midpoint_signed_errors_s"][phase_index]
                          for chain in chain_summaries
                          if len(chain["midpoint_signed_errors_s"]) > phase_index]
        phase_metrics.append({
            "from_phase": config.phases[phase_index],
            "to_phase": config.phases[phase_index + 1],
            "detected_count": len(phase_raw),
            "recall": len(phase_raw) / len(chain_summaries),
            "raw_mae_s_conditional_on_detection": (
                statistics.mean(map(abs, phase_raw)) if phase_raw else None),
            "raw_mean_signed_error_s": statistics.mean(phase_raw) if phase_raw else None,
            "midpoint_mae_s_conditional_on_detection": (
                statistics.mean(map(abs, phase_midpoint)) if phase_midpoint else None),
        })
    return {
        "dataset": "jren313/g1-pipette-2view-teleop0925-5task-eerel",
        "model": config.model,
        "decoder": chain_summaries[0]["decoder"],
        "votes_per_assessment": config.votes,
        "assessment_hz": assessment_hz,
        "camera_count": config.camera_count,
        "chain_count": len(chain_summaries),
        "ground_truth_outcomes": {"success": len(chain_summaries), "failure": 0},
        "five_phase_sequence_accuracy": (
            sum(chain["detected_all_five_phases"] for chain in chain_summaries)
            / len(chain_summaries)),
        "success_recall": (
            sum(chain["predicted_outcome"] == "success" for chain in chain_summaries)
            / len(chain_summaries)),
        "failure_metrics_available": False,
        "transition_count_scored": len(raw_errors),
        "transition_count_total": transition_total,
        "transition_recall": len(raw_errors) / transition_total,
        "raw_transition_mae_s": statistics.mean(map(abs, raw_errors)) if raw_errors else None,
        "raw_transition_mae_s_conditional_on_detection": (
            statistics.mean(map(abs, raw_errors)) if raw_errors else None),
        "midpoint_transition_mae_s": (statistics.mean(map(abs, midpoint_errors))
                                      if midpoint_errors else None),
        "midpoint_transition_mae_s_conditional_on_detection": (
            statistics.mean(map(abs, midpoint_errors)) if midpoint_errors else None),
        "raw_transition_mean_signed_error_s": (statistics.mean(raw_errors)
                                                 if raw_errors else None),
        "midpoint_transition_mean_signed_error_s": (
            statistics.mean(midpoint_errors) if midpoint_errors else None),
        "backend_error_count": sum(chain["backend_error_count"] for chain in chain_summaries),
        "vote_error_count": sum(chain["vote_error_count"] for chain in chain_summaries),
        "phase_transition_metrics": phase_metrics,
        "chains": chain_summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/monitor.toml")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--first-episode", type=int, default=237)
    parser.add_argument("--chains", type=int, default=10)
    parser.add_argument("--assessment-hz", type=float, default=2.5)
    parser.add_argument("--decoder", choices=["baseline", "monotonic"],
                        default="baseline")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.chains <= 0 or args.assessment_hz <= 0:
        parser.error("chains and assessment-hz must be positive")

    config = replace(load_config(args.config), votes=5, temperature=0.7,
                     camera_count=2, assessment_hz=args.assessment_hz)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for chain_index in range(args.chains):
        first_episode = args.first_episode + chain_index * 5
        paths = chain_paths(args.dataset_root, first_episode)
        summary = evaluate_chain(
            config, paths, chain_index + 1, first_episode,
            args.output_dir, args.assessment_hz, args.decoder, args.quiet)
        summaries.append(summary)
        aggregate_summary = aggregate(summaries, config, args.assessment_hz)
        (args.output_dir / "summary.json").write_text(
            json.dumps(aggregate_summary, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({
            "completed_chain": chain_index + 1,
            "detected_all": summary["detected_all_five_phases"],
            "raw_mae_s": summary["raw_mae_s"],
            "midpoint_mae_s": summary["midpoint_mae_s"],
        }, ensure_ascii=False), flush=True)

    print(json.dumps(aggregate(summaries, config, args.assessment_hz),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
