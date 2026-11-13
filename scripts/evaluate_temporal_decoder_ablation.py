"""Collect a fixed five-vote evidence bank and causally replay temporal decoders."""
import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import time

from evaluate_five_phase_chains import (
    chain_paths, inspect_chain, prepare_chain_windows, transition_estimate,
)
from robot_agent.agent import MonitorAgent
from robot_agent.backends import create_backend
from robot_agent.config import load_config
from robot_agent.sequence import MonotonicEvidenceDecoder


STRONG_EVIDENCE_MASS = 0.45
FALSE_TRANSITION_EARLY_S = 2.0


def vote_masses(votes):
    masses = {"success": 0.0, "fail": 0.0, "in_progress": 0.0}
    counts = Counter()
    for vote in votes:
        verdict = vote.get("verdict")
        if verdict:
            counts[verdict["status"]] += 1
            masses[verdict["status"]] += verdict["confidence"] / 5
    return masses, dict(counts)


def collect_evidence(config, dataset_root, first_episode, chain_index, assessment_hz,
                     evidence_path, quiet):
    paths = chain_paths(dataset_root, first_episode)
    _, fps, frame_counts, truth = inspect_chain(paths)
    duration_s = sum(frame_counts) / fps
    first_end_s = (config.frames_per_camera - 1) * config.frame_interval_s
    cadence_s = 1 / assessment_hz
    end_times = []
    timestamp = first_end_s
    while timestamp < duration_s:
        end_times.append(timestamp)
        timestamp += cadence_s

    preparation_started = time.perf_counter()
    windows = prepare_chain_windows(paths, frame_counts, fps, end_times, config)
    preparation_s = time.perf_counter() - preparation_started
    agents = [(MonitorAgent(config, create_backend(config))) for _ in config.phases]
    rows = []
    run_started = time.perf_counter()
    with evidence_path.open("w") as output:
        for window_index, (source_time_s, images) in enumerate(zip(end_times, windows)):
            phase_evidence = {}
            for phase_index, (phase, agent) in enumerate(zip(config.phases, agents)):
                result = agent.observe({
                    "episode_id": f"evidence-{first_episode:06d}-{phase}",
                    "frame_id": f"window-{window_index:04d}",
                    "timestamp_s": source_time_s,
                    "expected_phase": phase,
                    "images": images,
                    "end_of_episode": window_index == len(end_times) - 1,
                })
                masses, counts = vote_masses(result["votes"])
                phase_evidence[phase] = {
                    "status": result["status"],
                    "confidence": result["confidence"],
                    "backend_error_type": result["backend_error_type"],
                    "terminal_confirmation_count": result["terminal_confirmation_count"],
                    "terminal_accepted": result["terminal_accepted"],
                    "votes": result["votes"],
                    "vote_masses": masses,
                    "vote_counts": counts,
                    "latency_ms": result["metrics"]["agent_latency_ms"],
                }
            row = {
                "chain_index": chain_index,
                "source_time_s": round(source_time_s, 3),
                "window_index": window_index,
                "phases": phase_evidence,
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            if not quiet and window_index % 10 == 0:
                print(json.dumps({"chain": chain_index, "window": window_index,
                                  "time_s": round(source_time_s, 2)}), flush=True)
    return {
        "chain_index": chain_index,
        "first_episode": first_episode,
        "frame_counts": frame_counts,
        "fps": fps,
        "duration_s": duration_s,
        "ground_truth_transition_times_s": truth,
        "assessment_count": len(rows),
        "input_preparation_s": preparation_s,
        "evidence_collection_s": time.perf_counter() - run_started,
        "evidence_path": str(evidence_path),
    }, rows


def load_evidence(evidence_path):
    return [json.loads(line) for line in evidence_path.open()]


def evidence_health(evidence_sets):
    assessments = 0
    votes = 0
    vote_errors = Counter()
    backend_errors = Counter()
    for _, rows in evidence_sets:
        for row in rows:
            for evidence in row["phases"].values():
                assessments += 1
                if evidence["backend_error_type"]:
                    backend_errors[evidence["backend_error_type"]] += 1
                for vote in evidence["votes"]:
                    votes += 1
                    if vote.get("error"):
                        vote_errors[vote["error"]] += 1
    return {
        "causal_windows": sum(len(rows) for _, rows in evidence_sets),
        "phase_assessments": assessments,
        "individual_votes": votes,
        "backend_error_count": sum(backend_errors.values()),
        "backend_errors_by_type": dict(backend_errors),
        "vote_error_count": sum(vote_errors.values()),
        "vote_errors_by_type": dict(vote_errors),
    }


def result_from_evidence(evidence):
    return {
        "status": evidence["status"],
        "confidence": evidence["confidence"],
        "backend_error_type": evidence["backend_error_type"],
        "votes": evidence["votes"],
    }


def replay_baseline(rows, phases, cadence_s):
    phase_index = 0
    pending = None
    pending_count = 0
    last_nonterminal_s = None
    first_evidence_s = None
    completed = []
    transitions = []
    for row in rows:
        timestamp_s = row["source_time_s"]
        phase = phases[phase_index]
        evidence = row["phases"][phase]
        counts = evidence["vote_counts"]
        status = evidence["status"]
        agreeing = counts.get(status, 0) if status in {"success", "fail"} else 0
        key = (phase, status) if agreeing >= 3 else None
        if key is not None and key == pending:
            pending_count += 1
        elif key is not None:
            pending, pending_count = key, 1
        else:
            pending, pending_count = None, 0
        if key is not None and first_evidence_s is None:
            first_evidence_s = timestamp_s
        elif key is None:
            last_nonterminal_s = timestamp_s
            first_evidence_s = None
        accepted = bool(key and (agreeing >= 4 or pending_count >= 2))
        if accepted and status == "success":
            completed.append(phase)
            if phase_index < len(phases) - 1:
                evidence_time_s = first_evidence_s or timestamp_s
                transitions.append({
                    "from_phase": phase,
                    "to_phase": phases[phase_index + 1],
                    "accepted_time_s": timestamp_s,
                    "first_evidence_time_s": evidence_time_s,
                    "last_nonterminal_time_s": last_nonterminal_s,
                    "bracket_midpoint_time_s": transition_estimate(
                        last_nonterminal_s, evidence_time_s, cadence_s),
                })
                phase_index += 1
                pending, pending_count = None, 0
                last_nonterminal_s = timestamp_s
                first_evidence_s = None
            else:
                break
    return transitions, len(completed) == len(phases)


def replay_monotonic(rows, phases, cadence_s, parameters):
    decoder = MonotonicEvidenceDecoder(phases, cadence_s, **parameters)
    transitions = []
    for row in rows:
        evidence = row["phases"][decoder.phase]
        transition, _, _ = decoder.update(
            row["source_time_s"], result_from_evidence(evidence))
        if transition and transition["to_phase"] is not None:
            transitions.append(transition)
        if decoder.finished:
            break
    return transitions, decoder.finished


def first_strong_after_ground_truth(rows, phase, ground_truth_s):
    for row in rows:
        if row["source_time_s"] < ground_truth_s:
            continue
        if row["phases"][phase]["vote_masses"]["success"] >= STRONG_EVIDENCE_MASS:
            return row["source_time_s"]
    return None


def score_replay(name, chain_replays):
    details = []
    full_chain_successes = 0
    for metadata, rows, transitions, complete in chain_replays:
        truth = metadata["ground_truth_transition_times_s"]
        boundary_rows = []
        for boundary_index, (transition, target_s) in enumerate(zip(transitions, truth)):
            phase = transition["from_phase"]
            strong_s = first_strong_after_ground_truth(rows, phase, target_s)
            commit_s = transition["accepted_time_s"]
            estimate_s = transition["bracket_midpoint_time_s"]
            boundary_rows.append({
                "boundary_index": boundary_index + 1,
                "from_phase": phase,
                "to_phase": transition["to_phase"],
                "ground_truth_time_s": target_s,
                "first_strong_vlm_evidence_time_s": strong_s,
                "decoder_commit_time_s": commit_s,
                "estimated_boundary_time_s": estimate_s,
                "perception_delay_s": None if strong_s is None else strong_s - target_s,
                "decoder_delay_s": None if strong_s is None else commit_s - strong_s,
                "online_detection_latency_s": commit_s - target_s,
                "boundary_signed_error_s": estimate_s - target_s,
                "boundary_absolute_error_s": abs(estimate_s - target_s),
                "false_transition": commit_s < target_s - FALSE_TRANSITION_EARLY_S,
            })
        details.append({
            "chain_index": metadata["chain_index"],
            "episode_range": [metadata["first_episode"], metadata["first_episode"] + 4],
            "full_chain_correct": complete,
            "detected_transition_count": len(transitions),
            "boundaries": boundary_rows,
        })
        full_chain_successes += complete

    boundaries = [boundary for chain in details for boundary in chain["boundaries"]]
    perception_delays = [row["perception_delay_s"] for row in boundaries
                         if row["perception_delay_s"] is not None]
    decoder_delays = [row["decoder_delay_s"] for row in boundaries
                      if row["decoder_delay_s"] is not None]
    return {
        "name": name,
        "transition_count_detected": len(boundaries),
        "transition_count_total": 4 * len(details),
        "transition_recall": len(boundaries) / (4 * len(details)),
        "boundary_mae_s_conditional_on_detection": (
            statistics.mean(row["boundary_absolute_error_s"] for row in boundaries)
            if boundaries else None),
        "boundary_mean_signed_error_s": (
            statistics.mean(row["boundary_signed_error_s"] for row in boundaries)
            if boundaries else None),
        "online_detection_latency_mean_s": (
            statistics.mean(row["online_detection_latency_s"] for row in boundaries)
            if boundaries else None),
        "perception_delay_mean_s": (statistics.mean(perception_delays)
                                    if perception_delays else None),
        "decoder_delay_mean_s": (statistics.mean(decoder_delays)
                                 if decoder_delays else None),
        "false_transition_count": sum(row["false_transition"] for row in boundaries),
        "full_chain_accuracy": full_chain_successes / len(details),
        "chains": details,
    }


def variants():
    full = {
        "soft_duration_center_s": 4.0,
        "soft_duration_scale_s": 0.8,
        "decay": 0.65,
        "threshold": 0.45,
        "use_duration_prior": True,
    }
    return {
        "monotonic_full": full,
        "decay_0.00": {**full, "decay": 0.0},
        "decay_0.35": {**full, "decay": 0.35},
        "decay_0.85": {**full, "decay": 0.85},
        "no_duration_prior": {**full, "use_duration_prior": False},
        "duration_center_2s": {**full, "soft_duration_center_s": 2.0},
        "duration_center_6s": {**full, "soft_duration_center_s": 6.0},
        "threshold_0.35": {**full, "threshold": 0.35},
        "threshold_0.55": {**full, "threshold": 0.55},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/monitor.toml")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--first-episode", type=int, default=237)
    parser.add_argument("--chains", type=int, default=5)
    parser.add_argument("--assessment-hz", type=float, default=2.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-evidence", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.chains != 5:
        parser.error("this controlled ablation requires exactly five test chains")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = replace(load_config(args.config), votes=5, temperature=0.7,
                     camera_count=2, assessment_hz=args.assessment_hz)
    cadence_s = 1 / args.assessment_hz

    evidence_sets = []
    for chain_offset in range(args.chains):
        chain_index = chain_offset + 1
        first_episode = args.first_episode + chain_offset * 5
        evidence_path = args.output_dir / f"chain-{chain_index:02d}-evidence.jsonl"
        metadata_path = args.output_dir / f"chain-{chain_index:02d}-metadata.json"
        if args.reuse_evidence and evidence_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            rows = load_evidence(evidence_path)
        else:
            metadata, rows = collect_evidence(
                config, args.dataset_root, first_episode, chain_index,
                args.assessment_hz, evidence_path, args.quiet)
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        evidence_sets.append((metadata, rows))
        print(json.dumps({"evidence_chain_complete": chain_index,
                          "windows": len(rows)}), flush=True)

    results = []
    baseline_replays = []
    for metadata, rows in evidence_sets:
        transitions, complete = replay_baseline(rows, config.phases, cadence_s)
        baseline_replays.append((metadata, rows, transitions, complete))
    results.append(score_replay("current_baseline", baseline_replays))

    for name, parameters in variants().items():
        replays = []
        for metadata, rows in evidence_sets:
            transitions, complete = replay_monotonic(
                rows, config.phases, cadence_s, parameters)
            replays.append((metadata, rows, transitions, complete))
        result = score_replay(name, replays)
        result["parameters"] = parameters
        results.append(result)

    output = {
        "protocol": {
            "strictly_causal": True,
            "future_observations_available_to_decoder": False,
            "ground_truth_used_only_for_post_replay_scoring": True,
            "fixed_vlm_evidence_shared_by_all_decoders": True,
            "chains": args.chains,
            "episode_range": [args.first_episode, args.first_episode + args.chains * 5 - 1],
            "assessment_hz": args.assessment_hz,
            "votes": 5,
            "strong_evidence_success_mass": STRONG_EVIDENCE_MASS,
            "false_transition_definition": (
                f"decoder commit more than {FALSE_TRANSITION_EARLY_S}s before ground truth"),
        },
        "evidence_health": evidence_health(evidence_sets),
        "evidence_metadata": [metadata for metadata, _ in evidence_sets],
        "results": results,
    }
    (args.output_dir / "ablation_summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps([{key: result[key] for key in (
        "name", "transition_recall", "boundary_mae_s_conditional_on_detection",
        "online_detection_latency_mean_s", "false_transition_count",
        "full_chain_accuracy")}
        for result in results], indent=2))


if __name__ == "__main__":
    main()
