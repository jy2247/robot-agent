"""Run the phase critic with a companion global object-state safety critic."""
import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import time

from benchmark_single_vote import latency_summary, prepare_windows, video_info
from robot_agent.agent import MonitorAgent
from robot_agent.backends import create_backend
from robot_agent.config import load_config
from robot_agent.safety import GlobalSafetyCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/monitor.toml')
    parser.add_argument('--camera', action='append', type=Path, required=True)
    parser.add_argument('--rotate-180-camera', action='append', type=int, default=[])
    parser.add_argument('--phase', required=True)
    parser.add_argument('--assessment-hz', type=float)
    parser.add_argument('--start-s', type=float, default=0.0,
                        help='Begin assessments at this source-video time')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if not 1 <= len(args.camera) <= 3:
        parser.error('one to three --camera arguments are supported')
    config = replace(config, backend='openai', votes=1, temperature=0.0,
                     camera_count=len(args.camera))
    if args.phase not in config.phases:
        parser.error('phase is not configured')
    if any(index < 0 or index >= config.camera_count
           for index in args.rotate_180_camera):
        parser.error('--rotate-180-camera index is out of range')
    assessment_hz = args.assessment_hz or config.assessment_hz
    if not math.isfinite(assessment_hz) or assessment_hz <= 0:
        parser.error('assessment rate must be finite and positive')
    if not math.isfinite(args.start_s) or args.start_s < 0:
        parser.error('start time must be finite and nonnegative')

    videos = [video_info(path) for path in args.camera]
    duration_s = min(item['duration_s'] for item in videos)
    first_end_s = (config.frames_per_camera - 1) * config.frame_interval_s
    times, timestamp = [], max(first_end_s, args.start_s)
    while timestamp < duration_s:
        times.append(timestamp)
        timestamp += 1 / assessment_hz
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = prepare_windows(
        args.camera, times, config, set(args.rotate_180_camera))

    phase_backend = create_backend(config)
    safety_backend = create_backend(config)
    phase_critic = MonitorAgent(config, phase_backend)
    safety_critic = GlobalSafetyCritic(config, safety_backend)
    rows, phase_ms, safety_ms = [], [], []
    predictions_path = args.output_dir / 'predictions.jsonl'
    run_started = time.perf_counter()
    with predictions_path.open('w') as output:
        for index, (source_time_s, images) in enumerate(zip(times, windows)):
            payload = {
                'episode_id': f'{args.phase}-composite-video',
                'frame_id': f'window-{index:04d}',
                'timestamp_s': source_time_s,
                'expected_phase': args.phase,
                'images': images,
            }
            phase_started = time.perf_counter()
            phase_result = phase_critic.observe(payload)
            phase_ms.append((time.perf_counter() - phase_started) * 1000)
            phase_raw = phase_backend.last_raw
            safety_started = time.perf_counter()
            safety_result = safety_critic.observe(payload)
            safety_ms.append((time.perf_counter() - safety_started) * 1000)
            safety_raw = safety_backend.last_raw
            combined_status = (
                'fail' if safety_result['failure_accepted']
                else phase_result['status'])
            combined_decision = (
                'stop' if safety_result['failure_accepted']
                else phase_result['decision'])
            row = {
                'assessment_index': index,
                'source_time_s': round(source_time_s, 3),
                'phase_raw_model_output': phase_raw,
                'phase': phase_result,
                'safety_raw_model_output': safety_raw,
                'safety': safety_result,
                'combined_status': combined_status,
                'combined_decision': combined_decision,
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
            print(json.dumps({
                'index': index, 'source_time_s': row['source_time_s'],
                'phase_status': phase_result['status'],
                'pipette_support': safety_result['pipette_support'],
                'tube_support': safety_result['tube_support'],
                'safety_status': safety_result['status'],
                'safety_accepted': safety_result['failure_accepted'],
                'combined_status': combined_status,
            }), flush=True)

    summary = {
        'measurement': 'phase_plus_global_state_safety_critic',
        'model': config.model,
        'phase': args.phase,
        'source_videos': videos,
        'camera_count': config.camera_count,
        'assessment_start_s': args.start_s,
        'assessment_count': len(rows),
        'phase_backend_error_count': sum(
            row['phase']['backend_error_type'] is not None for row in rows),
        'safety_backend_error_count': sum(
            row['safety']['backend_error_type'] is not None for row in rows),
        'safety_statuses': dict(Counter(
            row['safety']['status'] for row in rows)),
        'pipette_supports': dict(Counter(
            row['safety']['pipette_support'] for row in rows)),
        'tube_supports': dict(Counter(
            row['safety']['tube_support'] for row in rows)),
        'combined_statuses': dict(Counter(
            row['combined_status'] for row in rows)),
        'first_safety_failure_source_time_s': next((
            row['source_time_s'] for row in rows
            if row['safety']['failure_accepted']), None),
        'phase_inference': latency_summary(phase_ms),
        'safety_inference': latency_summary(safety_ms),
        'combined_request_latency': latency_summary([
            phase + safety for phase, safety in zip(phase_ms, safety_ms)]),
        'run_wall_s': round(time.perf_counter() - run_started, 3),
        'predictions': str(predictions_path),
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary['phase_backend_error_count'] or summary['safety_backend_error_count']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
