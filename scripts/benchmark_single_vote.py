"""Run one phase video through the real agent and record every assessment."""
import argparse
import base64
from collections import Counter
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import statistics
import time

import cv2
from PIL import Image

from robot_agent.agent import MonitorAgent
from robot_agent.backends import create_backend
from robot_agent.config import load_config


def encode(frame, width, height):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (width, height))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    output = io.BytesIO()
    canvas.save(output, format='JPEG', quality=95, subsampling=0)
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode()


def video_info(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f'cannot open {path}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {'path': str(path), 'fps': fps, 'frames': frames,
            'duration_s': frames / fps,
            'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    capture.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f'invalid video metadata: {path}')
    return info


def prepare_windows(paths, end_times, config, rotate_indexes):
    """Decode each source frame once; input preparation is outside model timing."""
    per_camera = []
    for camera_index, path in enumerate(paths):
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or fps <= 0:
            raise RuntimeError(f'cannot open {path}')
        indexes_by_window = [[round((end_s - offset * config.frame_interval_s) * fps)
                              for offset in reversed(range(config.frames_per_camera))]
                             for end_s in end_times]
        needed = {index for indexes in indexes_by_window for index in indexes}
        encoded = {}
        for frame_index in range(max(needed) + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'cannot decode {path} at frame {frame_index}')
            if frame_index not in needed:
                continue
            if camera_index in rotate_indexes:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            encoded[frame_index] = encode(frame, config.image_width, config.image_height)
        capture.release()
        per_camera.append([[encoded[index] for index in indexes]
                           for indexes in indexes_by_window])
    return [[url for camera in per_camera for url in camera[window_index]]
            for window_index in range(len(end_times))]


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def latency_summary(values):
    if not values:
        return {'p50_ms': None, 'p95_ms': None, 'mean_ms': None}
    return {'p50_ms': round(statistics.median(values), 1),
            'p95_ms': round(percentile(values, .95), 1),
            'mean_ms': round(statistics.mean(values), 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/monitor.toml')
    parser.add_argument('--backend', choices=['openai', 'mock'])
    parser.add_argument('--camera', action='append', type=Path, required=True,
                        help='Repeat in camera-major order; the G1 dataset uses head then wrist_left')
    parser.add_argument('--rotate-180-camera', action='append', type=int, default=[])
    parser.add_argument('--phase', default='step1_pipette_pick')
    parser.add_argument('--assessment-hz', type=float)
    parser.add_argument('--timeout-s', type=float,
                        help='Override only the benchmark request timeout')
    parser.add_argument('--max-assessments', type=int, default=0)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('outputs/benchmark/phase1-single-vote'))
    args = parser.parse_args()

    config = load_config(args.config)
    if args.backend:
        config = replace(config, backend=args.backend)
    if not 1 <= len(args.camera) <= 3:
        parser.error('one to three --camera arguments are supported')
    config = replace(config, camera_count=len(args.camera))
    if args.timeout_s is not None:
        if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
            parser.error('--timeout-s must be finite and positive')
        config = replace(config, timeout_s=args.timeout_s)
    if args.phase not in config.phases:
        parser.error('phase is not configured')
    if any(index < 0 or index >= config.camera_count for index in args.rotate_180_camera):
        parser.error('--rotate-180-camera index is out of range')
    assessment_hz = args.assessment_hz or config.assessment_hz
    if assessment_hz <= 0 or args.max_assessments < 0:
        parser.error('assessment rate must be positive and max assessments nonnegative')

    videos = [video_info(path) for path in args.camera]
    duration_s = min(item['duration_s'] for item in videos)
    first_end_s = (config.frames_per_camera - 1) * config.frame_interval_s
    times = []
    timestamp = first_end_s
    while timestamp < duration_s:
        times.append(timestamp)
        timestamp += 1 / assessment_hz
    if args.max_assessments:
        times = times[:args.max_assessments]
    if not times:
        raise RuntimeError('video is shorter than one temporal window')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preparation_started = time.perf_counter()
    windows = prepare_windows(args.camera, times, config, set(args.rotate_180_camera))
    input_preparation_s = time.perf_counter() - preparation_started
    predictions_path = args.output_dir / 'predictions.jsonl'
    backend = create_backend(config)
    agent = MonitorAgent(config, backend)
    rows = []
    run_started = time.perf_counter()
    with predictions_path.open('w') as output:
        for index, (source_time_s, images) in enumerate(zip(times, windows)):
            result = agent.observe({
                'episode_id': f'{args.phase}-video',
                'frame_id': f'window-{index:04d}',
                'timestamp_s': source_time_s,
                'expected_phase': args.phase,
                'images': images,
            })
            agent_ms = result['metrics']['agent_latency_ms']
            row = {
                'assessment_index': index,
                'source_time_s': round(source_time_s, 3),
                'raw_model_output': getattr(backend, 'last_raw', None),
                **result,
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
            print(json.dumps({
                'index': index, 'source_time_s': row['source_time_s'],
                'status': result['status'], 'confidence': result['confidence'],
                'accepted': result['terminal_accepted'],
                'latency_ms': round(agent_ms, 1),
                'error': result['backend_error_type'],
            }), flush=True)

    run_wall_s = time.perf_counter() - run_started
    valid = [row for row in rows if row['backend_error_type'] is None]
    inference_ms = [row['metrics']['inference_and_parse_ms'] for row in valid]
    inference_stats = latency_summary(inference_ms)
    measurement_valid = config.backend == 'openai' and len(valid) == len(rows)
    prompt_tokens = [row['metrics']['prompt_tokens'] for row in valid
                     if isinstance(row['metrics']['prompt_tokens'], int)]
    completion_tokens = [row['metrics']['completion_tokens'] for row in valid
                         if isinstance(row['metrics']['completion_tokens'], int)]
    cached_tokens = [row['metrics']['cached_prompt_tokens'] for row in valid
                     if isinstance(row['metrics']['cached_prompt_tokens'], int)]
    accepted_times = [row['source_time_s'] for row in valid if row['terminal_accepted']]
    summary = {
        'measurement': 'single_vote_full_response',
        'measurement_valid': measurement_valid,
        'backend': config.backend,
        'model': config.model,
        'phase': args.phase,
        'source_videos': videos,
        'camera_count': config.camera_count,
        'camera_order': [str(path) for path in args.camera],
        'rotate_180_camera_indexes': args.rotate_180_camera,
        'frames_per_camera': config.frames_per_camera,
        'frame_interval_s': config.frame_interval_s,
        'image_size': [config.image_width, config.image_height],
        'assessment_hz_requested': assessment_hz,
        'thinking': config.chat_template_kwargs.get('enable_thinking'),
        'temperature': config.temperature,
        'max_tokens': config.max_tokens,
        'request_timeout_s': config.timeout_s,
        'assessment_count': len(rows),
        'valid_inference_count': len(valid),
        'backend_error_count': len(rows) - len(valid),
        'input_preparation_s': round(input_preparation_s, 3),
        'input_preparation_included_in_inference': False,
        'statuses': dict(Counter(row['status'] for row in valid)),
        'accepted_terminal_count': sum(row['terminal_accepted'] for row in valid),
        'first_accepted_source_time_s': accepted_times[0] if accepted_times else None,
        'tokens': {
            'prompt_median': statistics.median(prompt_tokens) if prompt_tokens else None,
            'completion_median': (statistics.median(completion_tokens)
                                  if completion_tokens else None),
            'cached_prompt_total': sum(cached_tokens) if cached_tokens else None,
            'prefix_cache_ratio': (round(sum(cached_tokens) / sum(prompt_tokens), 4)
                                   if cached_tokens and sum(prompt_tokens) else None),
        },
        'inference': inference_stats,
        'sequential_inference_hz': (
            round(1000 / inference_stats['p50_ms'], 3)
            if measurement_valid and inference_stats['p50_ms'] else None),
        'offline_request_loop_hz': round(len(rows) / run_wall_s, 3),
        'run_wall_s': round(run_wall_s, 3),
        'predictions': str(predictions_path),
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if config.backend == 'openai' and not summary['measurement_valid']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
