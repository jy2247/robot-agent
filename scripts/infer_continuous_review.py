"""Run unbiased temporal inference on the approved continuous review samples."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import time

import cv2

from robot_agent.agent import ObserverAgent
from robot_agent.backends import create_backend
from robot_agent.config import load_config
from scripts.demo_failure_videos import expected_errors, image_url, text


SHORT_NAMES = ('PIPETTE PICK', 'TUBE PICK', 'AIM AT TUBE',
               'TUBE RETURN', 'PIPETTE RETURN')
COLORS = {'done': (70, 220, 90), 'current': (255, 210, 50),
          'pending': (105, 105, 105), 'bad': (60, 60, 240)}


def sample_indexes(frame_count, fps, sample_hz):
    stride = max(1, round(fps / sample_hz))
    indexes = list(range(0, frame_count, stride))
    if indexes[-1] != frame_count - 1:
        indexes.append(frame_count - 1)
    return indexes


assert sample_indexes(10, 5, 1) == [0, 5, 9]


def capped_indexes(frame_count, fps, sample_hz, limit):
    indexes = sample_indexes(frame_count, fps, sample_hz)
    if not limit or len(indexes) <= limit:
        return indexes
    return sorted({indexes[round(position * (len(indexes) - 1) / (limit - 1))]
                   for position in range(limit)})


assert capped_indexes(101, 10, 1, 4) == [0, 30, 70, 100]


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def open_views(paths):
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if any(not capture.isOpened() for capture in captures):
        raise RuntimeError(f'could not open all views: {paths}')
    fps = captures[0].get(cv2.CAP_PROP_FPS)
    counts = [int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures]
    if fps <= 0 or min(counts) <= 0:
        raise RuntimeError(f'invalid video metadata: {paths}')
    return captures, fps, min(counts)


def observe(agent, sample_id, frame_id, timestamp_s, frames, view_names, image_side,
            end_of_episode=False):
    started = time.perf_counter()
    prediction = agent.observe({
        'episode_id': sample_id,
        'frame_id': frame_id,
        'timestamp_s': timestamp_s,
        'end_of_episode': end_of_episode,
        'note': ('Ordered simultaneous camera views: '
                 + ', then '.join(view_names)
                 + '. Infer the current phase and outcome only from visible evidence.'),
        'images': [image_url(frame, image_side) for frame in frames],
    })
    return prediction, (time.perf_counter() - started) * 1000


def infer_success(sample, config, sample_hz, image_side, max_phase_samples):
    agent = ObserverAgent(replace(config, history_size=1), create_backend(config))
    rows, global_offset = [], 0.0
    for phase_index, phase in enumerate(sample['phases']):
        paths = [Path(phase['head']), Path(phase['left_wrist'])]
        names = ['head', 'left_wrist']
        captures, fps, count = open_views(paths)
        indexes = capped_indexes(count, fps, sample_hz, max_phase_samples)
        for frame_index in indexes:
            frames = []
            for name, capture in zip(names, captures):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f'could not decode {paths[0]}:{frame_index}')
                if name == 'left_wrist':
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                frames.append(frame)
            prediction, wall_ms = observe(
                agent, sample['sample_id'], f'{phase["episode_index"]}:{frame_index}',
                global_offset + frame_index / fps, frames, names, image_side,
                end_of_episode=(phase_index == len(sample['phases']) - 1
                                and frame_index == count - 1))
            rows.append({
                'global_time_s': global_offset + frame_index / fps,
                'source_phase': phase['phase'], 'source_episode': phase['episode_index'],
                'source_frame': frame_index, 'wall_latency_ms': round(wall_ms, 3),
                'display_completed_phases': prediction['completed_phases'],
                'prediction': prediction,
            })
            print(f'{sample["sample_id"]} {phase["phase"]} frame={frame_index} '
                  f'-> {prediction["phase"]} {prediction["status"]}/'
                  f'{prediction["decision"]} {wall_ms:.1f}ms', flush=True)
        for capture in captures:
            capture.release()
        global_offset += count / fps
    return rows


def infer_failure(sample, config, sample_hz, image_side):
    agent = ObserverAgent(replace(config, history_size=1), create_backend(config))
    names = ['ego', 'left_wrist', 'side']
    paths = [Path(sample['videos'][name]) for name in names]
    captures, fps, count = open_views(paths)
    rows = []
    for frame_index in sample_indexes(count, fps, sample_hz):
        frames = []
        for capture in captures:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'could not decode {paths[0]}:{frame_index}')
            frames.append(frame)
        prediction, wall_ms = observe(
            agent, sample['sample_id'], str(frame_index), frame_index / fps,
            frames, names, image_side, end_of_episode=frame_index == count - 1)
        rows.append({
            'global_time_s': frame_index / fps, 'source_frame': frame_index,
            'wall_latency_ms': round(wall_ms, 3),
            'display_completed_phases': prediction['completed_phases'],
            'prediction': prediction,
        })
        print(f'{sample["sample_id"]} frame={frame_index} -> '
              f'{prediction["phase"]} {prediction["status"]}/'
              f'{prediction["decision"]} {wall_ms:.1f}ms', flush=True)
    for capture in captures:
        capture.release()
    return rows


def evaluate(sample, rows, phases):
    predictions = [row['prediction'] for row in rows]
    terminal = {key: predictions[-1].get(key) for key in
                ('overall_status', 'overall_reason', 'completed_phases', 'failed_phases')}
    if sample['kind'] == 'success':
        phase_matches = [row['source_phase'] == row['prediction']['phase'] for row in rows]
        accepted = []
        for prediction in predictions:
            phase = prediction.get('confirmed_phase')
            if phase and phase not in accepted:
                accepted.append(phase)
        return {
            'reference_kind': 'success',
            **terminal,
            'phase_accuracy': round(sum(phase_matches) / len(phase_matches), 4),
            'accepted_phases': accepted,
            'all_phases_accepted_in_order': accepted == list(phases),
            'false_failure_count': sum(p['status'] == 'failed' for p in predictions),
        }
    reference = sample['reference_label']
    expected = expected_errors({'failure': reference})
    failures = [row for row in rows if row['prediction']['status'] == 'failed']
    matching = [row for row in failures
                if set(row['prediction']['errors']) & set(expected)]
    return {
        'reference_kind': 'failure',
        **terminal,
        'reference_subtask': reference['subtask'],
        'reference_failure_kind': reference['kind'],
        'reference_visible_row': reference['visible_row'],
        'expected_error_codes': expected,
        'model_detected_failure': bool(failures),
        'model_error_matches_reference': bool(matching),
        'first_model_failure_frame': failures[0]['source_frame'] if failures else None,
        'first_matching_failure_frame': matching[0]['source_frame'] if matching else None,
    }


def draw_hud(frame, row, phases):
    prediction = row['prediction'] if row else None
    completed = set(row.get('display_completed_phases', ())) if row else set()
    failed = set(prediction.get('failed_phases', ())) if prediction else set()
    x0, y0 = 650, 370
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (1260, 700), (10, 14, 20), -1)
    cv2.addWeighted(overlay, .84, frame, .16, 0, frame)
    text(frame, 'QWEN3.8-27B TEMPORAL CRITIC', (x0 + 20, y0 + 34),
         .64, (255, 255, 255), 2)
    for index, (phase, label) in enumerate(zip(phases, SHORT_NAMES)):
        y = y0 + 75 + index * 35
        state = 'done' if phase in completed else 'pending'
        if phase in failed or (prediction and prediction['phase'] == phase
                               and prediction['status'] == 'failed'):
            state = 'bad'
        elif prediction and prediction['phase'] == phase and phase not in completed:
            state = 'current'
        cv2.rectangle(frame, (x0 + 20, y - 15), (x0 + 43, y + 8), COLORS[state], -1)
        text(frame, f'{index + 1}. {label}', (x0 + 54, y + 4), .48)
    if prediction:
        errors = prediction.get('overall_reason') or ','.join(prediction['errors']) or 'none'
        status = (prediction['overall_status'] if prediction.get('is_terminal')
                  else prediction['status'])
        color = COLORS['bad'] if status == 'failed' else (235, 235, 235)
        text(frame, f"{'FINAL' if prediction.get('is_terminal') else 'STATUS'} {status.upper()}   "
             f"DECISION {prediction['decision'].upper()}",
             (x0 + 295, y0 + 87), .47, color, 1)
        text(frame, f"CONF {prediction['confidence']:.2f}   "
             f"INFERENCE {row['wall_latency_ms']:.0f} ms",
             (x0 + 295, y0 + 124), .45, (220, 220, 220), 1)
        text(frame, f'ERROR {errors}'[:58], (x0 + 295, y0 + 161),
             .42, color, 1)
        evidence = prediction['evidence'][0] if prediction['evidence'] else 'No evidence returned'
        text(frame, evidence[:62], (x0 + 295, y0 + 200),
             .36, (210, 210, 210), 1)
    text(frame, 'YELLOW=current  GREEN=accepted  RED=failure',
         (x0 + 20, y0 + 306), .4, (200, 200, 200), 1)


def render(review_video, rows, phases, output):
    capture = cv2.VideoCapture(str(review_video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                             (1280, 720))
    if not capture.isOpened() or not writer.isOpened() or fps <= 0:
        raise RuntimeError(f'could not render {review_video}')
    row_index = 0
    final_frame = None
    for frame_index in range(count):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f'could not decode review frame {frame_index}')
        if frame_index == count - 1:
            final_frame = frame.copy()
        now = frame_index / fps
        while row_index + 1 < len(rows) and rows[row_index + 1]['global_time_s'] <= now:
            row_index += 1
        row = rows[row_index] if rows and rows[row_index]['global_time_s'] <= now else None
        draw_hud(frame, row, phases)
        writer.write(frame)
    if final_frame is not None and rows:
        draw_hud(final_frame, rows[-1], phases)
        text(final_frame, 'FINAL RESULT - 2s DISPLAY HOLD', (670, 716), .4)
        for _ in range(round(2 * fps)):
            writer.write(final_frame)
    capture.release()
    writer.release()


def run_sample(review_root, output_root, sample, configs, sample_hz, image_side,
               max_success_samples_per_phase):
    directory = output_root / sample['sample_id']
    directory.mkdir(parents=True, exist_ok=True)
    predictions_path = directory / 'predictions.json'
    started = time.perf_counter()
    if predictions_path.exists():
        raise FileExistsError(f'{predictions_path}: use a new --out-dir to evaluate changed prompts')
    else:
        rows = (infer_success(sample, configs['success'], sample_hz, image_side,
                              max_success_samples_per_phase)
                if sample['kind'] == 'success'
                else infer_failure(sample, configs['failure'], sample_hz, image_side))
        predictions_path.write_text(json.dumps(rows, indent=2) + '\n')
    inference_s = sum(row['wall_latency_ms'] for row in rows) / 1000
    latencies = [row['wall_latency_ms'] for row in rows]
    evaluation = evaluate(sample, rows, configs['success'].phases)
    render_started = time.perf_counter()
    output_video = directory / 'result.mp4'
    render(review_root / sample['sample_id'] / 'sample.mp4', rows,
           configs['success'].phases, output_video)
    render_s = time.perf_counter() - render_started
    with (directory / 'inference_times.jsonl').open('w') as output_file:
        for row in rows:
            output_file.write(json.dumps({
                'source_frame': row['source_frame'],
                'global_time_s': row['global_time_s'],
                'wall_latency_ms': row['wall_latency_ms'],
                'agent_latency_ms': row['prediction']['metrics']['agent_latency_ms'],
                'over_budget': row['prediction']['metrics']['over_budget'],
            }) + '\n')
    (directory / 'source_manifest.json').write_text(json.dumps(sample, indent=2) + '\n')
    summary = {
        'sample_id': sample['sample_id'], 'kind': sample['kind'],
        'model': configs['success'].model, 'sample_hz': sample_hz,
        'image_side': image_side, 'history_size': 1,
        'max_success_samples_per_phase': max_success_samples_per_phase,
        'inference_calls': len(rows),
        'mean_wall_latency_ms': round(statistics.mean(latencies), 1),
        'p50_wall_latency_ms': round(statistics.median(latencies), 1),
        'p95_wall_latency_ms': round(percentile(latencies, .95), 1),
        'effective_sequential_hz': round(1000 / statistics.mean(latencies), 3),
        'inference_wall_time_s': round(inference_s, 2),
        'render_wall_time_s': round(render_s, 2),
        'evaluation': evaluation,
        'output_video': str(output_video),
        'predictions': str(predictions_path),
        'labels_sent_to_model': False,
        'current_command_time_s': round(time.perf_counter() - started, 2),
    }
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review-dir', type=Path,
                        default=Path('outputs/continuous-sample-review-seed-20261021'))
    parser.add_argument('--out-dir', type=Path,
                        default=Path('outputs/continuous-inference-seed-20261021'))
    parser.add_argument('--sample-hz', type=float, default=2.0)
    parser.add_argument('--image-side', type=int, default=336)
    parser.add_argument('--max-success-samples-per-phase', type=int, default=0,
                        help='0 keeps every sampled frame')
    parser.add_argument('--workers', type=int, default=2,
                        help='parallel episodes; observations within an episode remain sequential')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sample-id', action='append',
                        help='select sample IDs from the manifest; repeat to select several')
    args = parser.parse_args()
    if (args.sample_hz <= 0 or args.image_side < 128
            or args.max_success_samples_per_phase in (1,)
            or args.max_success_samples_per_phase < 0 or args.workers < 1):
        parser.error('sample-hz must be positive and image-side at least 128')
    manifest = json.loads((args.review_dir / 'manifest.json').read_text())
    samples = manifest['samples'][:args.limit] if args.limit else manifest['samples']
    if args.sample_id:
        samples = [sample for sample in samples if sample['sample_id'] in args.sample_id]
        if set(args.sample_id) != {sample['sample_id'] for sample in samples}:
            parser.error('sample-id must exist in the selected manifest samples')
    configs = {'success': load_config('configs/g1_pipette.toml'),
               'failure': load_config('configs/vega_pipette.toml')}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    def run(sample):
        return run_sample(args.review_dir, args.out_dir, sample, configs,
                          args.sample_hz, args.image_side,
                          args.max_success_samples_per_phase)
    with ThreadPoolExecutor(max_workers=min(args.workers, len(samples))) as executor:
        summaries = list(executor.map(run, samples))
    all_latencies = [
        json.loads(line)['wall_latency_ms']
        for sample in samples
        for line in (args.out_dir / sample['sample_id'] / 'inference_times.jsonl').read_text().splitlines()
    ]
    batch = {
        'source_manifest': str(args.review_dir / 'manifest.json'),
        'model': configs['success'].model, 'vllm': True,
        'sample_hz': args.sample_hz, 'image_side': args.image_side,
        'max_success_samples_per_phase': args.max_success_samples_per_phase,
        'parallel_episode_workers': args.workers,
        'labels_sent_to_model': False, 'samples': summaries,
        'total_inference_calls': len(all_latencies),
        'mean_wall_latency_ms': round(statistics.mean(all_latencies), 1),
        'p50_wall_latency_ms': round(statistics.median(all_latencies), 1),
        'p95_wall_latency_ms': round(percentile(all_latencies, .95), 1),
        'effective_sequential_hz': round(1000 / statistics.mean(all_latencies), 3),
        'current_command_time_s': round(time.perf_counter() - started, 2),
    }
    (args.out_dir / 'batch_summary.json').write_text(json.dumps(batch, indent=2) + '\n')
    print(json.dumps(batch, indent=2), flush=True)


if __name__ == '__main__':
    main()
