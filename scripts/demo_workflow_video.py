"""Run the five-stage critic and render its actual decisions onto one MP4."""
import argparse
import base64
from dataclasses import replace
import io
import json
from pathlib import Path
import time

import cv2
from PIL import Image

from robot_agent.agent import ObserverAgent
from robot_agent.backends import create_backend
from robot_agent.config import load_config
from robot_agent.workflow import WorkflowTracker


EPISODES = (237, 238, 239, 240, 241)
WRIST_STAGES = frozenset((1, 2, 3))
SHORT_NAMES = ('PIPETTE PICK', 'TUBE PICK', 'AIM AT TUBE', 'TUBE RETURN', 'PIPETTE RETURN')
COLORS = {'done': (70, 210, 80), 'current': (255, 210, 50),
          'target': (60, 170, 255), 'pending': (120, 120, 120), 'bad': (60, 60, 240)}


def image_url(frame, max_side):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.thumbnail((max_side, max_side))
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=78)
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode()


def sample_indexes(frame_count, fps, sample_hz):
    stride = max(1, round(fps / sample_hz))
    indexes = list(range(0, frame_count, stride))
    if indexes[-1] != frame_count - 1:
        indexes.append(frame_count - 1)
    return indexes


assert sample_indexes(10, 5, 1) == [0, 5, 9]


def observe_chain(config, paths, sample_hz, image_side):
    backend = create_backend(config)
    tracker = WorkflowTracker(config.phases, config.max_retries)
    completed, records, offset = set(), [], 0.0
    for stage, (episode, path, phase) in enumerate(zip(EPISODES, paths, config.phases)):
        wrist_path = path.parent / f'wrist_{path.name}'
        view_paths = [path, *([wrist_path] if stage in WRIST_STAGES and wrist_path.exists() else [])]
        captures = [cv2.VideoCapture(str(view_path)) for view_path in view_paths]
        capture = captures[0]
        if any(not item.isOpened() for item in captures):
            raise RuntimeError(f'could not open {path}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # The FC owns the ordered workflow and asks the critic to verify one
        # concrete phase.  Keeping only that phase in the per-episode prompt
        # prevents the VLM from confusing a completed endpoint with the next
        # phase's starting pose.  WorkflowTracker still enforces all five.
        phase_config = replace(
            config,
            task_id=f'{config.task_id}:{phase}',
            instruction=(f'Verify whether the robot completes {phase}. '
                         'Judge the ordered observations against the configured success and failure criteria.'),
            phases=(phase,),
            criteria=(config.criteria[stage],),
            failure_criteria=((config.failure_criteria[stage],)
                              if config.failure_criteria else ()),
        )
        agent = ObserverAgent(phase_config, backend)
        transition_applied = False
        observations = []
        for frame_index in sample_indexes(count, fps, sample_hz):
            frames = []
            for view_index, item in enumerate(captures):
                item.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = item.read()
                if not ok:
                    raise RuntimeError(f'could not decode {path}:{frame_index}')
                if view_index:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                frames.append(frame)
            observations.append((f'{episode}:{frame_index}', frame_index / fps,
                                 offset + frame_index / fps, frames))
        if stage + 1 < len(paths):
            next_head = paths[stage + 1]
            next_wrist = next_head.parent / f'wrist_{next_head.name}'
            frames = []
            next_views = [next_head, *([next_wrist]
                                      if stage in WRIST_STAGES and next_wrist.exists() else [])]
            for view_index, view_path in enumerate(next_views):
                next_capture = cv2.VideoCapture(str(view_path))
                ok, frame = next_capture.read()
                next_capture.release()
                if not ok:
                    raise RuntimeError(f'could not decode transition after {path}')
                if view_index:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                frames.append(frame)
            observations.append((f'{episode}:next-episode-first-frame', count / fps,
                                 offset + count / fps, frames))
        for frame_id, local_time, global_time, frames in observations:
            result = agent.observe({
                'episode_id': f'chain-{episode}', 'frame_id': frame_id,
                'timestamp_s': local_time, 'expected_phase': phase,
                'note': ('FC target is the configured phase. Image order: head view, '
                         'then upright left-wrist view when present.'),
                'images': [image_url(frame, image_side) for frame in frames],
            })
            transition = None
            if not transition_applied and result['decision'] in ('success', 'replan', 'fail'):
                transition = tracker.apply(f'chain-{episode}', result)
                transition_applied = True
                if transition['accepted']:
                    completed.add(phase)
            records.append({
                'global_time_s': global_time,
                'source_episode': episode, 'source_stage': stage,
                'completed_phases': sorted(completed),
                'transition': transition, **result,
            })
            # FC advances immediately after accepting a phase. Retain this
            # success record on the HUD instead of judging irrelevant tail
            # frames from the already completed source segment.
            if transition and transition['accepted']:
                break
        for item in captures:
            item.release()
        offset += count / fps
    return records


def benchmark(config, path, image_sizes):
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for index in (0, count - 1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError('could not decode benchmark frame')
        frames.append((index, frame))
    capture.release()
    rows = []
    for size in image_sizes:
        agent = ObserverAgent(config, create_backend(config))
        results = []
        for index, frame in frames:
            started = time.perf_counter()
            result = agent.observe({
                'episode_id': f'bench-{size}', 'frame_id': f'{size}:{index}',
                'timestamp_s': index / 60, 'expected_phase': config.phases[0],
                'images': [image_url(frame, size)],
            })
            results.append((time.perf_counter() - started, result))
        latency = results[-1][0]
        rows.append({'image_side': size, 'history_size': config.history_size,
                     'latency_ms': round(latency * 1000, 1), 'effective_hz': round(1 / latency, 3),
                     'phase': results[-1][1]['phase'], 'status': results[-1][1]['status'],
                     'backend_error_type': results[-1][1]['backend_error_type']})
    return rows


def draw_hud(frame, record, target_index, config):
    height, width = frame.shape[:2]
    panel_w, panel_h = min(500, width - 30), 330
    x0, y0 = width - panel_w - 15, height - panel_h - 15
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (width - 15, height - 15), (12, 18, 25), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(frame, 'QWEN3.8-27B TEMPORAL CRITIC', (x0 + 18, y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2, cv2.LINE_AA)
    completed = set(record.get('completed_phases', ())) if record else set()
    observed = record.get('phase') if record else None
    decision = record.get('decision') if record else 'waiting'
    for index, (phase, label) in enumerate(zip(config.phases, SHORT_NAMES)):
        y = y0 + 65 + index * 38
        state = 'done' if phase in completed else 'current' if phase == observed else 'target' if index == target_index else 'pending'
        if decision in ('replan', 'fail') and phase == observed:
            state = 'bad'
        cv2.rectangle(frame, (x0 + 18, y - 16), (x0 + 42, y + 8), COLORS[state], -1)
        cv2.putText(frame, f'{index + 1}. {label}', (x0 + 54, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    status = record.get('status', 'waiting') if record else 'waiting'
    confidence = record.get('confidence', 0.0) if record else 0.0
    latency = record.get('metrics', {}).get('agent_latency_ms', 0) if record else 0
    cv2.putText(frame, f'STATUS {status.upper()}  DECISION {decision.upper()}', (x0 + 18, y0 + 275),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, f'CONF {confidence:.2f}   LAST INFERENCE {latency:.0f} ms', (x0 + 18, y0 + 305),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)


def render(paths, records, output, config, output_fps=30, output_width=960):
    first = cv2.VideoCapture(str(paths[0]))
    source_w, source_h = int(first.get(cv2.CAP_PROP_FRAME_WIDTH)), int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first.release()
    output_height = round(source_h * output_width / source_w)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), output_fps,
                             (output_width, output_height))
    if not writer.isOpened():
        raise RuntimeError('could not create output MP4')
    record_index, elapsed = 0, 0.0
    for target_index, path in enumerate(paths):
        capture = cv2.VideoCapture(str(path))
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        output_count = round(source_count / source_fps * output_fps)
        frame = None
        decoded_index = -1
        for out_index in range(output_count):
            source_index = min(source_count - 1, round(out_index * source_fps / output_fps))
            while decoded_index < source_index:
                ok, frame = capture.read()
                decoded_index += 1
                if not ok:
                    raise RuntimeError(f'could not render {path}:{source_index}')
            now = elapsed + source_index / source_fps
            while record_index + 1 < len(records) and records[record_index + 1]['global_time_s'] <= now:
                record_index += 1
            record = records[record_index] if records and records[record_index]['global_time_s'] <= now else None
            frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            draw_hud(frame, record, target_index, config)
            writer.write(frame)
        capture.release()
        elapsed += source_count / source_fps
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/g1_pipette_fast.toml')
    parser.add_argument('--input-dir', type=Path, default=Path('outputs/workflow-demo'))
    parser.add_argument('--sample-hz', type=float, default=0.5)
    parser.add_argument('--image-side', type=int, default=336)
    parser.add_argument('--benchmark-only', action='store_true')
    args = parser.parse_args()
    if args.sample_hz <= 0 or args.image_side < 128:
        parser.error('sample-hz must be positive and image-side at least 128')
    config = load_config(args.config)
    paths = [args.input_dir / f'episode_{episode:06d}.mp4' for episode in EPISODES]
    if not all(path.exists() for path in paths):
        parser.error('download episodes 237-241 first')

    benchmark_rows = benchmark(config, paths[0], (224, 336, 448))
    (args.input_dir / 'vllm-benchmark.json').write_text(json.dumps(benchmark_rows, indent=2) + '\n')
    print(json.dumps({'benchmark': benchmark_rows}, indent=2), flush=True)
    if args.benchmark_only:
        return
    records = observe_chain(replace(config, history_size=1), paths, args.sample_hz, args.image_side)
    (args.input_dir / 'predictions.json').write_text(json.dumps(records, indent=2) + '\n')
    render(paths, records, args.input_dir / 'five_phase_critic_demo.mp4', config)
    accepted = [row['phase'] for row in records if (row.get('transition') or {}).get('accepted')]
    summary = {'episodes': list(EPISODES), 'sample_hz': args.sample_hz,
               'image_side': args.image_side, 'accepted_phases': accepted,
               'all_phases_accepted': accepted == list(config.phases),
               'mean_latency_ms': round(sum(row['metrics']['agent_latency_ms'] for row in records) / len(records), 1),
               'effective_sequential_hz': round(1000 / (sum(row['metrics']['agent_latency_ms'] for row in records) / len(records)), 3),
               'video': str(args.input_dir / 'five_phase_critic_demo.mp4')}
    (args.input_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
