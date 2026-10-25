"""Build and infer ten reproducible random phase-spliced pipette videos."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import statistics
import time
from urllib.request import urlopen

import cv2

from robot_agent.agent import ObserverAgent
from robot_agent.backends import create_backend
from robot_agent.config import load_config
from robot_agent.workflow import WorkflowTracker
from scripts.demo_failure_videos import frame_window, image_url, fit, text
from scripts.infer_video import focused_config


SUCCESS_REPO = 'jren313/g1-pipette-2view-teleop0925-5task-eerel'
FAIL_REPO = 'birbirll/vega-pipette-fail'
HF = 'https://huggingface.co/datasets'
PHASES = (
    'step1_pipette_pick', 'step2_tube_pick', 'step3_pipette_aim_at_tube',
    'step4_tube_return', 'step5_pipette_return',
)
PHASE_LABELS = ('PIPETTE PICK', 'TUBE PICK', 'AIM AT TUBE',
                'TUBE RETURN', 'PIPETTE RETURN')
TASK_PHASE = {
    'Pick up the pipette from the holder on the right with the right hand.': PHASES[0],
    'Pick up the tube from the blue rack on the left with the left hand.': PHASES[1],
    'Aim the pipette held in the right hand at the tube held in the left hand.': PHASES[2],
    'Put the tube held in the left hand back into the blue rack on the left.': PHASES[3],
    'Put the pipette held in the right hand back into the holder on the right.': PHASES[4],
}
SUBTASK_STAGE = {
    'S1': 0, 'R1': 0, 'S2': 1, 'S3': 2, 'S4': 2, 'S5': 2,
    'S6': 3, 'S7': 4, 'S8': 4,
}
SOURCE_LABEL_FIELDS = frozenset({
    'subtask', 'source_label', 'visible_row', 'confirmed_row',
})


def fetch(url, path):
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'DOWNLOAD {url}', flush=True)
    with urlopen(url, timeout=180) as source, path.open('wb') as output:
        while chunk := source.read(1024 * 1024):
            output.write(chunk)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def make_manifest(root, seed):
    metadata = root / '_metadata'
    success_meta = metadata / 'success_episodes.jsonl'
    failure_meta = metadata / 'failure_index.jsonl'
    fetch(f'{HF}/{SUCCESS_REPO}/resolve/main/meta/episodes.jsonl', success_meta)
    fetch(f'{HF}/{FAIL_REPO}/resolve/main/index.jsonl', failure_meta)

    pools = {phase: [] for phase in PHASES}
    for row in read_jsonl(success_meta):
        phase = TASK_PHASE.get(row['tasks'][0])
        if phase:
            pools[phase].append(row)
    if any(len(pool) < 10 for pool in pools.values()):
        raise RuntimeError('success dataset does not contain ten episodes for every phase')

    candidates = [
        row for row in read_jsonl(failure_meta)
        if row.get('usable') and row['failure']['occurred']
        and row['failure']['visible_row'] is not None
        and row['failure']['subtask'] in SUBTASK_STAGE
    ]
    failures = random.Random(seed).sample(candidates, 5)
    success_rng = random.Random(seed + 1)
    success_choices = {
        phase: success_rng.sample(pool, 10) for phase, pool in pools.items()
    }

    chains = []
    for chain_index in range(10):
        failure = failures[chain_index - 5] if chain_index >= 5 else None
        last_stage = SUBTASK_STAGE[failure['failure']['subtask']] if failure else 4
        stages = []
        for stage in range(last_stage + 1):
            phase = PHASES[stage]
            if failure and stage == last_stage:
                start, end, indexes = frame_window({'rows': failure['rows'],
                                                     'failure': failure['failure']})
                stages.append({
                    'phase': phase, 'dataset': 'vega_failure',
                    'episode': failure['episode'],
                    'subtask': failure['failure']['subtask'],
                    'source_label': failure['failure']['kind'],
                    'visible_row': failure['failure']['visible_row'],
                    'confirmed_row': failure['failure']['confirmed_row'],
                    'start_frame': start, 'end_frame': end,
                    'sample_frames': indexes,
                    'views': (['right_thumb', 'side'] if failure['failure']['subtask'] == 'S4'
                              else ['ego', 'left_wrist', 'side']),
                })
            else:
                episode = success_choices[phase][chain_index]
                stages.append({
                    'phase': phase, 'dataset': 'g1_success',
                    'episode': episode['episode_index'],
                    'source_label': 'completed', 'length': episode['length'],
                    'start_frame': 0, 'end_frame': episode['length'] - 1,
                    'sample_frames': sorted(set((0, max(0, episode['length'] - 31),
                                                 episode['length'] - 1))),
                    'views': ['head', 'left_wrist'],
                })
        chains.append({
            'chain_id': f'chain_{chain_index + 1:02d}',
            'kind': 'success' if failure is None else 'failure',
            'stages': stages,
        })
    manifest = {
        'seed': seed,
        'sample_count': len(chains),
        'sampling': '5 complete G1 success chains + 5 G1 prefixes ending at a random visual Vega failure',
        'labels_are_not_sent_to_model': True,
        'chains': chains,
    }
    (root / 'batch_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def source_paths(root, stage):
    episode = stage['episode']
    if stage['dataset'] == 'g1_success':
        directory = root / '_sources' / 'g1' / f'episode_{episode:06d}'
        names = stage['views']
        paths = [directory / f'{name}.mp4' for name in names]
    else:
        directory = root / '_sources' / 'vega' / episode
        names = stage['views']
        paths = [directory / f'{name}.mp4' for name in names]
    return paths, names


def prepare_sources(root, manifest):
    for chain in manifest['chains']:
        for stage in chain['stages']:
            paths, names = source_paths(root, stage)
            if stage['dataset'] == 'g1_success':
                episode = stage['episode']
                remotes = [
                    f'videos/chunk-000/observation.images.rgb/episode_{episode:06d}.mp4',
                    f'videos/chunk-000/observation.images.wrist_left/episode_{episode:06d}.mp4',
                ]
            else:
                remotes = [f"episodes/{stage['episode']}/cams/{name}_view.mp4"
                           for name in names]
            for remote, path in zip(remotes, paths):
                fetch(f'{HF}/{SUCCESS_REPO if stage["dataset"] == "g1_success" else FAIL_REPO}/resolve/main/{remote}', path)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def without_source_labels(stage):
    return {key: value for key, value in stage.items() if key not in SOURCE_LABEL_FIELDS}


def infer_stage(root, chain_id, stage, configs, backend):
    paths, names = source_paths(root, stage)
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if any(not capture.isOpened() for capture in captures):
        raise RuntimeError(f'could not open source videos for {chain_id} {stage["phase"]}')
    fps = captures[0].get(cv2.CAP_PROP_FPS)
    counts = [int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures]
    end = min(stage['end_frame'], min(counts) - 1)
    indexes = [index for index in stage['sample_frames'] if stage['start_frame'] <= index <= end]
    base = configs[stage['dataset']]
    config = focused_config(base, stage['phase'], len(paths))
    agent = ObserverAgent(config, backend)
    rows = []
    for call_index, frame_index in enumerate(indexes):
        frames = []
        for name, capture in zip(names, captures):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'could not decode {paths[0]} frame {frame_index}')
            if stage['dataset'] == 'g1_success' and name == 'left_wrist':
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            frames.append(frame)
        started = time.perf_counter()
        prediction = agent.observe({
            'episode_id': f'{chain_id}:{stage["phase"]}',
            'frame_id': str(frame_index), 'timestamp_s': frame_index / fps,
            'expected_phase': stage['phase'],
            'note': 'Image order: ' + ', then '.join(names) + '.',
            'images': [image_url(frame, 448) for frame in frames],
        })
        wall_ms = (time.perf_counter() - started) * 1000
        rows.append({'call_index': call_index, 'source_frame': frame_index,
                     'source_time_s': frame_index / fps,
                     'wall_latency_ms': round(wall_ms, 3),
                     'prediction': prediction})
        print(f'{chain_id} {stage["phase"]} frame={frame_index} '
              f'{prediction["status"]}/{prediction["decision"]} {wall_ms:.1f}ms', flush=True)
    for capture in captures:
        capture.release()
    return {
        **without_source_labels(stage), 'fps': fps, 'end_frame': end,
        'source_paths': [str(path) for path in paths], 'results': rows,
    }


def terminal_prediction(stage_result):
    predictions = [row['prediction'] for row in stage_result['results']]
    actionable = next((row for row in predictions
                       if row['decision'] in ('retry', 'replan', 'fail')), None)
    successful = next((row for row in reversed(predictions)
                       if row['decision'] == 'success'), None)
    return actionable or successful or predictions[-1]


def last_prediction(stage_result):
    return stage_result['results'][-1]['prediction']


def draw_hud(frame, current, phase_index, completed, source):
    x0, y0 = 700, 400
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (1260, 700), (10, 14, 20), -1)
    cv2.addWeighted(overlay, .84, frame, .16, 0, frame)
    text(frame, 'QWEN3.8-27B RAW MODEL OUTPUT', (x0 + 20, y0 + 34), .64,
         (255, 255, 255), 2)
    text(frame, f'SOURCE: {source.upper()}', (x0 + 20, y0 + 66), .46,
         (190, 210, 230), 1)
    for index, (phase, label) in enumerate(zip(PHASES, PHASE_LABELS)):
        y = y0 + 98 + index * 28
        if phase in completed:
            color = (70, 220, 90)
        elif index == phase_index:
            decision = current['decision'] if current else 'continue'
            color = ((60, 60, 240) if decision in ('replan', 'fail') else
                     (40, 170, 255) if decision == 'retry' else (255, 210, 50))
        else:
            color = (110, 110, 110)
        cv2.rectangle(frame, (x0 + 20, y - 14), (x0 + 40, y + 6), color, -1)
        text(frame, f'{index + 1}. {label}', (x0 + 50, y + 3), .42)
    if current:
        errors = ','.join(current['errors']) or 'none'
        text(frame, f"{current['status'].upper()} / {current['decision'].upper()}  "
             f"CONF {current['confidence']:.2f}", (x0 + 280, y0 + 112), .47,
             (235, 235, 235), 1)
        text(frame, f'ERROR: {errors}', (x0 + 280, y0 + 145), .45,
             (235, 235, 235), 1)
        latency = current['metrics']['agent_latency_ms']
        text(frame, f'INFERENCE: {latency:.1f} ms', (x0 + 280, y0 + 178), .43,
             (210, 210, 210), 1)
        evidence = current['evidence'][0] if current['evidence'] else 'No visual evidence returned'
        text(frame, evidence[:58], (x0 + 280, y0 + 214), .36, (210, 210, 210), 1)


def render_chain(root, stages, output):
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(f'could not create {output}')
    completed = set()
    for phase_index, stage in enumerate(stages):
        paths = [Path(path) for path in stage['source_paths']]
        names = stage['views']
        captures = [cv2.VideoCapture(str(path)) for path in paths]
        fps = stage['fps']
        results = stage['results']
        result_index = 0
        start, end = stage['start_frame'], stage['end_frame']
        output_count = max(1, round((end - start + 1) / fps * 30))
        for capture in captures:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        decoded_index, decoded = start - 1, []
        source_indexes = [min(end, start + round(index * fps / 30))
                          for index in range(output_count)]
        for source_index in source_indexes:
            while decoded_index < source_index:
                decoded = []
                for capture in captures:
                    ok, item = capture.read()
                    if not ok:
                        raise RuntimeError(f'could not render {paths[0]}:{source_index}')
                    decoded.append(item)
                decoded_index += 1
            frames = []
            for name, item in zip(names, decoded):
                if stage['dataset'] == 'g1_success' and name == 'left_wrist':
                    item = cv2.rotate(item, cv2.ROTATE_180)
                frames.append(item)
            main_index = names.index('side') if 'side' in names else 0
            frame = fit(frames[main_index], 1280, 720)
            inset_x = 18
            for index, (name, item) in enumerate(zip(names, frames)):
                if index == main_index:
                    continue
                inset = fit(item, 300, 170)
                frame[18:188, inset_x:inset_x + 300] = inset
                cv2.rectangle(frame, (inset_x, 18), (inset_x + 300, 188),
                              (230, 230, 230), 2)
                text(frame, name.replace('_', ' ').upper(), (inset_x + 10, 44),
                     .45, (255, 255, 255), 2)
                inset_x += 316
            while (result_index + 1 < len(results)
                   and results[result_index + 1]['source_frame'] <= source_index):
                result_index += 1
            row = results[result_index] if results[result_index]['source_frame'] <= source_index else None
            current = row['prediction'] if row else None
            draw_hud(frame, current, phase_index, completed, stage['dataset'])
            text(frame, f"{stage['phase']}  FRAME {source_index}", (22, 695), .45)
            writer.write(frame)
        outcome = terminal_prediction(stage)
        if outcome['decision'] == 'success':
            completed.add(stage['phase'])
        for capture in captures:
            capture.release()
    writer.release()


def run_batch(root, manifest):
    configs = {
        'g1_success': load_config('configs/g1_pipette.toml'),
        'vega_failure': load_config('configs/vega_pipette.toml'),
    }
    backend = create_backend(configs['g1_success'])
    batch_started = time.perf_counter()
    batch_rows = []
    for chain in manifest['chains']:
        directory = root / chain['chain_id']
        directory.mkdir(parents=True, exist_ok=True)
        predictions_path = directory / 'predictions.json'
        if predictions_path.exists():
            raw_output = json.loads(predictions_path.read_text())
            stages = [without_source_labels(stage) for stage in raw_output['stages']]
            transitions = raw_output['workflow_transitions']
            print(f"{chain['chain_id']} reusing saved raw predictions", flush=True)
        else:
            tracker = WorkflowTracker(PHASES)
            stages = []
            transitions = []
            for stage in chain['stages']:
                result = infer_stage(root, chain['chain_id'], stage, configs, backend)
                stages.append(result)
                outcome = terminal_prediction(result)
                transitions.append(tracker.apply(
                    f"{chain['chain_id']}:{stage['phase']}", outcome))
        inference_rows = [
            {'phase': stage['phase'], 'dataset': stage['dataset'],
             'episode': stage['episode'], **{key: value for key, value in row.items()
                                             if key != 'prediction'},
             'agent_latency_ms': row['prediction']['metrics']['agent_latency_ms']}
            for stage in stages for row in stage['results']
        ]
        latencies = [row['wall_latency_ms'] for row in inference_rows]
        raw_output = {
            'chain_id': chain['chain_id'], 'kind': chain['kind'],
            'stages': stages, 'workflow_transitions': transitions,
        }
        predictions_path.write_text(json.dumps(raw_output, indent=2) + '\n')
        with (directory / 'inference_times.jsonl').open('w') as output_file:
            for row in inference_rows:
                output_file.write(json.dumps(row) + '\n')
        source_only = {key: value for key, value in chain.items()}
        (directory / 'source_manifest.json').write_text(json.dumps(source_only, indent=2) + '\n')
        render_started = time.perf_counter()
        render_chain(root, stages, directory / 'result.mp4')
        render_s = time.perf_counter() - render_started
        summary = {
            'chain_id': chain['chain_id'], 'kind': chain['kind'],
            'inference_calls': len(latencies),
            'mean_wall_latency_ms': round(statistics.mean(latencies), 1),
            'p50_wall_latency_ms': round(statistics.median(latencies), 1),
            'p95_wall_latency_ms': round(percentile(latencies, .95), 1),
            'inference_wall_time_s': round(sum(latencies) / 1000, 2),
            'render_wall_time_s': round(render_s, 2),
            'total_recorded_work_s': round(sum(latencies) / 1000 + render_s, 2),
            'last_raw_prediction': last_prediction(stages[-1]),
            'fc_transition_prediction': terminal_prediction(stages[-1]),
            'video': str(directory / 'result.mp4'),
        }
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        batch_rows.append(summary)
        print(json.dumps(summary), flush=True)
    all_latencies = [
        json.loads(line)['wall_latency_ms']
        for chain in manifest['chains']
        for line in (root / chain['chain_id'] / 'inference_times.jsonl').read_text().splitlines()
    ]
    batch_summary = {
        'seed': manifest['seed'], 'chains': batch_rows,
        'total_inference_calls': len(all_latencies),
        'mean_wall_latency_ms': round(statistics.mean(all_latencies), 1),
        'p50_wall_latency_ms': round(statistics.median(all_latencies), 1),
        'p95_wall_latency_ms': round(percentile(all_latencies, .95), 1),
        'current_command_time_s': round(time.perf_counter() - batch_started, 2),
        'total_recorded_work_s': round(sum(row['total_recorded_work_s']
                                           for row in batch_rows), 2),
        'outputs_are_unedited_model_results': True,
    }
    (root / 'batch_summary.json').write_text(json.dumps(batch_summary, indent=2) + '\n')
    print(json.dumps(batch_summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path,
                        default=Path('outputs/random-phase-batch-seed-20261020'))
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only and args.run_only:
        parser.error('--prepare-only and --run-only are mutually exclusive')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / 'batch_manifest.json'
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists()
                else make_manifest(args.out_dir, args.seed))
    if not args.run_only:
        prepare_sources(args.out_dir, manifest)
    if not args.prepare_only:
        run_batch(args.out_dir, manifest)


if __name__ == '__main__':
    main()
