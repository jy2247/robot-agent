"""Infer one video or episode directory and render the critic output to MP4."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import cv2

from robot_agent.agent import ObserverAgent
from robot_agent.backends import create_backend
from robot_agent.config import load_config
from scripts.demo_failure_videos import (
    ERROR_DECISION, SUBTASK_PHASE, expected_errors, fit, frame_window, image_url, text,
)


CAMERAS = ('head', 'ego', 'left_wrist', 'right_wrist', 'right_thumb', 'side')


def camera_path(directory, name):
    candidates = (directory / f'{name}.mp4', directory / 'cams' / f'{name}_view.mp4')
    return next((path for path in candidates if path.exists()), None)


def discover_input(path, extra_views):
    if path.is_file():
        if path.suffix.lower() != '.mp4':
            raise ValueError('input file must be an MP4')
        paths = [path, *extra_views]
        return paths, [item.stem for item in paths], None
    if not path.is_dir():
        raise ValueError(f'input does not exist: {path}')
    annotation_path = path / 'annotation.json'
    annotation = json.loads(annotation_path.read_text()) if annotation_path.exists() else None
    available = {name: camera_path(path, name) for name in CAMERAS}
    available = {name: value for name, value in available.items() if value}
    if not available:
        mp4s = sorted(path.glob('*.mp4'))
        if not mp4s:
            raise ValueError('directory has no recognized camera MP4 files')
        return mp4s[:3], [item.stem for item in mp4s[:3]], annotation
    if annotation:
        subtask = annotation['failure']['subtask']
        preferred = (('right_thumb', 'side') if subtask == 'S4'
                     else ('ego', 'left_wrist', 'side'))
    else:
        preferred = ('head', 'ego', 'left_wrist', 'side')
    names = [name for name in preferred if name in available][:3]
    return [available[name] for name in names], names, annotation


def focused_config(config, phase, view_count):
    history_limit = max(1, 7 // view_count - 1)
    config = replace(config, history_size=min(config.history_size, history_limit))
    if phase is None:
        return config
    if phase not in config.phases:
        raise ValueError(f'unknown phase {phase}; choose one of {config.phases}')
    stage = config.phases.index(phase)
    return replace(
        config,
        task_id=f'{config.task_id}:one_click:{phase}',
        instruction=(f'Verify whether the robot completes or fails {phase}. '
                     'Track the relevant objects across the ordered camera observations.'),
        phases=(phase,), criteria=(config.criteria[stage],),
        failure_criteria=((config.failure_criteria[stage],)
                          if config.failure_criteria else ()),
    )


def sample_indexes(start, end, fps, sample_hz):
    stride = max(1, round(fps / sample_hz))
    indexes = list(range(start, end + 1, stride))
    if indexes[-1] != end:
        indexes.append(end)
    return indexes


assert sample_indexes(3, 12, 5, 1) == [3, 8, 12]


def infer(config, paths, names, phase, annotation, sample_hz, image_side, start_arg, end_arg):
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if any(not item.isOpened() for item in captures):
        raise RuntimeError('could not open every input video')
    fps = captures[0].get(cv2.CAP_PROP_FPS)
    counts = [int(item.get(cv2.CAP_PROP_FRAME_COUNT)) for item in captures]
    if fps <= 0 or min(counts) <= 0:
        raise RuntimeError('invalid video metadata')
    if annotation and start_arg is None and end_arg is None:
        start, end, indexes = frame_window(annotation)
        end = min(end, min(counts) - 1)
        indexes = [index for index in indexes if start <= index <= end]
    else:
        start = max(0, start_arg or 0)
        end = min(min(counts) - 1, end_arg if end_arg is not None else min(counts) - 1)
        indexes = sample_indexes(start, end, fps, sample_hz)
    if start > end:
        raise ValueError('start-frame must not exceed end-frame')

    active = focused_config(config, phase, len(paths))
    agent = ObserverAgent(active, create_backend(config))
    source_parent = paths[0].parent
    episode_id = (source_parent.parent.name if source_parent.name == 'cams'
                  else source_parent.name or paths[0].stem)
    results = []
    for index in indexes:
        frames = []
        for capture in captures:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'could not decode frame {index}')
            frames.append(frame)
        payload = {
            'episode_id': episode_id,
            'frame_id': str(index), 'timestamp_s': index / fps,
            'end_of_episode': index == min(counts) - 1,
            'note': 'Image order: ' + ', then '.join(names) + '.',
            'images': [image_url(frame, image_side) for frame in frames],
        }
        if phase:
            payload['expected_phase'] = phase
        results.append({'source_frame': index, **agent.observe(payload)})
    for capture in captures:
        capture.release()
    return results, start, end, fps


def evaluation(annotation, results):
    if not annotation:
        return None
    expected = expected_errors(annotation)
    decisions = sorted({ERROR_DECISION[error] for error in expected})
    actionable = next((row for row in results if row['decision'] in ('retry', 'replan', 'fail')), None)
    prediction = actionable or results[-1]
    matched = (prediction['status'] == 'failed'
               and bool(set(prediction['errors']) & set(expected))
               and prediction['decision'] in decisions)
    return {
        'label': {'subtask': annotation['failure']['subtask'],
                  'kind': annotation['failure']['kind'],
                  'visible_row': annotation['failure']['visible_row'],
                  'confirmed_row': annotation['failure']['confirmed_row'],
                  'human_verified': annotation.get('label_provenance', {}).get('human_verified')},
        'expected_errors': expected, 'expected_decisions': decisions,
        'prediction': prediction, 'matched': matched,
    }


def render(paths, names, results, start, end, fps, output, evaluation_result):
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    base_index = names.index('side') if 'side' in names else 0
    width, height = 1280, 720
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'could not create {output}')
    record_index, latched = 0, None
    for source_index in range(start, end + 1):
        frames = []
        for capture in captures:
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'could not render frame {source_index}')
            frames.append(frame)
        frame = fit(frames[base_index], width, height)
        inset_x = 18
        for index, (name, inset_frame) in enumerate(zip(names, frames)):
            if index == base_index:
                continue
            inset = fit(inset_frame, 300, 170)
            frame[18:188, inset_x:inset_x + 300] = inset
            cv2.rectangle(frame, (inset_x, 18), (inset_x + 300, 188), (230, 230, 230), 2)
            text(frame, name.replace('_', ' ').upper(), (inset_x + 10, 44), .48,
                 (255, 255, 255), 2)
            inset_x += 316

        while record_index + 1 < len(results) and results[record_index + 1]['source_frame'] <= source_index:
            record_index += 1
        current = results[record_index] if results[record_index]['source_frame'] <= source_index else None
        if current and current['decision'] in ('success', 'retry', 'replan', 'fail') and latched is None:
            latched = current
        shown = current if current and current.get('is_terminal') else latched or current
        overlay = frame.copy()
        cv2.rectangle(overlay, (500, 425), (1260, 700), (10, 14, 20), -1)
        cv2.addWeighted(overlay, .84, frame, .16, 0, frame)
        text(frame, 'QWEN3.8-27B TEMPORAL CRITIC', (525, 462), .72,
             (255, 255, 255), 2)
        if evaluation_result:
            label = evaluation_result['label']
            text(frame, f"DATASET: {label['subtask']} {label['kind'].upper()}",
                 (525, 502), .6, (120, 210, 255), 2)
        if shown:
            errors = shown.get('overall_reason') or ','.join(shown['errors']) or 'none'
            decision = shown['decision'].upper()
            color = ((60, 60, 240) if decision in ('REPLAN', 'FAIL') else
                     (40, 170, 255) if decision == 'RETRY' else
                     (70, 220, 90) if decision == 'SUCCESS' else (210, 210, 210))
            text(frame, f"PHASE: {shown['phase']}", (525, 540), .52)
            text(frame, f"STATUS: {shown['status'].upper()}  DECISION: {decision}",
                 (525, 580), .62, color, 2)
            text(frame, f"ERROR: {errors}  CONF: {shown['confidence']:.2f}",
                 (525, 618), .55, color, 2)
            evidence = shown['evidence'][0] if shown['evidence'] else 'No visual evidence returned'
            text(frame, evidence[:95], (525, 657), .42, (220, 220, 220))
        if evaluation_result:
            prediction_frame = evaluation_result['prediction']['source_frame']
            ready = source_index >= prediction_frame
            if ready:
                verdict = 'MATCH' if evaluation_result['matched'] else 'MISS / REVIEW'
                color = (70, 220, 80) if evaluation_result['matched'] else (50, 170, 255)
            else:
                verdict, color = 'PENDING', (210, 210, 210)
            text(frame, verdict, (1080, 470), .62, color, 2)
        text(frame, f'FRAME {source_index}', (22, 695), .45)
        writer.write(frame)
    for capture in captures:
        capture.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--view', type=Path, action='append', default=[])
    parser.add_argument('--phase')
    parser.add_argument('--config')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sample-hz', type=float, default=2.0)
    parser.add_argument('--image-side', type=int, default=448)
    parser.add_argument('--start-frame', type=int)
    parser.add_argument('--end-frame', type=int)
    args = parser.parse_args()
    if args.sample_hz <= 0 or args.image_side < 128:
        parser.error('sample-hz must be positive and image-side at least 128')
    paths, names, annotation = discover_input(args.input, args.view)
    config_path = args.config or ('configs/vega_pipette.toml' if annotation else
                                  'configs/g1_pipette.toml')
    config = load_config(config_path)
    phase = args.phase
    if phase is None and annotation:
        phase = config.phases[SUBTASK_PHASE[annotation['failure']['subtask']]]
    output = args.output or ((args.input / 'agent_output.mp4') if args.input.is_dir()
                             else args.input.with_name(args.input.stem + '_agent.mp4'))
    output.parent.mkdir(parents=True, exist_ok=True)
    results, start, end, fps = infer(config, paths, names, phase, annotation,
                                     args.sample_hz, args.image_side,
                                     args.start_frame, args.end_frame)
    evaluated = evaluation(annotation, results)
    render(paths, names, results, start, end, fps, output, evaluated)
    stem = output.with_suffix('')
    predictions_path = Path(str(stem) + '.predictions.json')
    summary_path = Path(str(stem) + '.summary.json')
    predictions_path.write_text(json.dumps(results, indent=2) + '\n')
    summary = {
        'model': config.model, 'input': [str(path) for path in paths],
        'views': names, 'phase': phase, 'frames': {'start': start, 'end': end, 'fps': fps},
        'inference_count': len(results),
        'mean_latency_ms': round(sum(row['metrics']['agent_latency_ms'] for row in results) / len(results), 1),
        'final_prediction': results[-1], 'evaluation': evaluated,
        'video': str(output), 'predictions': str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'video': str(output), 'predictions': str(predictions_path),
                      'summary': str(summary_path), 'final_prediction': results[-1],
                      'evaluation': evaluated}, indent=2))


if __name__ == '__main__':
    main()
