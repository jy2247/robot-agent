"""Evaluate several annotated Vega failures and render model-driven MP4 demos."""
import argparse
import base64
from dataclasses import replace
import io
import json
from pathlib import Path
from urllib.request import urlopen

import cv2
from PIL import Image

from robot_agent.agent import ObserverAgent
from robot_agent.backends import create_backend
from robot_agent.config import load_config


DATASET_ROOT = ('https://huggingface.co/datasets/birbirll/vega-pipette-fail/'
                'resolve/main/episodes')
CASES = (
    'ws_bridge_v31_30k_b0828',       # S2: tube dropped during pick
    'ws_bridge_v31_10k_b0911',       # S3: object dropped during aim
    'ws_a2mem_chain_b0065',          # S6: tube dropped and tipped during return
    'ws_a2mem_chain_b0911',          # S8: pipette stand displaced
    'ws_p05front_b3_30k_hval_b0227', # S4: press stops short of target
)
CASE_VIEWS = {
    'ws_bridge_v31_30k_b0828': ('ego', 'left_wrist', 'side'),
    'ws_bridge_v31_10k_b0911': ('ego', 'left_wrist', 'side'),
    'ws_a2mem_chain_b0065': ('ego', 'left_wrist', 'side'),
    'ws_a2mem_chain_b0911': ('ego', 'left_wrist', 'side'),
    'ws_p05front_b3_30k_hval_b0227': ('right_thumb', 'side'),
}
SUBTASK_PHASE = {
    'S1': 0, 'R1': 0, 'S2': 1, 'S3': 2, 'S4': 2, 'S5': 2,
    'S6': 3, 'S7': 4, 'S8': 4,
}
KIND_ERROR = {
    'dropped': 'object_dropped', 'tipped_over': 'object_tipped',
    'pushed_stand': 'fixture_displaced', 'short_of_target': 'short_of_target',
    'stalled': 'stalled',
}
ERROR_DECISION = {
    'object_dropped': 'replan', 'object_tipped': 'replan',
    'fixture_displaced': 'replan', 'short_of_target': 'retry', 'stalled': 'retry',
}


def download(url, path):
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as source, path.open('wb') as output:
        while chunk := source.read(1024 * 1024):
            output.write(chunk)


def ensure_case(root, episode, views):
    directory = root / episode
    files = [('annotation.json', 'annotation.json')]
    files.extend((f'cams/{view}_view.mp4', f'{view}.mp4') for view in views)
    for remote, local in files:
        download(f'{DATASET_ROOT}/{episode}/{remote}', directory / local)
    return directory


def image_url(frame, max_side):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.thumbnail((max_side, max_side))
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=82)
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode()


def frame_window(annotation):
    failure = annotation['failure']
    anchor = failure['visible_row'] if failure['visible_row'] is not None else failure['confirmed_row']
    start = max(0, anchor - 120)
    end = min(annotation['rows'] - 1, failure['confirmed_row'] + 15)
    indexes = list(range(start, end + 1, 30))
    indexes.extend((anchor, failure['confirmed_row'], end))
    return start, end, sorted(set(indexes))


assert frame_window({'rows': 200, 'failure': {'visible_row': 150, 'confirmed_row': 160}}) == (
    30, 175, [30, 60, 90, 120, 150, 160, 175])


def expected_errors(annotation):
    kinds = {annotation['failure']['kind']}
    kinds.update(signal['kind'] for signal in annotation['failure'].get('signals', ()))
    return sorted({KIND_ERROR[kind] for kind in kinds if kind in KIND_ERROR})


def phase_config(config, annotation):
    stage = SUBTASK_PHASE[annotation['failure']['subtask']]
    phase = config.phases[stage]
    return replace(
        config,
        history_size=2,
        task_id=f'{config.task_id}:failure:{phase}',
        instruction=(f'Verify execution of {phase} in a simulated pipette task. '
                     'Detect only failures visible in the ordered ego and side views. '
                     'Do not copy labels from metadata; none are provided.'),
        phases=(phase,),
        criteria=(config.criteria[stage],),
        failure_criteria=(config.failure_criteria[stage],),
    ), stage, phase


def infer_case(config, backend, directory, annotation, image_side, views):
    focused, stage, phase = phase_config(config, annotation)
    if len(views) >= 3:
        focused = replace(focused, history_size=1)
    agent = ObserverAgent(focused, backend)
    captures = [cv2.VideoCapture(str(directory / f'{view}.mp4')) for view in views]
    if any(not capture.isOpened() for capture in captures):
        raise RuntimeError(f'could not open videos in {directory}')
    fps = captures[0].get(cv2.CAP_PROP_FPS)
    start, end, indexes = frame_window(annotation)
    results = []
    for index in indexes:
        frames = []
        for capture in captures:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'could not decode {directory.name}:{index}')
            frames.append(frame)
        result = agent.observe({
            'episode_id': directory.name,
            'frame_id': f'{directory.name}:{index}',
            'timestamp_s': index / fps,
            'expected_phase': phase,
            'note': 'Image order: ' + ', then '.join(view.replace('_', '-') + ' view'
                                                       for view in views) + '.',
            'images': [image_url(frame, image_side) for frame in frames],
        })
        results.append({'source_frame': index, **result})
    for capture in captures:
        capture.release()

    expected = expected_errors(annotation)
    expected_decisions = sorted({ERROR_DECISION[error] for error in expected})
    actionable = next((row for row in results if row['decision'] in ('retry', 'replan', 'fail')), None)
    prediction = actionable or results[-1]
    identified = (prediction['status'] == 'failed'
                  and bool(set(prediction['errors']) & set(expected))
                  and prediction['decision'] in expected_decisions)
    summary = {
        'episode': directory.name,
        'source_label': {
            'subtask': annotation['failure']['subtask'],
            'kind': annotation['failure']['kind'],
            'visible_row': annotation['failure']['visible_row'],
            'confirmed_row': annotation['failure']['confirmed_row'],
            'human_verified': annotation.get('label_provenance', {}).get('human_verified'),
        },
        'phase': phase,
        'views': list(views),
        'expected_errors': expected,
        'expected_decisions': expected_decisions,
        'prediction': prediction,
        'first_detection_frame': actionable['source_frame'] if actionable else None,
        'identified': identified,
        'window': {'start': start, 'end': end, 'fps': fps},
        'results': results,
    }
    return summary


def fit(frame, width, height):
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
    canvas = cv2.copyMakeBorder(
        resized, 0, height - resized.shape[0], 0, width - resized.shape[1],
        cv2.BORDER_CONSTANT, value=(12, 14, 18))
    return canvas


def text(frame, value, xy, scale=.55, color=(235, 235, 235), thickness=1):
    cv2.putText(frame, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def render_case(directory, summary, output):
    side = cv2.VideoCapture(str(directory / 'side.mp4'))
    detail_name = summary['views'][0]
    detail = cv2.VideoCapture(str(directory / f'{detail_name}.mp4'))
    fps = summary['window']['fps']
    start, end = summary['window']['start'], summary['window']['end']
    width, height = 1280, 720
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'could not write {output}')
    results = summary['results']
    result_index = 0
    latched = None
    annotation = summary['source_label']
    for source_index in range(start, end + 1):
        side.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        detail.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok_side, side_frame = side.read()
        ok_detail, detail_frame = detail.read()
        if not ok_side or not ok_detail:
            raise RuntimeError(f'could not render {directory.name}:{source_index}')
        frame = fit(side_frame, width, height)
        inset = fit(detail_frame, 424, 240)
        frame[18:258, 18:442] = inset
        cv2.rectangle(frame, (18, 18), (442, 258), (230, 230, 230), 2)
        text(frame, detail_name.replace('_', ' ').upper() + ' VIEW',
             (28, 45), .55, (255, 255, 255), 2)

        while result_index + 1 < len(results) and results[result_index + 1]['source_frame'] <= source_index:
            result_index += 1
        current = results[result_index] if results[result_index]['source_frame'] <= source_index else None
        if current and current['decision'] in ('retry', 'replan', 'fail') and latched is None:
            latched = current
        shown = latched or current

        panel = frame.copy()
        cv2.rectangle(panel, (520, 430), (1260, 700), (10, 14, 20), -1)
        cv2.addWeighted(panel, .84, frame, .16, 0, frame)
        text(frame, 'QWEN3.8-27B FAILURE CRITIC', (545, 465), .72, (255, 255, 255), 2)
        text(frame, f"DATASET: {annotation['subtask']}  {annotation['kind'].upper()}",
             (545, 505), .62, (120, 210, 255), 2)
        text(frame, f"TARGET PHASE: {summary['phase']}", (545, 540), .52)
        if shown:
            decision = shown['decision'].upper()
            color = ((60, 60, 240) if decision in ('REPLAN', 'FAIL') else
                     (40, 170, 255) if decision == 'RETRY' else (80, 210, 230))
            errors = ','.join(shown['errors']) or 'none'
            text(frame, f"MODEL: {shown['status'].upper()}  DECISION: {decision}",
                 (545, 580), .62, color, 2)
            text(frame, f"ERROR: {errors}  CONF: {shown['confidence']:.2f}",
                 (545, 617), .55, color, 2)
            evidence = shown['evidence'][0] if shown['evidence'] else 'No visual evidence returned'
            text(frame, evidence[:92], (545, 655), .43, (220, 220, 220))
        verdict = 'MATCH' if summary['identified'] else 'MISS / REVIEW'
        verdict_color = (70, 220, 80) if summary['identified'] else (50, 170, 255)
        text(frame, verdict, (1090, 475), .65, verdict_color, 2)
        text(frame, f'FRAME {source_index}', (22, 695), .45)
        writer.write(frame)
    side.release()
    detail.release()
    writer.release()


def concatenate(paths, output):
    first = cv2.VideoCapture(str(paths[0]))
    fps = first.get(cv2.CAP_PROP_FPS)
    width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first.release()
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
        capture.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/vega_pipette.toml')
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/failure-demos'))
    parser.add_argument('--image-side', type=int, default=448)
    args = parser.parse_args()
    if args.image_side < 128:
        parser.error('image-side must be at least 128')
    config = load_config(args.config)
    backend = create_backend(config)
    summaries, videos = [], []
    for episode in CASES:
        views = CASE_VIEWS.get(episode, ('ego', 'side'))
        directory = ensure_case(args.out_dir, episode, views)
        annotation = json.loads((directory / 'annotation.json').read_text())
        summary = infer_case(config, backend, directory, annotation, args.image_side, views)
        output = directory / 'failure_demo.mp4'
        render_case(directory, summary, output)
        (directory / 'predictions.json').write_text(json.dumps(summary['results'], indent=2) + '\n')
        public_summary = {key: value for key, value in summary.items() if key != 'results'}
        (directory / 'summary.json').write_text(json.dumps(public_summary, indent=2) + '\n')
        summaries.append(public_summary)
        videos.append(output)
        print(json.dumps(public_summary), flush=True)
    concatenate(videos, args.out_dir / 'failure_gallery.mp4')
    report = {
        'model': config.model,
        'cases': summaries,
        'identified_count': sum(item['identified'] for item in summaries),
        'case_count': len(summaries),
        'gallery': str(args.out_dir / 'failure_gallery.mp4'),
    }
    (args.out_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
