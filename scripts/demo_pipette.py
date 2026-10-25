"""Download two small public clips and run the local temporal Qwen observer."""
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
from robot_agent.workflow import WorkflowTracker


SUCCESS_VIDEO = ('https://huggingface.co/datasets/jren313/'
                 'g1-pipette-2view-teleop0925-5task-eerel/resolve/main/'
                 'videos/chunk-000/observation.images.rgb/episode_000000.mp4')
FAIL_ROOT = ('https://huggingface.co/datasets/birbirll/vega-pipette-fail/resolve/main/'
             'episodes/ws_a2mem_chain_b0065')


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as source, path.open('wb') as output:
        while chunk := source.read(1024 * 1024):
            output.write(chunk)


def image_url(frame, max_side=640):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.thumbnail((max_side, max_side))
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=82)
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode()


def run_clip(config, episode_id, expected_phase, videos, frame_indexes):
    captures = [cv2.VideoCapture(str(path)) for path in videos]
    if any(not capture.isOpened() for capture in captures):
        raise RuntimeError('could not open a demo video')
    fps = captures[0].get(cv2.CAP_PROP_FPS)
    agent = ObserverAgent(replace(config, history_size=4), create_backend(config))
    results = []
    for frame_index in frame_indexes:
        images = []
        for capture in captures:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'could not decode frame {frame_index}')
            images.append(image_url(frame))
        results.append(agent.observe({
            'episode_id': episode_id,
            'frame_id': f'{episode_id}:{frame_index}',
            'timestamp_s': frame_index / fps,
            'expected_phase': expected_phase,
            'note': 'Views are ordered as supplied by the demo.',
            'images': images,
        }))
    for capture in captures:
        capture.release()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/g1_pipette.toml')
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/pipette-demo'))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)

    success_video = args.out_dir / 'success-episode-000000-head.mp4'
    failure_annotation = args.out_dir / 'failure-b0065-annotation.json'
    failure_ego = args.out_dir / 'failure-b0065-ego.mp4'
    failure_side = args.out_dir / 'failure-b0065-side.mp4'
    download(SUCCESS_VIDEO, success_video)
    download(FAIL_ROOT + '/annotation.json', failure_annotation)
    download(FAIL_ROOT + '/cams/ego_view.mp4', failure_ego)
    download(FAIL_ROOT + '/cams/side_view.mp4', failure_side)

    success = run_clip(config, 'success-real-000000', 'step1_pipette_pick',
                       [success_video], [0, 234, 468, 702, 935])
    failure = run_clip(config, 'failure-sim-b0065', 'step4_tube_return', [failure_ego, failure_side],
                       [1050, 1120, 1145, 1157, 1170])
    annotation = json.loads(failure_annotation.read_text())

    success_transition = WorkflowTracker(config.phases, config.max_retries).apply(
        'success-real-000000', success[-1])
    failure_transition = WorkflowTracker(
        config.phases, config.max_retries, start_phase='step4_tube_return').apply(
        'failure-sim-b0065', failure[-1])
    summary = {
        'model': config.model,
        'success_demo': {
            'ground_truth': {'phase': 'step1_pipette_pick', 'status': 'completed'},
            'prediction': success[-1],
            'episode_transition': success_transition,
        },
        'failure_demo': {
            'ground_truth': {
                'source_subtask': annotation['failure']['subtask'],
                'mapped_phase': 'step4_tube_return',
                'failure_kind': annotation['failure']['kind'],
                'expected_decision': annotation['coordinator']['decision'],
                'visible_row': annotation['failure']['visible_row'],
            },
            'prediction': failure[-1],
            'episode_transition': failure_transition,
        },
    }
    (args.out_dir / 'success-results.json').write_text(json.dumps(success, indent=2) + '\n')
    (args.out_dir / 'failure-results.json').write_text(json.dumps(failure, indent=2) + '\n')
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
