"""Prepare ten same-source videos for human review; never runs the VLM."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
from urllib.request import urlopen

import cv2

from scripts.demo_failure_videos import fit, text


SUCCESS_REPO = 'jren313/g1-pipette-2view-teleop0925-5task-eerel'
FAIL_REPO = 'birbirll/vega-pipette-fail'
HF = 'https://huggingface.co/datasets'
PHASES = (
    'step1_pipette_pick', 'step2_tube_pick', 'step3_pipette_aim_at_tube',
    'step4_tube_return', 'step5_pipette_return',
)
TASKS = (
    'Pick up the pipette from the holder on the right with the right hand.',
    'Pick up the tube from the blue rack on the left with the left hand.',
    'Aim the pipette held in the right hand at the tube held in the left hand.',
    'Put the tube held in the left hand back into the blue rack on the left.',
    'Put the pipette held in the right hand back into the holder on the right.',
)
VIEWS = ('ego', 'left_wrist', 'side')


def fetch(url, path):
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'DOWNLOAD {url}', flush=True)
    with urlopen(url, timeout=180) as source, path.open('wb') as output:
        while chunk := source.read(1024 * 1024):
            output.write(chunk)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def step_key(source_episode):
    match = re.fullmatch(r'(.+)_step([1-5])', source_episode)
    return (match.group(1), int(match.group(2))) if match else None


assert step_key('episode_0002_step5') == ('episode_0002', 5)


def build_manifest(output, cache, seed):
    metadata = cache / 'metadata'
    episodes_path = metadata / 'success_episodes.jsonl'
    provenance_path = metadata / 'success_provenance.jsonl'
    failures_path = metadata / 'failure_index.jsonl'
    fetch(f'{HF}/{SUCCESS_REPO}/resolve/main/meta/episodes.jsonl', episodes_path)
    fetch(f'{HF}/{SUCCESS_REPO}/resolve/main/meta/source_provenance.jsonl', provenance_path)
    fetch(f'{HF}/{FAIL_REPO}/resolve/main/index.jsonl', failures_path)

    episodes = {row['episode_index']: row for row in rows(episodes_path)}
    grouped = defaultdict(dict)
    for row in rows(provenance_path):
        parsed = step_key(row['source_episode'])
        if parsed:
            base, step = parsed
            grouped[(row['source_session'], base)][step] = row

    success_groups = []
    for (session, base), steps in grouped.items():
        if set(steps) != set(range(1, 6)):
            continue
        ordered = [steps[index] for index in range(1, 6)]
        if any(episodes[row['episode_index']]['tasks'][0] != TASKS[index]
               for index, row in enumerate(ordered)):
            continue
        gaps = [ordered[index + 1]['t_img_first_ms'] - ordered[index]['t_img_last_ms']
                for index in range(4)]
        if any(gap < 0 or gap > 100 for gap in gaps):
            continue
        success_groups.append((session, base, ordered, gaps))
    if len(success_groups) < 5:
        raise RuntimeError('fewer than five timestamp-continuous G1 workflows')

    failure_candidates = [
        row for row in rows(failures_path)
        if row.get('usable') and row['failure']['occurred']
        and row['failure']['visible_row'] is not None
        and all(f'{view}_view' in row['videos'] for view in VIEWS)
    ]
    rng = random.Random(seed)
    selected_success = rng.sample(success_groups, 5)
    selected_failure = rng.sample(failure_candidates, 5)

    samples = []
    for index, (session, base, workflow, gaps) in enumerate(selected_success, 1):
        phases = []
        for phase_index, source in enumerate(workflow):
            episode = source['episode_index']
            phases.append({
                'phase': PHASES[phase_index], 'episode_index': episode,
                'source_episode': source['source_episode'],
                'length': episodes[episode]['length'],
                'head': str(cache / 'g1' / f'episode_{episode:06d}' / 'head.mp4'),
                'left_wrist': str(cache / 'g1' / f'episode_{episode:06d}' / 'left_wrist.mp4'),
            })
        samples.append({
            'sample_id': f'success_{index:02d}', 'kind': 'success',
            'dataset': SUCCESS_REPO, 'source_session': session,
            'source_workflow': base, 'continuous_source': True,
            'inter_phase_gaps_ms': gaps, 'phases': phases,
        })
    for index, failure in enumerate(selected_failure, 1):
        videos = {
            view: str(cache / 'vega' / failure['episode'] / f'{view}.mp4')
            for view in VIEWS
        }
        samples.append({
            'sample_id': f'failure_{index:02d}', 'kind': 'failure',
            'dataset': FAIL_REPO, 'episode': failure['episode'],
            'continuous_source': True, 'rows': failure['rows'], 'fps': failure['fps'],
            'reference_label': failure['failure'], 'videos': videos,
        })
    manifest = {
        'seed': seed, 'inference_run': False,
        'selection_rule': ('success: same session/workflow step1-step5 with 0-100ms source gaps; '
                           'failure: one complete uncut Vega episode'),
        'samples': samples,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def prepare_sources(manifest):
    for sample in manifest['samples']:
        if sample['kind'] == 'success':
            for phase in sample['phases']:
                episode = phase['episode_index']
                fetch(f'{HF}/{SUCCESS_REPO}/resolve/main/videos/chunk-000/'
                      f'observation.images.rgb/episode_{episode:06d}.mp4', Path(phase['head']))
                fetch(f'{HF}/{SUCCESS_REPO}/resolve/main/videos/chunk-000/'
                      f'observation.images.wrist_left/episode_{episode:06d}.mp4',
                      Path(phase['left_wrist']))
        else:
            for view, local in sample['videos'].items():
                remote = f"episodes/{sample['episode']}/cams/{view}_view.mp4"
                fetch(f'{HF}/{FAIL_REPO}/resolve/main/{remote}', Path(local))


def render_frames(writer, paths, names, label, rotate_wrist=False):
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if any(not capture.isOpened() for capture in captures):
        raise RuntimeError(f'could not open {paths}')
    fps = captures[0].get(cv2.CAP_PROP_FPS)
    count = min(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures)
    output_count = max(1, round(count / fps * 30))
    decoded_index, decoded = -1, []
    for out_index in range(output_count):
        source_index = min(count - 1, round(out_index * fps / 30))
        while decoded_index < source_index:
            decoded = []
            for capture in captures:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f'could not decode {paths[0]}:{source_index}')
                decoded.append(frame)
            decoded_index += 1
        frames = [cv2.rotate(frame, cv2.ROTATE_180)
                  if rotate_wrist and name == 'left_wrist' else frame
                  for name, frame in zip(names, decoded)]
        main_index = names.index('side') if 'side' in names else 0
        canvas = fit(frames[main_index], 1280, 720)
        inset_x = 18
        for index, (name, frame) in enumerate(zip(names, frames)):
            if index == main_index:
                continue
            inset = fit(frame, 300, 170)
            canvas[18:188, inset_x:inset_x + 300] = inset
            cv2.rectangle(canvas, (inset_x, 18), (inset_x + 300, 188),
                          (230, 230, 230), 2)
            text(canvas, name.replace('_', ' ').upper(), (inset_x + 10, 44),
                 .45, (255, 255, 255), 2)
            inset_x += 316
        overlay = canvas.copy()
        cv2.rectangle(overlay, (18, 640), (1260, 705), (10, 14, 20), -1)
        cv2.addWeighted(overlay, .8, canvas, .2, 0, canvas)
        text(canvas, label, (36, 677), .62, (255, 255, 255), 2)
        text(canvas, f'SOURCE FRAME {source_index}', (1030, 677), .42,
             (210, 210, 210), 1)
        writer.write(canvas)
    for capture in captures:
        capture.release()


def render_sample(sample, directory):
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / 'sample.mp4'
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(f'could not create {output}')
    if sample['kind'] == 'success':
        for phase in sample['phases']:
            label = (f"G1 REAL  {sample['source_session']} / {sample['source_workflow']}  "
                     f"{phase['phase']}")
            render_frames(writer, [Path(phase['head']), Path(phase['left_wrist'])],
                          ['head', 'left_wrist'], label, rotate_wrist=True)
    else:
        label = f"VEGA SIM  {sample['episode']}  COMPLETE CONTINUOUS ROLLOUT"
        render_frames(writer, [Path(sample['videos'][view]) for view in VIEWS],
                      list(VIEWS), label)
    writer.release()
    (directory / 'source_manifest.json').write_text(json.dumps(sample, indent=2) + '\n')
    print(f'CREATED {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path,
                        default=Path('outputs/continuous-sample-review-seed-20261021'))
    parser.add_argument('--cache-dir', type=Path,
                        default=Path('outputs/continuous-source-cache'))
    parser.add_argument('--seed', type=int, default=20261021)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / 'manifest.json'
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists()
                else build_manifest(args.out_dir, args.cache_dir, args.seed))
    prepare_sources(manifest)
    for sample in manifest['samples']:
        render_sample(sample, args.out_dir / sample['sample_id'])


if __name__ == '__main__':
    main()
