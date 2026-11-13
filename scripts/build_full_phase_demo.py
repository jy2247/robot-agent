"""Render a five-phase G1 demo from recorded chain video and monitor predictions."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np


RESULTS = Path('/data/jliu3292/robot-agent-results/g1-five-vote-10chains')
DEFAULT_EXPORT = Path(__file__).resolve().parents[1] / 'demo/critic-comparison/exports/success_critic_demo.mp4'
PHASES = ('PIPETTE PICK', 'TUBE PICK', 'PIPETTE AIM', 'TUBE RETURN', 'PIPETTE RETURN')


def put(image, label, x, y, scale=.65, color=(235, 240, 240), thickness=1):
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def fit(frame, width, height):
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
    canvas = np.full((height, width, 3), (11, 17, 20), np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def render(chain, rows, output, fps):
    source = chain['source_videos']
    cameras = (source['observation.images.rgb'], source['observation.images.wrist_left'])
    boundaries = [0, *chain['ground_truth_transition_times_s'], chain['duration_s']]
    transitions = chain['phase_completion_times_s']
    captures = [[cv2.VideoCapture(item['path']) for item in camera] for camera in cameras]
    if not all(cap.isOpened() for camera in captures for cap in camera):
        raise RuntimeError('one or more source videos could not be opened')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='five-phase-demo-') as temp:
        intermediate = Path(temp) / 'intermediate.mp4'
        writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))
        row_index = -1
        for frame_index in range(int(chain['duration_s'] * fps)):
            t = frame_index / fps
            segment = next(i for i in range(5) if boundaries[i] <= t < boundaries[i + 1])
            local_t = t - boundaries[segment]
            views = []
            for camera_index in range(2):
                cap = captures[camera_index][segment]
                cap.set(cv2.CAP_PROP_POS_MSEC, local_t * 1000)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'video decode failed at {t:.2f}s, camera {camera_index}')
                if camera_index == 1:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                views.append(fit(frame, 610, 390))
            while row_index + 1 < len(rows) and rows[row_index + 1]['source_time_s'] <= t + .01:
                row_index += 1
            row = rows[row_index] if row_index >= 0 else None
            canvas = np.full((720, 1280, 3), (8, 13, 16), np.uint8)
            put(canvas, 'FULL FIVE-PHASE MONITOR RUN', 28, 42, .8, (194, 243, 125), 2)
            put(canvas, f'{t:05.1f} / {chain["duration_s"]:.1f} s', 1050, 42, .55)
            canvas[70:460, 28:638] = views[0]
            canvas[70:460, 642:1252] = views[1]
            put(canvas, 'OVERHEAD RGB', 40, 95, .52)
            put(canvas, 'LEFT WRIST (ROTATED)', 654, 95, .52)
            if row:
                phase_index = int(row['phase'][4]) - 1
                status = row['status'].replace('_', ' ').upper()
                confidence = row.get('confidence')
                put(canvas, f'EXPECTED P{phase_index + 1}: {PHASES[phase_index]}', 30, 502, .67, (238, 227, 161), 2)
                put(canvas, f'RECOGNIZED: {status}', 30, 539, .64, (171, 225, 241), 2)
                put(canvas, f'CONFIDENCE: {confidence:.0%}' if confidence is not None else 'CONFIDENCE: N/A', 525, 539, .58)
                if row.get('terminal_accepted') and row['status'] == 'success':
                    put(canvas, f'P{phase_index + 1} CONFIRMED', 900, 539, .58, (194, 243, 125), 2)
                elif row.get('terminal_confirmation_count'):
                    put(canvas, f'CONFIRMING {row["terminal_confirmation_count"]}/2', 900, 539, .52)
            else:
                put(canvas, 'WAITING FOR FIRST ASSESSMENT', 30, 519, .62)
            for i, label in enumerate(PHASES):
                left = 30 + i * 250
                phase_key = ('step1_pipette_pick', 'step2_tube_pick', 'step3_pipette_aim_at_tube', 'step4_tube_return', 'step5_pipette_return')[i]
                accepted = transitions.get(phase_key, float('inf')) <= t
                active = row is not None and row['phase'] == phase_key
                color = (194, 243, 125) if accepted else (171, 225, 241) if active else (65, 78, 83)
                cv2.rectangle(canvas, (left, 590), (left + 238, 634), color, 2)
                put(canvas, f'P{i + 1} {"CONFIRMED" if accepted else "ACTIVE" if active else "PENDING"}', left + 9, 618, .46, color)
                put(canvas, label, left + 4, 660, .40)
            cv2.line(canvas, (30, 688), (1250, 688), (65, 78, 83), 3)
            for boundary in transitions.values():
                if boundary <= t:
                    x = 30 + round(boundary / chain['duration_s'] * 1220)
                    cv2.circle(canvas, (x, 688), 5, (194, 243, 125), -1)
            x = 30 + round(t / chain['duration_s'] * 1220)
            cv2.circle(canvas, (x, 688), 8, (240, 240, 240), -1)
            writer.write(canvas)
        writer.release()
        for camera in captures:
            for cap in camera:
                cap.release()
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(intermediate), '-c:v', 'libx264',
                        '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=int, default=5)
    parser.add_argument('--output', type=Path, default=DEFAULT_EXPORT)
    parser.add_argument('--fps', type=int, default=12)
    args = parser.parse_args()
    summary = json.loads((RESULTS / 'summary.json').read_text())
    chain = summary['chains'][args.chain - 1]
    if not chain['detected_all_five_phases']:
        parser.error('selected chain did not confirm all five phases')
    rows = [json.loads(line) for line in (RESULTS / f'chain-{args.chain:02d}-predictions.jsonl').read_text().splitlines()]
    render(chain, rows, args.output, args.fps)
    print(f'Wrote {args.output}: {chain["duration_s"]:.2f}s, five confirmed phases')


if __name__ == '__main__':
    main()
