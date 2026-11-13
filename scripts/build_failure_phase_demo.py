"""Render the single-source VEGA failure with a three-second final-frame hold."""
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np

from build_full_phase_demo import fit, put, PHASES

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/data/jliu3292/robot-agent-data/vega-pipette-fail-5')
RESULTS = Path('/data/jliu3292/robot-agent-results/vega-pipette-fail-5-three-camera/case-01-composite-global-safety-phase-aligned-freeze10s')
EXPORT = ROOT / 'demo/critic-comparison/exports/failure_critic_demo.mp4'
STEM = 'vega_fail_01_S2_dropped_ws_bridge_v31_30k_b0828'
FPS = 12
HOLD_S = 3.0


def main():
    rows = [json.loads(line) for line in (RESULTS / 'predictions.jsonl').read_text().splitlines() if line.strip()]
    caps = [cv2.VideoCapture(str(SOURCE / f'{STEM}_{name}.mp4')) for name in ('ego', 'left_wrist', 'right_wrist')]
    if not all(cap.isOpened() for cap in caps):
        raise RuntimeError('VEGA source video could not be opened')
    source_duration = min(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) for cap in caps)
    duration = source_duration + HOLD_S
    # Original inference sampled a ten-second frozen tail. Its identical-frame assessments
    # are replayed across the shorter hold, preserving verdict order and confidence values.
    times = [(min(row['source_time_s'], source_duration) +
              max(0, row['source_time_s'] - source_duration) * HOLD_S / 10, row) for row in rows]
    with tempfile.TemporaryDirectory(prefix='failure-phase-demo-') as temp:
        intermediate = Path(temp) / 'intermediate.mp4'
        writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (1280, 720))
        row_index = -1
        for frame_index in range(round(duration * FPS)):
            t = frame_index / FPS
            frames = []
            for camera_index, cap in enumerate(caps):
                cap.set(cv2.CAP_PROP_POS_MSEC, min(t, source_duration - 1 / 30) * 1000)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'decode failed at {t:.2f}s camera {camera_index}')
                frames.append(frame)
            while row_index + 1 < len(times) and times[row_index + 1][0] <= t + .01:
                row_index += 1
            row = times[row_index][1] if row_index >= 0 else None
            canvas = np.full((720, 1280, 3), (8, 13, 16), np.uint8)
            put(canvas, 'VEGA PHASE-2 FAILURE / GLOBAL SAFETY CRITIC', 28, 37, .68, (194, 243, 125), 2)
            put(canvas, 'SINGLE VEGA RECORDING | FINAL-FRAME HOLD: 3 S', 28, 62, .45, (139, 182, 242))
            put(canvas, f'{t:05.1f} / {duration:.1f} s', 1060, 44, .51)
            canvas[85:465, 28:638] = fit(frames[0], 610, 380)
            canvas[85:271, 642:1252] = fit(frames[1], 610, 186)
            canvas[279:465, 642:1252] = fit(frames[2], 610, 186)
            put(canvas, 'VEGA EGO', 40, 110, .48)
            put(canvas, 'VEGA LEFT / RIGHT WRIST', 654, 110, .48)
            put(canvas, 'EXPECTED P2: TUBE PICK', 30, 502, .65, (238, 227, 161), 2)
            if row:
                verdict = row['phase']
                put(canvas, 'RECOGNIZED: ' + verdict['status'].replace('_', ' ').upper(), 30, 538, .57, (171, 225, 241), 2)
                confidence = verdict.get('confidence')
                put(canvas, f'CONFIDENCE: {confidence:.0%}' if confidence is not None else 'CONFIDENCE: N/A', 525, 538, .53)
            else:
                put(canvas, 'WAITING FOR FIRST ASSESSMENT', 30, 538, .52)
            failed = row is not None and row['safety']['failure_accepted']
            if failed:
                put(canvas, 'GLOBAL CRITIC: STOP', 905, 538, .56, (95, 102, 255), 2)
            elif row:
                put(canvas, f'SAFETY CONFIRMATION: {row["safety"]["failure_confirmation_count"]}/2', 855, 538, .43)
            for index, label in enumerate(PHASES):
                left = 30 + index * 250
                state = 'PRIOR / NOT SHOWN' if index == 0 else 'STOPPED' if index == 1 and failed else 'ACTIVE' if index == 1 else 'NOT REACHED'
                color = (95, 102, 255) if state == 'STOPPED' else (171, 225, 241) if state == 'ACTIVE' else (65, 78, 83)
                cv2.rectangle(canvas, (left, 590), (left + 238, 634), color, 2)
                put(canvas, f'P{index + 1} {state}', left + 8, 618, .40, color)
                put(canvas, label, left + 4, 660, .40)
            cv2.line(canvas, (30, 688), (1250, 688), (65, 78, 83), 3)
            hold_x = 30 + round(source_duration / duration * 1220)
            cv2.line(canvas, (hold_x, 678), (hold_x, 699), (139, 182, 242), 2)
            cv2.circle(canvas, (30 + round(t / duration * 1220), 688), 8, (240, 240, 240), -1)
            writer.write(canvas)
        writer.release()
        for cap in caps:
            cap.release()
        EXPORT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(intermediate), '-c:v', 'libx264',
                        '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(EXPORT)], check=True)
    print(f'Wrote {EXPORT}: {duration:.2f}s; three-second hold begins at {source_duration:.2f}s')


if __name__ == '__main__':
    main()
