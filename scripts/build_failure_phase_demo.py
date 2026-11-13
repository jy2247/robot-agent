"""Render an explicitly labeled G1 phase-1 / VEGA phase-2 failure composite."""
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np

from build_full_phase_demo import fit, put, PHASES, RESULTS


ROOT = Path(__file__).resolve().parents[1]
FAILURE = Path('/data/jliu3292/robot-agent-results/vega-pipette-fail-5-three-camera/case-01-composite-global-safety-phase-aligned-freeze10s')
EXPORT = ROOT / 'demo/critic-comparison/exports/failure_critic_demo.mp4'
FPS = 12


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    chain = json.loads((RESULTS / 'summary.json').read_text())['chains'][4]
    p1_rows = read_rows(RESULTS / 'chain-05-predictions.jsonl')
    failure_summary = json.loads((FAILURE / 'summary.json').read_text())
    failure_rows = read_rows(FAILURE / 'predictions.jsonl')
    p1_end = chain['phase_completion_times_s']['step1_pipette_pick']
    p2_end = min(item['duration_s'] for item in failure_summary['source_videos'])
    total = p1_end + p2_end
    g1_sources = [chain['source_videos'][key][0]['path'] for key in ('observation.images.rgb', 'observation.images.wrist_left')]
    vega_sources = [item['path'] for item in failure_summary['source_videos']]
    g1_caps = [cv2.VideoCapture(path) for path in g1_sources]
    vega_caps = [cv2.VideoCapture(path) for path in vega_sources]
    if not all(cap.isOpened() for cap in g1_caps + vega_caps):
        raise RuntimeError('source video could not be opened')
    with tempfile.TemporaryDirectory(prefix='failure-phase-demo-') as temp:
        intermediate = Path(temp) / 'intermediate.mp4'
        writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (1280, 720))
        p1_index = p2_index = -1
        for frame_index in range(round(total * FPS)):
            t = frame_index / FPS
            in_p1 = t < p1_end
            local_t = t if in_p1 else t - p1_end
            caps = g1_caps if in_p1 else vega_caps
            frames = []
            for index, cap in enumerate(caps):
                cap.set(cv2.CAP_PROP_POS_MSEC, local_t * 1000)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'decode failed at {t:.2f}s, camera {index}')
                if in_p1 and index == 1:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                frames.append(frame)
            if in_p1:
                while p1_index + 1 < len(p1_rows) and p1_rows[p1_index + 1]['source_time_s'] <= local_t + .01:
                    p1_index += 1
                row = p1_rows[p1_index] if p1_index >= 0 else None
            else:
                while p2_index + 1 < len(failure_rows) and failure_rows[p2_index + 1]['source_time_s'] <= local_t + .01:
                    p2_index += 1
                row = failure_rows[p2_index] if p2_index >= 0 else None
            canvas = np.full((720, 1280, 3), (8, 13, 16), np.uint8)
            put(canvas, 'PHASE TIMELINE / FAILURE EXAMPLE', 28, 37, .70, (194, 243, 125), 2)
            put(canvas, 'ILLUSTRATIVE COMPOSITE: G1 P1 + VEGA P2', 28, 62, .46, (139, 182, 242), 1)
            put(canvas, f'{t:05.1f} / {total:.1f} s', 1060, 44, .51)
            if in_p1:
                canvas[85:465, 28:638] = fit(frames[0], 610, 380)
                canvas[85:465, 642:1252] = fit(frames[1], 610, 380)
                labels = ('G1 OVERHEAD', 'G1 LEFT WRIST')
            else:
                canvas[85:465, 28:638] = fit(frames[0], 610, 380)
                canvas[85:271, 642:1252] = fit(frames[1], 610, 186)
                canvas[279:465, 642:1252] = fit(frames[2], 610, 186)
                labels = ('VEGA EGO', 'VEGA LEFT / RIGHT WRIST')
            put(canvas, labels[0], 40, 110, .48)
            put(canvas, labels[1], 654, 110, .48)
            phase = 1 if in_p1 else 2
            put(canvas, f'EXPECTED P{phase}: {PHASES[phase - 1]}', 30, 502, .65, (238, 227, 161), 2)
            verdict = row if in_p1 else row['phase'] if row else None
            if verdict:
                put(canvas, 'RECOGNIZED: ' + verdict['status'].replace('_', ' ').upper(), 30, 538, .57, (171, 225, 241), 2)
                confidence = verdict.get('confidence')
                put(canvas, f'CONFIDENCE: {confidence:.0%}' if confidence is not None else 'CONFIDENCE: N/A', 525, 538, .53)
            else:
                put(canvas, 'WAITING FOR FIRST ASSESSMENT', 30, 538, .52)
            failed = not in_p1 and row is not None and row['safety']['failure_accepted']
            if failed:
                put(canvas, 'GLOBAL CRITIC: STOP', 905, 538, .56, (95, 102, 255), 2)
            elif not in_p1 and row:
                count = row['safety']['failure_confirmation_count']
                put(canvas, f'SAFETY CONFIRMATION: {count}/2', 855, 538, .43)
            elif in_p1 and row and row['terminal_accepted']:
                put(canvas, 'P1 CONFIRMED', 975, 538, .52, (194, 243, 125), 2)
            for index, label in enumerate(PHASES):
                left = 30 + index * 250
                state = 'CONFIRMED' if index == 0 and t >= p1_end else 'ACTIVE' if index == phase - 1 else 'NOT REACHED'
                if failed and index == 1:
                    state = 'STOPPED'
                color = (95, 102, 255) if state == 'STOPPED' else (194, 243, 125) if state == 'CONFIRMED' else (171, 225, 241) if state == 'ACTIVE' else (65, 78, 83)
                cv2.rectangle(canvas, (left, 590), (left + 238, 634), color, 2)
                put(canvas, f'P{index + 1} {state}', left + 8, 618, .42, color)
                put(canvas, label, left + 4, 660, .40)
            cv2.line(canvas, (30, 688), (1250, 688), (65, 78, 83), 3)
            splice_x = 30 + round(p1_end / total * 1220)
            cv2.line(canvas, (splice_x, 678), (splice_x, 699), (139, 182, 242), 2)
            cv2.circle(canvas, (30 + round(t / total * 1220), 688), 8, (240, 240, 240), -1)
            writer.write(canvas)
        writer.release()
        for cap in g1_caps + vega_caps:
            cap.release()
        EXPORT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(intermediate), '-c:v', 'libx264',
                        '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(EXPORT)], check=True)
    print(f'Wrote {EXPORT}: {total:.2f}s; source splice at {p1_end:.2f}s')


if __name__ == '__main__':
    main()
