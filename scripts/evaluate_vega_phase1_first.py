"""Run the VEGA failure clip from phase 1, advancing only on accepted judgments."""
from dataclasses import replace
import json
from pathlib import Path

from benchmark_single_vote import prepare_windows
from robot_agent.agent import MonitorAgent
from robot_agent.backends import create_backend
from robot_agent.config import load_config
from robot_agent.safety import GlobalSafetyCritic


ROOT = Path('/data/jliu3292/robot-agent-results/vega-phase1-first-3s')


def main():
    config = replace(load_config('configs/monitor.toml'), camera_count=3, votes=1, temperature=0.0)
    videos = [ROOT / 'videos' / f'{camera}.mp4' for camera in ('ego', 'left_wrist', 'right_wrist')]
    # A fresh 1.75-second window every 0.4 seconds, as in the five-phase evaluations.
    times = [round(1.75 + index * 0.4, 3) for index in range(42)]
    windows = prepare_windows(videos, times, config, set())
    phase_backend, safety_backend = create_backend(config), create_backend(config)
    phase_agent = MonitorAgent(config, phase_backend)
    safety_agent = GlobalSafetyCritic(config, safety_backend)
    expected = config.phases[0]
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'predictions.jsonl').open('w') as output:
        for index, (time_s, images) in enumerate(zip(times, windows)):
            payload = {'episode_id': 'vega-failure-phase1-first', 'frame_id': f'window-{index:04d}',
                       'timestamp_s': time_s, 'expected_phase': expected, 'images': images}
            phase = phase_agent.observe(payload)
            safety = safety_agent.observe(payload)
            row = {'assessment_index': index, 'source_time_s': time_s,
                   'expected_phase_before_result': expected, 'phase_raw_model_output': phase_backend.last_raw,
                   'phase': phase, 'safety_raw_model_output': safety_backend.last_raw, 'safety': safety}
            output.write(json.dumps(row) + '\n')
            output.flush()
            print(json.dumps({'t': time_s, 'phase': expected, 'status': phase['status'],
                              'confidence': phase['confidence'], 'accepted': phase['terminal_accepted'],
                              'safety': safety['status'], 'stop': safety['failure_accepted']}), flush=True)
            if safety['failure_accepted'] or phase['decision'] == 'stop':
                break
            if phase['decision'] == 'advance':
                current = config.phases.index(expected)
                if current + 1 < len(config.phases):
                    expected = config.phases[current + 1]


if __name__ == '__main__':
    main()
