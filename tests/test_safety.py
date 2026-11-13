from dataclasses import replace
import json
from pathlib import Path
import unittest

from robot_agent.config import load_config
from robot_agent.safety import GlobalSafetyCritic, object_state_verdict


ROOT = Path(__file__).resolve().parents[1]


class SequenceBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def generate(self, prompt):
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)


class GlobalSafetyTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(load_config(str(ROOT / 'configs/monitor.toml')), camera_count=1)
        self.clock = 0

    def payload(self):
        self.clock += 1
        return {
            'episode_id': 'e', 'frame_id': str(self.clock),
            'timestamp_s': self.clock, 'expected_phase': self.config.phases[1],
            'images': ['data:image/jpeg;base64,eA=='] * self.config.image_count,
        }

    def test_parser_enforces_exact_support_fields(self):
        raw = '```json\n{"pipette_support":"hand","tube_support":"rack"}\n```'
        self.assertEqual(object_state_verdict(raw), {
            'pipette_support': 'hand', 'tube_support': 'rack'})
        for bad in ('{}', '{"pipette_support":"table","tube_support":"rack"}',
                    '{"pipette_support":"hand","tube_support":"rack","x":1}'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                object_state_verdict(bad)

    def test_two_consecutive_unsafe_states_are_accepted(self):
        unsafe = {'pipette_support': 'work_surface', 'tube_support': 'hand'}
        critic = GlobalSafetyCritic(self.config, SequenceBackend([unsafe, unsafe]))
        first = critic.observe(self.payload())
        second = critic.observe(self.payload())
        self.assertEqual(first['reason'], 'PIPETTE_ON_UNINTENDED_SURFACE')
        self.assertFalse(first['failure_accepted'])
        self.assertTrue(second['failure_accepted'])

    def test_unknown_and_backend_error_abstain(self):
        critic = GlobalSafetyCritic(self.config, SequenceBackend([
            {'pipette_support': 'unknown', 'tube_support': 'rack'}, TimeoutError()]))
        self.assertEqual(critic.observe(self.payload())['status'], 'unknown')
        result = critic.observe(self.payload())
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['backend_error_type'], 'TimeoutError')


if __name__ == '__main__':
    unittest.main()
