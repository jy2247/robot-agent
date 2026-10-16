from dataclasses import replace
import json
from pathlib import Path
import unittest

from robot_agent.agent import ObserverAgent
from robot_agent.backends.mock import MockBackend
from robot_agent.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(str(ROOT / 'configs/default.toml'))
        self.agent = ObserverAgent(self.config, MockBackend())
        self.clock = 0

    def observe(self, phase, status='completed', episode='test'):
        self.clock += 1
        return self.agent.observe({'episode_id': episode, 'frame_id': str(self.clock),
                                   'timestamp_s': self.clock, 'note': f'phase={phase} status={status}'})

    def test_all_phases_need_consecutive_confirmation(self):
        for phase in self.config.phases:
            self.assertFalse(self.observe(phase)['task_completed'])
            result = self.observe(phase)
            self.assertTrue(result['phase_completion_confirmed'])
            self.assertEqual(result['task_completed'], phase == self.config.phases[-1])
        self.assertEqual(result['recommendation'], 'task_completed')

    def test_failure_interrupts_completion_streak(self):
        self.observe('approach')
        self.assertEqual(self.observe('approach', 'failed')['recommendation'], 'review')
        self.assertFalse(self.observe('approach')['phase_completion_confirmed'])

    def test_final_phase_alone_does_not_finish_task(self):
        self.observe('release_retract')
        result = self.observe('release_retract')
        self.assertFalse(result['task_completed'])
        self.assertEqual(result['recommendation'], 'review')

    def test_episode_isolation_and_bounded_history(self):
        for _ in range(10):
            self.observe('approach', 'in_progress')
        self.assertEqual(len(self.agent.history), self.config.history_size)
        with self.assertRaises(ValueError):
            self.observe('approach', episode='another')
        self.agent.reset()
        self.assertEqual(len(self.agent.history), 0)
        self.observe('approach', episode='another')

    def test_out_of_order_frames_rejected_without_mutation(self):
        self.observe('approach')
        with self.assertRaises(ValueError):
            self.agent.observe({'episode_id': 'test', 'frame_id': 'old', 'timestamp_s': 0})
        self.assertEqual(len(self.agent.history), 1)

    def test_bad_backend_outputs_are_uncertain(self):
        class Broken:
            def __init__(self, raw):
                self.raw = raw
            def generate(self, messages):
                if isinstance(self.raw, Exception):
                    raise self.raw
                return self.raw
        for bad in ('not json', '{"phase":"invented"}', TimeoutError(), json.dumps({
                'phase': 'approach', 'status': 'completed', 'confidence': float('nan'),
                'errors': [], 'evidence': ['f1']})):
            with self.subTest(bad=repr(bad)):
                self.agent = ObserverAgent(self.config, Broken(bad))
                result = self.observe('approach')
                self.assertEqual(result['status'], 'uncertain')
                self.assertFalse(result['task_completed'])

    def test_low_confidence_cannot_confirm(self):
        self.agent = ObserverAgent(replace(self.config, confidence_threshold=0.99), MockBackend())
        self.observe('approach')
        self.assertFalse(self.observe('approach')['phase_completion_confirmed'])


if __name__ == '__main__':
    unittest.main()
