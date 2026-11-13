from dataclasses import replace
import json
from pathlib import Path
import unittest

from robot_agent.agent import MonitorAgent
from robot_agent.config import load_config
from robot_agent.context import messages
from robot_agent.schemas import monitor_verdict


ROOT = Path(__file__).resolve().parents[1]


class SequenceBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0
        self.last_usage = {'prompt_tokens': 100, 'completion_tokens': 12,
                           'prompt_tokens_details': {'cached_tokens': 50}}

    def generate(self, prompt):
        self.calls += 1
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)


def wire(status='in_progress', confidence=.9):
    return {'status': status, 'confidence': confidence}


class MonitorTests(unittest.TestCase):
    def test_five_vote_acceptance(self):
        for count, immediate in ((3, False), (4, True), (5, True)):
            backend = SequenceBackend([wire('success', .4), wire('success', .4)])
            backend.last_vote_count = count
            agent = MonitorAgent(replace(self.config, votes=5), backend)
            self.assertEqual(agent.observe(self.payload())['terminal_accepted'], immediate)
            self.assertTrue(agent.observe(self.payload())['terminal_accepted'])

    def setUp(self):
        self.config = load_config(str(ROOT / 'configs/monitor.toml'))
        self.images = ['data:image/jpeg;base64,eA=='] * self.config.image_count
        self.clock = 0

    def payload(self, phase=None, terminal=False):
        self.clock += 1
        return {'episode_id': 'e', 'frame_id': str(self.clock),
                'timestamp_s': self.clock,
                'expected_phase': phase or self.config.phases[0],
                'end_of_episode': terminal, 'images': self.images}

    def test_single_request_and_two_window_terminal_confirmation(self):
        backend = SequenceBackend([wire('success'), wire('success')])
        agent = MonitorAgent(self.config, backend)
        first = agent.observe(self.payload())
        second = agent.observe(self.payload())
        self.assertEqual(backend.calls, 2)
        self.assertFalse(first['terminal_accepted'])
        self.assertTrue(second['terminal_accepted'])
        self.assertEqual(second['decision'], 'advance')

    def test_in_progress_or_phase_change_resets_terminal_streak(self):
        backend = SequenceBackend([
            wire('success'), wire('in_progress'), wire('success'), wire('success')])
        agent = MonitorAgent(self.config, backend)
        self.assertEqual(agent.observe(self.payload())['terminal_confirmation_count'], 1)
        self.assertEqual(agent.observe(self.payload())['terminal_confirmation_count'], 0)
        other = self.config.phases[1]
        self.assertEqual(agent.observe(self.payload(other))['terminal_confirmation_count'], 1)
        self.assertTrue(agent.observe(self.payload(other))['terminal_accepted'])

    def test_backend_error_is_not_physical_failure(self):
        agent = MonitorAgent(self.config, SequenceBackend([TimeoutError()]))
        result = agent.observe(self.payload())
        self.assertEqual(result['status'], 'in_progress')
        self.assertEqual(result['confidence'], 0)
        self.assertEqual(result['decision'], 'continue')
        self.assertEqual(result['backend_error_type'], 'TimeoutError')

    def test_accepted_failure_revokes_completion(self):
        backend = SequenceBackend([
            wire('success'), wire('success'), wire('fail'), wire('fail')])
        agent = MonitorAgent(self.config, backend)
        agent.observe(self.payload())
        self.assertIn(self.config.phases[0], agent.observe(self.payload())['completed_phases'])
        agent.observe(self.payload())
        result = agent.observe(self.payload())
        self.assertEqual(result['completed_phases'], [])
        self.assertEqual(result['failed_phases'], [self.config.phases[0]])
        self.assertEqual(result['decision'], 'stop')

    def test_terminal_requires_all_phases(self):
        agent = MonitorAgent(self.config, SequenceBackend([wire()]))
        result = agent.observe(self.payload(terminal=True))
        self.assertEqual(result['overall_status'], 'failed')
        self.assertEqual(result['overall_reason'], 'INCOMPLETE_AT_END')
        with self.assertRaises(ValueError):
            agent.observe(self.payload())

    def test_input_is_strict_and_does_not_mutate_on_rejection(self):
        agent = MonitorAgent(self.config, SequenceBackend([wire()]))
        with self.assertRaises(ValueError):
            agent.observe({**self.payload(), 'images': self.images[:-1]})
        self.assertIsNone(agent.episode_id)
        agent.observe(self.payload())
        with self.assertRaises(ValueError):
            agent.observe({**self.payload(), 'timestamp_s': 1})

    def test_prompt_has_only_active_phase_and_stable_metadata(self):
        phase = self.config.phases[1]
        prompt = messages(self.config, self.payload(phase))
        system = prompt[0]['content']
        self.assertIn(self.config.criteria[1], system)
        self.assertNotIn(self.config.criteria[0], system)
        self.assertNotIn('timestamp_s', json.dumps(prompt))
        images = [part for part in prompt[1]['content'] if part['type'] == 'image_url']
        self.assertEqual(len(images), 24)

    def test_parser_tolerates_wrapper_but_enforces_two_fields(self):
        parsed = monitor_verdict('text ```json\n{"status":"success","confidence":0.8}\n```')
        self.assertEqual(parsed, wire('success', .8))
        for bad in ('no json', '{"status":"success"}',
                    '{"status":"done","confidence":0.8}',
                    '{"status":"success","confidence":2}'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                monitor_verdict(bad)

    def test_config_is_single_vote_latency_profile(self):
        self.assertEqual(self.config.image_count, 24)
        self.assertEqual((self.config.image_width, self.config.image_height), (448, 256))
        self.assertEqual(self.config.max_tokens, 32)
        self.assertEqual(self.config.temperature, 0)
        self.assertFalse(self.config.chat_template_kwargs['enable_thinking'])


if __name__ == '__main__':
    unittest.main()
