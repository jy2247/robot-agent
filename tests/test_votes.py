from dataclasses import replace
import unittest
from unittest.mock import patch
from robot_agent.config import load_config
from robot_agent.backends.openai import OpenAIBackend


class VoteTests(unittest.TestCase):
    def test_errors_abstain_without_reducing_threshold(self):
        backend = OpenAIBackend(replace(load_config('configs/monitor.toml'), votes=5))
        outputs = ['{"status":"success","confidence":0.9}'] * 2 + [TimeoutError()] * 3
        with patch.object(OpenAIBackend, 'generate', side_effect=outputs):
            result = backend.generate_votes([])
        self.assertIn('in_progress', result)
        self.assertEqual(backend.last_vote_count, 0)
        self.assertEqual(sum(v['error'] is not None for v in backend.last_votes), 3)
