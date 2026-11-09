from io import BytesIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from robot_agent.backends.openai import OpenAIBackend
from robot_agent.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class BackendTests(unittest.TestCase):
    def test_hot_path_sends_one_short_non_thinking_choice(self):
        config = load_config(str(ROOT / 'configs/monitor.toml'))
        response = {'choices': [{'finish_reason': 'stop', 'message': {
            'content': '{"status":"in_progress","confidence":0.5}'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 8}}
        captured = {}

        def fake_urlopen(request, timeout):
            captured.update(json.loads(request.data))
            self.assertEqual(timeout, 8)
            return Response(json.dumps(response).encode())

        backend = OpenAIBackend(config)
        with patch('robot_agent.backends.openai.urlopen', fake_urlopen):
            output = backend.generate([{'role': 'user', 'content': 'x'}])
        self.assertIn('in_progress', output)
        self.assertEqual(backend.last_raw, output)
        self.assertEqual(captured['max_tokens'], 32)
        self.assertEqual(captured['temperature'], 0)
        self.assertFalse(captured['chat_template_kwargs']['enable_thinking'])
        self.assertNotIn('n', captured)
        self.assertNotIn('response_format', captured)


if __name__ == '__main__':
    unittest.main()
