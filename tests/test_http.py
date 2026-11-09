from dataclasses import replace
import json
from pathlib import Path
from threading import Thread
import unittest
from urllib.request import Request, urlopen

from robot_agent.agent import MonitorAgent
from robot_agent.backends.mock import MockBackend
from robot_agent.config import load_config
from robot_agent.server import make_server


ROOT = Path(__file__).resolve().parents[1]


class HttpTests(unittest.TestCase):
    def setUp(self):
        config = replace(load_config(str(ROOT / 'configs/monitor.toml')), backend='mock')
        self.server = make_server(MonitorAgent(config, MockBackend()), '127.0.0.1', 0)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.images = ['data:image/jpeg;base64,eA=='] * config.image_count

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, path, body):
        request = Request(self.base + path, data=json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        with urlopen(request) as response:
            return json.load(response)

    def test_health_observe_and_reset(self):
        with urlopen(self.base + '/health') as response:
            self.assertEqual(json.load(response)['status'], 'ok')
        result = self.post('/observe', {
            'episode_id': 'e', 'frame_id': '1', 'timestamp_s': 1,
            'expected_phase': 'step1_pipette_pick', 'images': self.images})
        self.assertEqual(result['status'], 'in_progress')
        self.assertEqual(self.post('/reset', {}), {'status': 'reset'})


if __name__ == '__main__':
    unittest.main()
