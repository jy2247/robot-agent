from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from robot_agent.agent import ObserverAgent
from robot_agent.backends.mock import MockBackend
from robot_agent.backends.openai import OpenAIBackend
from robot_agent.config import load_config
from robot_agent.context import messages
from robot_agent.server import make_server

CONFIG = load_config(str(Path(__file__).resolve().parents[1] / 'configs/default.toml'))


class HTTPTests(unittest.TestCase):
    def start(self, server):
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        return f'http://127.0.0.1:{server.server_port}'

    def post(self, url, payload):
        request = Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=2) as response:
            return json.load(response)

    def test_fc_roundtrip_and_reset(self):
        url = self.start(make_server(ObserverAgent(CONFIG, MockBackend()), port=0))
        payload = {'episode_id': 'one', 'frame_id': 'f1', 'timestamp_s': 1, 'note': 'phase=grasp status=in_progress'}
        result = self.post(url + '/observe', payload)
        self.assertEqual(result['phase'], 'grasp')
        with self.assertRaises(HTTPError) as error:
            self.post(url + '/observe', payload)
        self.assertEqual(error.exception.code, 400)
        self.post(url + '/reset', {})
        self.assertEqual(self.post(url + '/observe', payload)['status'], 'in_progress')

    def test_multimodal_backend_wire_format(self):
        captured = []
        class FakeModel(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                raw = json.dumps({'phase': 'approach', 'status': 'in_progress', 'confidence': 0.9,
                                  'errors': [], 'evidence': ['f1']})
                body = json.dumps({'choices': [{'message': {'content': raw}}]}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)
        url = self.start(HTTPServer(('127.0.0.1', 0), FakeModel))
        config = replace(CONFIG, base_url=url + '/v1', model='test-vlm', backend='openai')
        obs = {'episode_id': 'one', 'frame_id': 'f1', 'timestamp_s': 1,
               'images': ['data:image/png;base64,aGVsbG8=']}
        result = ObserverAgent(config, OpenAIBackend(config)).observe(obs)
        self.assertEqual(result['phase'], 'approach')
        path, body = captured[0]
        self.assertEqual(path, '/v1/chat/completions')
        self.assertEqual(body['model'], 'test-vlm')
        self.assertEqual(body['messages'][1]['content'][1]['image_url']['url'], obs['images'][0])


if __name__ == '__main__':
    unittest.main()
