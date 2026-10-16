import json
import os
from urllib.request import Request, urlopen


class OpenAIBackend:
    """Client for a separately deployed multimodal Chat Completions server."""

    def __init__(self, config):
        self.config = config
        if config.model == 'replace-with-actual-served-model-name':
            raise ValueError('Set ROBOT_VLM_MODEL to the actual server model name')

    def generate(self, messages):
        c = self.config
        payload = {'model': c.model, 'messages': messages, 'temperature': 0,
                   'max_tokens': c.max_tokens, 'stream': False}
        headers = {'Content-Type': 'application/json'}
        key = os.getenv('ROBOT_VLM_API_KEY')
        if key:
            headers['Authorization'] = 'Bearer ' + key
        req = Request(c.base_url.rstrip('/') + '/chat/completions',
                      data=json.dumps(payload).encode(), headers=headers, method='POST')
        with urlopen(req, timeout=c.timeout_s) as response:
            result = json.load(response)
        return result['choices'][0]['message']['content']
