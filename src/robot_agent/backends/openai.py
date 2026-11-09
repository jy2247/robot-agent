import json
import os
from urllib.request import Request, urlopen


class OpenAIBackend:
    """One-shot client for a local OpenAI-compatible multimodal server."""

    def __init__(self, config):
        self.config = config
        self.last_usage = {}
        self.last_raw = None

    def generate(self, messages):
        config = self.config
        self.last_usage = {}
        self.last_raw = None
        payload = {
            'model': config.model,
            'messages': messages,
            'temperature': config.temperature,
            'top_p': config.top_p,
            'max_tokens': config.max_tokens,
            'stream': False,
        }
        if config.chat_template_kwargs:
            payload['chat_template_kwargs'] = config.chat_template_kwargs
        headers = {'Content-Type': 'application/json'}
        key = os.getenv('ROBOT_VLM_API_KEY')
        if key:
            headers['Authorization'] = 'Bearer ' + key
        request = Request(
            config.base_url.rstrip('/') + '/chat/completions',
            data=json.dumps(payload).encode(), headers=headers, method='POST')
        with urlopen(request, timeout=config.timeout_s) as response:
            response_body = json.load(response)
        self.last_usage = response_body.get('usage', {})
        choices = response_body.get('choices')
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError('model must return exactly one choice')
        if choices[0].get('finish_reason') == 'length':
            raise ValueError('model output truncated')
        content = choices[0].get('message', {}).get('content')
        if not isinstance(content, str) or not content.strip():
            raise ValueError('model returned no content')
        self.last_raw = content
        return content
