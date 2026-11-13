import json
import os
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from ..schemas import monitor_verdict
from urllib.request import Request, urlopen


class OpenAIBackend:
    """One-shot client for a local OpenAI-compatible multimodal server."""

    def __init__(self, config):
        self.config = config
        self.last_usage = {}
        self.last_raw = None

    def generate(self, messages):
        if self.config.votes == 5:
            return self.generate_votes(messages)
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

    def generate_votes(self, messages):
        from dataclasses import replace
        def vote(_):
            client = OpenAIBackend(replace(self.config, votes=1))
            try:
                raw = client.generate(messages)
                return {'verdict': monitor_verdict(raw), 'raw': raw,
                        'usage': client.last_usage, 'error': None}
            except Exception as exc:
                return {'verdict': None, 'raw': client.last_raw,
                        'usage': client.last_usage, 'error': type(exc).__name__}
        with ThreadPoolExecutor(max_workers=5) as pool:
            self.last_votes = list(pool.map(vote, range(5)))
        valid = [v['verdict'] for v in self.last_votes if v['error'] is None]
        counts = Counter(v['status'] for v in valid)
        status, count = counts.most_common(1)[0] if counts else ('in_progress', 0)
        self.last_vote_count = count if count >= 3 else 0
        if count < 3:
            status = 'in_progress'
        agreeing = [v['confidence'] for v in valid if v['status'] == status]
        result = {'status': status, 'confidence': sum(agreeing) / len(agreeing) if agreeing else 0.0}
        self.last_usage = {key: sum(v['usage'].get(key, 0) or 0 for v in self.last_votes)
                           for key in ('prompt_tokens', 'completion_tokens')}
        self.last_usage['prompt_tokens_details'] = {'cached_tokens': sum(
            (v['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0
            for v in self.last_votes)}
        self.last_raw = json.dumps(result)
        return self.last_raw
