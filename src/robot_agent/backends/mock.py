import json


class MockBackend:
    """Scripted plumbing demo; reads explicit tags, does not inspect images."""

    def generate(self, messages):
        frames = [json.loads(x['text']) for x in messages[-1]['content']
                  if x['type'] == 'text' and x['text'].startswith('{')]
        current = frames[-1]
        tags = dict(part.split('=', 1) for part in current.get('note', '').split() if '=' in part)
        status = tags.get('status', 'uncertain')
        error = tags.get('error', 'stalled')
        return json.dumps({'phase': tags.get('phase', 'unknown'), 'status': status,
                           'confidence': 0.95 if status != 'uncertain' else 0.2,
                           'errors': [error] if status == 'failed' else [],
                           'evidence': [current['frame_id'] + ': scripted mock tag, not visual evidence']})
