import json


class MockBackend:
    """Pipeline smoke test; never claims a visual terminal result."""

    def generate(self, messages):
        return json.dumps({'status': 'in_progress', 'confidence': 0.5})
