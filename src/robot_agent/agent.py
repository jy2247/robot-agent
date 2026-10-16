from collections import deque
import time

from .context import messages
from .schemas import observation, verdict


class ObserverAgent:
    """One episode per instance; callers must serialize access."""

    def __init__(self, config, backend):
        self.config, self.backend = config, backend
        self.history = deque(maxlen=config.history_size)
        self.episode_id = None
        self.previous = None
        self.last_timestamp = -1
        self.last_frame_id = None
        self.completed = set()
        self.candidate = None
        self.streak = 0

    def reset(self):
        self.__init__(self.config, self.backend)

    def observe(self, payload):
        started = time.perf_counter()
        obs = observation(payload)
        if self.episode_id is not None and self.episode_id != obs['episode_id']:
            raise ValueError('episode mismatch: reset before starting a new episode')
        if obs['timestamp_s'] <= self.last_timestamp or obs['frame_id'] == self.last_frame_id:
            raise ValueError('frames must be distinct with strictly increasing timestamps')
        self.episode_id = obs['episode_id']
        inference_start = time.perf_counter()
        error = None
        try:
            raw = self.backend.generate(messages(self.config, self.history, obs, self.previous))
            result = verdict(raw, self.config.phases)
        except Exception as exc:
            # Do not leak credentials or raw provider response bodies into FC output.
            error = type(exc).__name__
            result = {'phase': 'unknown', 'status': 'uncertain', 'confidence': 0.0,
                      'errors': ['backend_or_parse_error'], 'evidence': []}
        inference_ms = (time.perf_counter() - inference_start) * 1000
        valid_completion = (result['status'] == 'completed'
                            and result['confidence'] >= self.config.confidence_threshold)
        candidate = result['phase'] if valid_completion else None
        self.streak = self.streak + 1 if candidate and candidate == self.candidate else int(bool(candidate))
        self.candidate = candidate
        phase_confirmed = self.streak >= self.config.completion_confirmations
        # Only sequential confirmations count toward whole-task completion.
        next_phase = next((p for p in self.config.phases if p not in self.completed), None)
        if phase_confirmed and candidate == next_phase:
            self.completed.add(candidate)
        task_complete = (len(self.completed) == len(self.config.phases)
                         and phase_confirmed and candidate == self.config.phases[-1])
        recommendation = 'continue'
        if result['status'] in ('failed', 'uncertain') or result['confidence'] < self.config.confidence_threshold:
            recommendation = 'review'
        elif task_complete:
            recommendation = 'task_completed'
        elif phase_confirmed:
            recommendation = 'phase_completed' if candidate in self.completed else 'review'
        self.previous = result
        self.history.append(obs)
        self.last_timestamp, self.last_frame_id = obs['timestamp_s'], obs['frame_id']
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {'schema_version': '0.1', 'episode_id': obs['episode_id'], 'frame_id': obs['frame_id'],
                'timestamp_s': obs['timestamp_s'], **result,
                'phase_completion_confirmed': phase_confirmed,
                'task_completed': task_complete, 'recommendation': recommendation,
                'backend': self.config.backend, 'backend_error_type': error,
                'metrics': {'agent_latency_ms': elapsed_ms, 'inference_and_parse_ms': inference_ms,
                            'over_budget': elapsed_ms > 1000 / self.config.target_hz}}
