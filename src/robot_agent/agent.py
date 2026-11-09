import time

from .context import messages
from .schemas import monitor_verdict, observation


class MonitorAgent:
    """Synchronous one-request monitor. One instance owns one episode."""

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.reset()

    def reset(self):
        self.episode_id = None
        self.last_timestamp = -1.0
        self.last_frame_id = None
        self.completed = set()
        self.failed = set()
        self.pending_terminal = None
        self.pending_count = 0
        self.terminated = False

    def observe(self, payload):
        started = time.perf_counter()
        obs = observation(payload, self.config.image_count)
        if self.terminated:
            raise ValueError('episode has ended: reset before observing again')
        if obs['expected_phase'] not in self.config.phases:
            raise ValueError('expected_phase must be configured')
        if self.episode_id is not None and obs['episode_id'] != self.episode_id:
            raise ValueError('episode mismatch: reset before a new episode')
        if obs['timestamp_s'] <= self.last_timestamp or obs['frame_id'] == self.last_frame_id:
            raise ValueError('frames must be distinct with strictly increasing timestamps')

        self.episode_id = obs['episode_id']
        inference_started = time.perf_counter()
        error = None
        try:
            result = monitor_verdict(self.backend.generate(messages(self.config, obs)))
        except Exception as exc:
            # A transport/parse failure is not physical evidence of task failure.
            error = type(exc).__name__
            result = {'status': 'in_progress', 'confidence': 0.0}
        inference_ms = (time.perf_counter() - inference_started) * 1000

        phase = obs['expected_phase']
        confident_terminal = (error is None and result['status'] in {'success', 'fail'}
                              and result['confidence'] >= self.config.confidence_threshold)
        key = (phase, result['status']) if confident_terminal else None
        if key is not None and key == self.pending_terminal:
            self.pending_count += 1
        elif key is not None:
            self.pending_terminal, self.pending_count = key, 1
        else:
            self.pending_terminal, self.pending_count = None, 0

        accepted = bool(key and self.pending_count >= self.config.terminal_confirmations)
        decision = 'continue'
        if accepted and result['status'] == 'success':
            self.completed.add(phase)
            self.failed.discard(phase)
            decision = 'advance'
        elif accepted:
            self.failed.add(phase)
            self.completed.discard(phase)
            decision = 'stop'

        task_complete = len(self.completed) == len(self.config.phases) and not self.failed
        if task_complete:
            decision = 'complete'
        overall_status, overall_reason = 'in_progress', None
        if obs['end_of_episode']:
            self.terminated = True
            if task_complete:
                overall_status, decision = 'success', 'complete'
            else:
                overall_status, decision = 'failed', 'stop'
                overall_reason = 'PHASE_FAILED' if self.failed else 'INCOMPLETE_AT_END'

        self.last_timestamp = obs['timestamp_s']
        self.last_frame_id = obs['frame_id']
        usage = getattr(self.backend, 'last_usage', {}) if error is None else {}
        prompt_details = usage.get('prompt_tokens_details') or {}
        prompt_tokens = usage.get('prompt_tokens')
        cached_tokens = prompt_details.get('cached_tokens')
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            'schema_version': '1.0',
            'episode_id': obs['episode_id'],
            'frame_id': obs['frame_id'],
            'timestamp_s': obs['timestamp_s'],
            'phase': phase,
            **result,
            'terminal_confirmation_count': self.pending_count,
            'terminal_accepted': accepted,
            'completed_phases': [p for p in self.config.phases if p in self.completed],
            'failed_phases': [p for p in self.config.phases if p in self.failed],
            'task_completed': task_complete,
            'decision': decision,
            'is_terminal': self.terminated,
            'overall_status': overall_status,
            'overall_reason': overall_reason,
            'backend': self.config.backend,
            'backend_error_type': error,
            'metrics': {
                'agent_latency_ms': latency_ms,
                'inference_and_parse_ms': inference_ms,
                'over_budget': latency_ms > 1000 / self.config.assessment_hz,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': usage.get('completion_tokens'),
                'cached_prompt_tokens': cached_tokens,
            },
        }
