from collections import deque
import time

from .context import messages
from .schemas import observation, verdict


RETRY_ERRORS = frozenset({'missed_grasp', 'misaligned', 'short_of_target',
                          'off_target_press', 'stalled'})
REPLAN_ERRORS = frozenset({'object_dropped', 'object_tipped', 'fixture_displaced',
                           'wrong_place', 'unexpected_state'})
FAIL_ERRORS = frozenset({'unsafe_motion', 'object_lost', 'hardware_fault', 'unrecoverable'})


class ObserverAgent:
    """One episode per instance; callers must serialize access."""

    def __init__(self, config, backend):
        self.config, self.backend = config, backend
        self.history = deque(maxlen=config.history_size)
        self.episode_id = None
        self.previous = None
        self.reference = None
        self.last_timestamp = -1
        self.last_frame_id = None
        self.completed = set()
        self.failed = set()
        self.terminated = False
        self.confirmations = {phase: 0 for phase in config.phases}
        self.confidence_phase = None
        self.low_confidence_streak = 0

    def reset(self):
        self.__init__(self.config, self.backend)

    def observe(self, payload):
        started = time.perf_counter()
        obs = observation(payload)
        if self.terminated:
            raise ValueError('episode has ended: reset before observing again')
        if obs.get('expected_phase') not in (None, *self.config.phases):
            raise ValueError('expected_phase must be a configured phase')
        if self.episode_id is not None and self.episode_id != obs['episode_id']:
            raise ValueError('episode mismatch: reset before starting a new episode')
        if obs['timestamp_s'] <= self.last_timestamp or obs['frame_id'] == self.last_frame_id:
            raise ValueError('frames must be distinct with strictly increasing timestamps')
        self.episode_id = obs['episode_id']
        if self.reference is None and obs.get('images'):
            # One fixed scene view retains initial support locations within the image budget.
            self.reference = {**obs, 'images': obs['images'][:1]}
        inference_start = time.perf_counter()
        error = None
        try:
            raw = self.backend.generate(messages(self.config, self.history, obs, self.previous,
                                                 self.completed, self.reference))
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
        if result['status'] == 'failed' and result['phase'] in self.confirmations:
            self.confirmations[result['phase']] = 0
        if candidate:
            self.confirmations[candidate] = min(self.config.completion_confirmations,
                                                 self.confirmations[candidate] + 1)
        end_confirmed = (obs.get('end_of_episode', False) and valid_completion
                         and candidate == self.config.phases[-1])
        phase_confirmed = bool(candidate and (
            self.confirmations[candidate] >= self.config.completion_confirmations or end_confirmed))
        # Stage transitions are not physical confirmation of an unseen outcome.
        transition_confirmed = False
        confirmed_phase = None
        if phase_confirmed:
            confirmed_phase = candidate
        if confirmed_phase:
            self.confirmations[confirmed_phase] = self.config.completion_confirmations
            self.completed.add(confirmed_phase)
            self.failed.discard(confirmed_phase)
        if (result['status'] == 'failed'
                and result['confidence'] >= self.config.confidence_threshold):
            self.failed.add(result['phase'])
            self.completed.discard(result['phase'])

        if result['phase'] != self.confidence_phase:
            self.confidence_phase = result['phase']
            self.low_confidence_streak = 0
        low_confidence = result['confidence'] < self.config.confidence_threshold
        self.low_confidence_streak = self.low_confidence_streak + 1 if low_confidence else 0
        confidence_grace = (result['status'] != 'failed' and low_confidence
                            and self.low_confidence_streak <= self.config.uncertainty_patience)
        task_complete = len(self.completed) == len(self.config.phases) and not self.failed
        recommendation, decision = 'continue', 'continue'
        errors = set(result['errors'])
        if result['status'] == 'failed':
            if low_confidence:
                recommendation, decision = 'review', 'review'
            elif errors & FAIL_ERRORS:
                recommendation, decision = 'stop', 'fail'
            elif errors & REPLAN_ERRORS:
                recommendation, decision = 'replan', 'replan'
            elif errors and errors <= RETRY_ERRORS:
                recommendation, decision = 'retry_phase', 'retry'
            else:
                recommendation, decision = 'review', 'review'
        elif (result['status'] == 'uncertain' or low_confidence) and not confidence_grace:
            recommendation = 'review'
            decision = 'review'
        elif task_complete:
            recommendation = 'task_completed'
            decision = 'success'
        elif phase_confirmed:
            recommendation = 'phase_completed' if candidate in self.completed else 'review'
            decision = 'success' if candidate in self.completed else 'review'
        overall_status, overall_reason = 'in_progress', None
        if obs.get('end_of_episode', False):
            self.terminated = True
            overall_status = 'success' if task_complete else 'failed'
            overall_reason = ('PHASE_FAILED' if self.failed else
                              None if task_complete else 'INCOMPLETE_AT_END')
            recommendation, decision = (('task_completed', 'success') if task_complete
                                        else ('stop', 'fail'))
        self.previous = result
        self.history.append(obs)
        self.last_timestamp, self.last_frame_id = obs['timestamp_s'], obs['frame_id']
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {'schema_version': '0.1', 'episode_id': obs['episode_id'], 'frame_id': obs['frame_id'],
                'timestamp_s': obs['timestamp_s'], **result,
                'phase_completion_confirmed': phase_confirmed,
                'end_of_episode_confirmation': end_confirmed,
                'transition_completion_confirmed': transition_confirmed,
                'confirmed_phase': confirmed_phase,
                'completed_phases': [p for p in self.config.phases if p in self.completed],
                'phase_confirmation_counts': dict(self.confirmations),
                'failed_phases': [p for p in (*self.config.phases, 'unknown') if p in self.failed],
                'is_terminal': self.terminated,
                'overall_status': overall_status, 'overall_reason': overall_reason,
                'low_confidence_streak': self.low_confidence_streak,
                'confidence_grace_active': confidence_grace,
                'task_completed': task_complete, 'recommendation': recommendation,
                'decision': decision,
                'backend': self.config.backend, 'backend_error_type': error,
                'metrics': {'agent_latency_ms': elapsed_ms, 'inference_and_parse_ms': inference_ms,
                            'over_budget': elapsed_ms > 1000 / self.config.target_hz}}
