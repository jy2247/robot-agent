from importlib.resources import files
import json

from .schemas import observation


PIPETTE_SUPPORTS = frozenset({'hand', 'holder', 'work_surface', 'floor', 'unknown'})
TUBE_SUPPORTS = frozenset({'hand', 'rack', 'work_surface', 'floor', 'unknown'})
UNSAFE_SUPPORTS = frozenset({'work_surface', 'floor'})


def object_state_verdict(raw: str) -> dict:
    """Parse the first valid exact two-field global object-state JSON object."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != '{':
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or set(value) != {
                'pipette_support', 'tube_support'}:
            continue
        if value['pipette_support'] not in PIPETTE_SUPPORTS:
            raise ValueError('invalid pipette support')
        if value['tube_support'] not in TUBE_SUPPORTS:
            raise ValueError('invalid tube support')
        return value
    raise ValueError('model output contains no valid global object state')


def state_messages(config, current):
    system = files('robot_agent').joinpath('prompts/global_state.txt').read_text()
    ordering = (
        f'{config.camera_count} cameras, camera-major order; '
        f'{config.frames_per_camera} oldest-to-newest frames per camera; '
        f'{config.frame_interval_s:g}s between frames.')
    content = [{'type': 'text', 'text': ordering}]
    content.extend({'type': 'image_url', 'image_url': {'url': url}}
                   for url in current['images'])
    content.append({'type': 'text', 'text': (
        'Report both object supports at the newest time as the exact two-field JSON object.')})
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': content}]


def consistency_verdict(state: dict, phase_index: int) -> dict:
    """Convert observed supports into a conservative cross-phase safety verdict."""
    violations = []
    if phase_index >= 1 and state['pipette_support'] in UNSAFE_SUPPORTS:
        violations.append('PIPETTE_ON_UNINTENDED_SURFACE')
    if phase_index >= 1 and state['tube_support'] in UNSAFE_SUPPORTS:
        violations.append('TUBE_ON_UNINTENDED_SURFACE')
    if violations:
        return {'status': 'fail', 'reason': '+'.join(violations)}
    if 'unknown' in state.values():
        return {'status': 'unknown', 'reason': 'OBJECT_SUPPORT_UNKNOWN'}
    return {'status': 'safe', 'reason': None}


class GlobalSafetyCritic:
    """Companion critic that observes object state and confirms cross-phase violations."""

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.reset()

    def reset(self):
        self.episode_id = None
        self.last_timestamp = -1.0
        self.pending_failure = None
        self.pending_count = 0

    def observe(self, payload):
        current = observation(payload, self.config.image_count)
        if (self.episode_id is not None
                and current['episode_id'] != self.episode_id):
            raise ValueError('episode mismatch: reset before a new episode')
        if current['timestamp_s'] <= self.last_timestamp:
            raise ValueError('frames must have strictly increasing timestamps')
        phase = current['expected_phase']
        if phase not in self.config.phases:
            raise ValueError('expected_phase must be configured')

        error = None
        try:
            state = object_state_verdict(
                self.backend.generate(state_messages(self.config, current)))
            verdict = consistency_verdict(state, self.config.phases.index(phase))
        except Exception as exc:
            error = type(exc).__name__
            state = {'pipette_support': 'unknown', 'tube_support': 'unknown'}
            verdict = {'status': 'unknown', 'reason': 'CRITIC_ERROR'}

        failure_key = verdict['reason'] if error is None and verdict['status'] == 'fail' else None
        if failure_key is not None and failure_key == self.pending_failure:
            self.pending_count += 1
        elif failure_key is not None:
            self.pending_failure, self.pending_count = failure_key, 1
        else:
            self.pending_failure, self.pending_count = None, 0
        accepted = bool(failure_key and
                        self.pending_count >= self.config.terminal_confirmations)

        self.episode_id = current['episode_id']
        self.last_timestamp = current['timestamp_s']
        return {
            **state,
            'status': verdict['status'],
            'reason': verdict['reason'],
            'failure_confirmation_count': self.pending_count,
            'failure_accepted': accepted,
            'backend_error_type': error,
        }
