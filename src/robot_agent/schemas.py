import json
import math


ERROR_CODES = frozenset({
    'missed_grasp', 'misaligned', 'short_of_target', 'off_target_press', 'stalled',
    'object_dropped', 'object_tipped', 'fixture_displaced', 'wrong_place', 'unexpected_state',
    'unsafe_motion', 'object_lost', 'hardware_fault', 'unrecoverable',
})


def observation(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError('observation must be an object')
    for key in ('episode_id', 'frame_id'):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f'{key} must be a nonempty string')
    ts = value.get('timestamp_s')
    if type(ts) not in (int, float) or not math.isfinite(ts) or ts < 0:
        raise ValueError('timestamp_s must be finite and nonnegative')
    if not isinstance(value.get('note', ''), str):
        raise ValueError('note must be text')
    if 'expected_phase' in value and (not isinstance(value['expected_phase'], str)
                                      or not value['expected_phase']):
        raise ValueError('expected_phase must be nonempty text')
    if 'end_of_episode' in value and type(value['end_of_episode']) is not bool:
        raise ValueError('end_of_episode must be boolean')
    if 'is_terminal' in value and type(value['is_terminal']) is not bool:
        raise ValueError('is_terminal must be boolean')
    if ('is_terminal' in value and 'end_of_episode' in value
            and value['is_terminal'] != value['end_of_episode']):
        raise ValueError('terminal flags must agree')
    if 'is_terminal' in value:
        value = {**value, 'end_of_episode': value['is_terminal']}
    images = value.get('images', [])
    if not isinstance(images, list) or len(images) > 4:
        raise ValueError('images must be a list of at most four image data URLs')
    if any(not isinstance(x, str) or not x.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,')) for x in images):
        raise ValueError('images must be PNG/JPEG base64 data URLs')
    return {k: value[k] for k in ('episode_id', 'frame_id', 'timestamp_s', 'expected_phase',
                                   'end_of_episode', 'note', 'images') if k in value}


def verdict(raw: str, phases: tuple[str, ...]) -> dict:
    raw = raw.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = raw.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    d = json.loads(raw)
    required = {'phase', 'status', 'confidence', 'errors', 'evidence'}
    if not isinstance(d, dict) or set(d) != required:
        raise ValueError('verdict must contain exactly phase/status/confidence/errors/evidence')
    if d['phase'] not in (*phases, 'unknown'):
        raise ValueError('invalid phase')
    if d['status'] not in ('in_progress', 'completed', 'failed', 'uncertain'):
        raise ValueError('invalid status')
    c = d['confidence']
    if type(c) not in (int, float) or not math.isfinite(c) or not 0 <= c <= 1:
        raise ValueError('invalid confidence')
    for key in ('errors', 'evidence'):
        if not isinstance(d[key], list) or not all(isinstance(x, str) for x in d[key]):
            raise ValueError(f'{key} must be a list of strings')
    if any(code not in ERROR_CODES for code in d['errors']):
        raise ValueError('invalid error code')
    if (d['status'] == 'failed') != bool(d['errors']):
        raise ValueError('failed status requires errors and other statuses forbid them')
    if d['status'] == 'completed' and (d['phase'] == 'unknown' or d['errors'] or not d['evidence']):
        raise ValueError('completion requires a known phase, evidence and no errors')
    if d['status'] == 'failed' and not d['evidence']:
        raise ValueError('failure requires evidence')
    return d
