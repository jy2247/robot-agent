import json
import math


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
    images = value.get('images', [])
    if not isinstance(images, list) or len(images) > 4:
        raise ValueError('images must be a list of at most four image data URLs')
    if any(not isinstance(x, str) or not x.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,')) for x in images):
        raise ValueError('images must be PNG/JPEG base64 data URLs')
    return {k: value[k] for k in ('episode_id', 'frame_id', 'timestamp_s', 'note', 'images') if k in value}


def verdict(raw: str, phases: tuple[str, ...]) -> dict:
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
    if d['status'] == 'completed' and (d['phase'] == 'unknown' or d['errors'] or not d['evidence']):
        raise ValueError('completion requires a known phase, evidence and no errors')
    return d
