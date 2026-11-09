import json
import math


WIRE_STATUSES = frozenset({'in_progress', 'success', 'fail'})


def observation(value: dict, expected_images: int | None = None) -> dict:
    if not isinstance(value, dict):
        raise ValueError('observation must be an object')
    for key in ('episode_id', 'frame_id', 'expected_phase'):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f'{key} must be nonempty text')
    timestamp = value.get('timestamp_s')
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError('timestamp_s must be finite and nonnegative')
    if 'end_of_episode' in value and type(value['end_of_episode']) is not bool:
        raise ValueError('end_of_episode must be boolean')
    if 'is_terminal' in value and type(value['is_terminal']) is not bool:
        raise ValueError('is_terminal must be boolean')
    if ('is_terminal' in value and 'end_of_episode' in value
            and value['is_terminal'] != value['end_of_episode']):
        raise ValueError('terminal flags must agree')
    terminal = value.get('is_terminal', value.get('end_of_episode', False))
    images = value.get('images', [])
    if not isinstance(images, list) or (expected_images is not None and len(images) != expected_images):
        raise ValueError(f'images must contain exactly {expected_images} frames')
    if any(not isinstance(item, str) or not item.startswith(
            ('data:image/jpeg;base64,', 'data:image/png;base64,')) for item in images):
        raise ValueError('images must be PNG/JPEG data URLs')
    return {
        'episode_id': value['episode_id'],
        'frame_id': value['frame_id'],
        'timestamp_s': timestamp,
        'expected_phase': value['expected_phase'],
        'end_of_episode': terminal,
        'images': images,
        **({'note': value['note']} if isinstance(value.get('note'), str) else {}),
    }


def monitor_verdict(raw: str) -> dict:
    """Parse the first valid two-field monitor JSON object from model output."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != '{':
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or set(value) != {'status', 'confidence'}:
            continue
        status, confidence = value['status'], value['confidence']
        if status not in WIRE_STATUSES:
            raise ValueError('invalid monitor status')
        if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1):
            raise ValueError('invalid confidence')
        return {'status': status, 'confidence': float(confidence)}
    raise ValueError('model output contains no valid monitor verdict')
