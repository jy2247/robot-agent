from importlib.resources import files
import json


def messages(config, current):
    """Build a stable prefix followed by one camera-major temporal window."""
    phase = current['expected_phase']
    if phase not in config.phases:
        raise ValueError('expected_phase must be configured')
    index = config.phases.index(phase)
    task = {
        'task': config.task_id,
        'active_phase': phase,
        'success': config.criteria[index],
        'failure': config.failure_criteria[index],
    }
    system = files('robot_agent').joinpath('prompts/monitor.txt').read_text()
    system += '\nACTIVE RULES:\n' + json.dumps(task, separators=(',', ':'))
    ordering = (
        f'{config.camera_count} cameras, camera-major order; '
        f'{config.frames_per_camera} oldest-to-newest frames per camera; '
        f'{config.frame_interval_s:g}s between frames.')
    content = [{'type': 'text', 'text': ordering}]
    content.extend({'type': 'image_url', 'image_url': {'url': url}}
                   for url in current['images'])
    content.append({'type': 'text', 'text': (
        'Output one JSON object, for example '
        '{"status":"in_progress","confidence":0.5}, without explanation.')})
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': content}]
