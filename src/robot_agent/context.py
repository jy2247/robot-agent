from importlib.resources import files
import json


def messages(config, history, current, previous, completed=(), reference=None):
    # Stable system/task prefix; dynamic frames follow. KV storage belongs to the server.
    task = {'task_id': config.task_id, 'instruction': config.instruction,
            'phases': {phase: {
                'success': config.criteria[index],
                'failure': config.failure_criteria[index] if config.failure_criteria else '',
            } for index, phase in enumerate(config.phases)}}
    system = files('robot_agent').joinpath('prompts/observer.txt').read_text()
    out = [{'role': 'system', 'content': system + '\nTASK:\n' + json.dumps(task)}]
    content = []
    observations = [*history, current]
    if reference and reference['frame_id'] not in {obs['frame_id'] for obs in observations}:
        metadata = {k: v for k, v in reference.items() if k != 'images'}
        metadata['temporal_role'] = 'initial_reference'
        content.append({'type': 'text', 'text': json.dumps(metadata)})
        content.extend({'type': 'image_url', 'image_url': {'url': url}}
                       for url in reference['images'])
    for position, obs in enumerate(observations):
        metadata = {k: v for k, v in obs.items() if k != 'images'}
        metadata['temporal_role'] = 'current' if position == len(observations) - 1 else 'history'
        content.append({'type': 'text', 'text': json.dumps(metadata)})
        content.extend({'type': 'image_url', 'image_url': {'url': url}} for url in obs.get('images', []))
    content.append({'type': 'text', 'text': 'Controller memory (not new visual evidence): '
                    + json.dumps({'completed_phases': [p for p in config.phases if p in completed]})})
    out.append({'role': 'user', 'content': content})
    return out
