from importlib.resources import files
import json


def messages(config, history, current, previous):
    # Stable system/task prefix; dynamic frames follow. KV storage belongs to the server.
    task = {'task_id': config.task_id, 'instruction': config.instruction,
            'phases': dict(zip(config.phases, config.criteria))}
    system = files('robot_agent').joinpath('prompts/observer.txt').read_text()
    out = [{'role': 'system', 'content': system + '\nTASK:\n' + json.dumps(task)}]
    content = []
    for obs in [*history, current]:
        metadata = {k: v for k, v in obs.items() if k != 'images'}
        content.append({'type': 'text', 'text': json.dumps(metadata)})
        content.extend({'type': 'image_url', 'image_url': {'url': url}} for url in obs.get('images', []))
    content.append({'type': 'text', 'text': 'Previous judgment (may be wrong): ' + json.dumps(previous)})
    out.append({'role': 'user', 'content': content})
    return out
