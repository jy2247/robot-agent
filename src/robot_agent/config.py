from dataclasses import dataclass, field
from pathlib import Path
import math
import os
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 model-serving environments
    import tomli as tomllib


@dataclass(frozen=True)
class Config:
    backend: str
    assessment_hz: float
    terminal_confirmations: int
    confidence_threshold: float
    base_url: str
    model: str
    timeout_s: float
    max_tokens: int
    temperature: float
    top_p: float
    task_id: str
    phases: tuple[str, ...]
    criteria: tuple[str, ...]
    failure_criteria: tuple[str, ...]
    frames_per_camera: int
    camera_count: int
    frame_interval_s: float
    image_width: int
    image_height: int
    chat_template_kwargs: dict = field(default_factory=dict)
    votes: int = 1

    @property
    def image_count(self) -> int:
        return self.frames_per_camera * self.camera_count


def load_config(path: str) -> Config:
    raw = tomllib.loads(Path(path).read_text())
    agent, model, task, sampling = (
        raw['agent'], raw['model'], raw['task'], raw['sampling'])
    config = Config(
        backend=agent['backend'],
        votes=agent.get('votes', 1),
        assessment_hz=agent['assessment_hz'],
        terminal_confirmations=agent['terminal_confirmations'],
        confidence_threshold=agent['confidence_threshold'],
        base_url=os.getenv('ROBOT_VLM_BASE_URL', model['base_url']),
        model=os.getenv('ROBOT_VLM_MODEL', model['name']),
        timeout_s=model['timeout_s'],
        max_tokens=model['max_tokens'],
        temperature=model['temperature'],
        top_p=model['top_p'],
        chat_template_kwargs=model.get('chat_template_kwargs', {}),
        task_id=task['id'],
        phases=tuple(task['phases']),
        criteria=tuple(task['criteria']),
        failure_criteria=tuple(task['failure_criteria']),
        frames_per_camera=sampling['frames_per_camera'],
        camera_count=sampling['camera_count'],
        frame_interval_s=sampling['frame_interval_s'],
        image_width=sampling['image_width'],
        image_height=sampling['image_height'],
    )
    if config.backend not in {'mock', 'openai'}:
        raise ValueError('backend must be mock or openai')
    if type(config.votes) is not int or config.votes not in (1, 5):
        raise ValueError('votes must be 1 or 5')
    for value in (config.terminal_confirmations, config.max_tokens,
                  config.frames_per_camera, config.camera_count,
                  config.image_width, config.image_height):
        if type(value) is not int or value < 1:
            raise ValueError('counts and image dimensions must be positive integers')
    for value in (config.assessment_hz, config.timeout_s, config.frame_interval_s):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('rates and timeouts must be finite and positive')
    if (type(config.temperature) not in (int, float) or config.temperature < 0
            or type(config.top_p) not in (int, float) or not 0 < config.top_p <= 1):
        raise ValueError('invalid sampling parameters')
    if not 0 <= config.confidence_threshold <= 1:
        raise ValueError('confidence_threshold must be between zero and one')
    if (not config.phases or len(set(config.phases)) != len(config.phases)
            or not (len(config.phases) == len(config.criteria)
                    == len(config.failure_criteria))):
        raise ValueError('each unique phase needs success and failure criteria')
    if not all(isinstance(value, str) and value for value in (
            *config.phases, *config.criteria, *config.failure_criteria)):
        raise ValueError('phase definitions must be nonempty strings')
    if not isinstance(config.chat_template_kwargs, dict):
        raise ValueError('chat_template_kwargs must be a table')
    if config.max_tokens > 64:
        raise ValueError('fast monitor max_tokens must not exceed 64')
    if config.image_count > 24:
        raise ValueError('at most 24 images are supported per assessment')
    return config
