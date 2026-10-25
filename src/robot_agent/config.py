from dataclasses import dataclass, field
from pathlib import Path
import math
import os
import tomllib


@dataclass(frozen=True)
class Config:
    backend: str
    history_size: int
    target_hz: float
    completion_confirmations: int
    confidence_threshold: float
    base_url: str
    model: str
    timeout_s: float
    max_tokens: int
    task_id: str
    instruction: str
    phases: tuple[str, ...]
    criteria: tuple[str, ...]
    chat_template_kwargs: dict = field(default_factory=dict)
    model_extra_body: dict = field(default_factory=dict)
    failure_criteria: tuple[str, ...] = ()
    max_retries: int = 2
    uncertainty_patience: int = 2


def load_config(path: str) -> Config:
    d = tomllib.loads(Path(path).read_text())
    a, m, t = d['agent'], d['model'], d['task']
    c = Config(**a, base_url=os.getenv('ROBOT_VLM_BASE_URL', m['base_url']),
               model=os.getenv('ROBOT_VLM_MODEL', m['name']),
               timeout_s=m['timeout_s'], max_tokens=m['max_tokens'],
               task_id=t['id'], instruction=t['instruction'],
               phases=tuple(t['phases']), criteria=tuple(t['criteria']),
               failure_criteria=tuple(t.get('failure_criteria', ())),
               chat_template_kwargs=m.get('chat_template_kwargs', {}),
               model_extra_body=m.get('extra_body', {}))
    if not isinstance(c.chat_template_kwargs, dict) or not isinstance(c.model_extra_body, dict):
        raise ValueError('model request options must be tables')
    if c.backend not in {'mock', 'openai'}:
        raise ValueError('backend must be mock or openai')
    for v in (c.history_size, c.completion_confirmations, c.max_tokens):
        if type(v) is not int or v < 1:
            raise ValueError('counts must be positive integers')
    for value in (c.max_retries, c.uncertainty_patience):
        if type(value) is not int or value < 0:
            raise ValueError('retry and uncertainty limits must be nonnegative integers')
    for v in (c.target_hz, c.timeout_s):
        if not math.isfinite(v) or v <= 0:
            raise ValueError('rates and timeouts must be finite and positive')
    if not 0 <= c.confidence_threshold <= 1:
        raise ValueError('confidence_threshold must be between zero and one')
    if (not c.phases or len(set(c.phases)) != len(c.phases)
            or len(c.phases) != len(c.criteria)
            or not all(isinstance(p, str) and p for p in (*c.phases, *c.criteria))):
        raise ValueError('each unique phase requires a nonempty completion criterion')
    if c.failure_criteria and (len(c.failure_criteria) != len(c.phases)
                               or not all(isinstance(x, str) and x for x in c.failure_criteria)):
        raise ValueError('failure_criteria must be empty or match phases')
    return c
