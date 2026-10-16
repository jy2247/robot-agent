from .mock import MockBackend
from .openai import OpenAIBackend


def create_backend(config):
    return MockBackend() if config.backend == 'mock' else OpenAIBackend(config)
