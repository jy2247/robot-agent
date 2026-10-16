from typing import Protocol


class Backend(Protocol):
    def generate(self, messages: list[dict]) -> str:
        """Return raw verdict JSON. Raise on transport/model failure."""
        ...
