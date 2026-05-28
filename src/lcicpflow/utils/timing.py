from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


class StageTimer:
    def __init__(self) -> None:
        self.times: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = perf_counter()
        try:
            yield
        finally:
            self.times[name] = self.times.get(name, 0.0) + perf_counter() - start
