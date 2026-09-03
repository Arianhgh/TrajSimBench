"""Minimal nanosecond timing context for later systems stages."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Timer:
    elapsed_ns: int = 0
    _started_ns: int | None = None

    def __enter__(self) -> Timer:
        self._started_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._started_ns is not None:
            self.elapsed_ns = time.perf_counter_ns() - self._started_ns


def timed_call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, int]:
    with Timer() as timer:
        result = function(*args, **kwargs)
    return result, timer.elapsed_ns
