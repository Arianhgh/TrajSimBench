"""Deterministic seed helpers for CPU preparation and experiments."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


@contextmanager
def seeded(seed: int) -> Iterator[np.random.Generator]:
    state = np.random.get_state()
    python_state = random.getstate()
    try:
        yield seed_everything(seed)
    finally:
        np.random.set_state(state)
        random.setstate(python_state)
