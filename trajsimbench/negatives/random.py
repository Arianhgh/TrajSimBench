"""Uniform random negatives with explicit self-match exclusion."""

from .base import NegativeGenerator


class RandomNegativeGenerator(NegativeGenerator):
    name = "random"
    version = "1.0"

    def _qualify(self, query, candidate, config):
        return True, {"policy": "candidate-universe sample without replacement"}, None
