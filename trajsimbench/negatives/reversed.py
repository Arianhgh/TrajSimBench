"""Candidates matching the query's reversed path shape."""

from types import SimpleNamespace

from .base import NegativeGenerator, resample_polyline, shape_distance


class ReversedNegativeGenerator(NegativeGenerator):
    name = "reversed"
    version = "1.0"

    def _qualify(self, query, candidate, config):
        sample_count = int(config.get("sample_count", 32))
        reversed_query = resample_polyline(query, sample_count)[::-1]
        candidate_path = resample_polyline(candidate, sample_count)
        # Compare lightweight projected views through the same normalized
        # metric used by the translated-shape generator.
        reverse_distance = shape_distance(
            SimpleNamespace(points=reversed_query, metadata={}),
            SimpleNamespace(points=candidate_path, metadata={}),
            sample_count=sample_count,
        )
        threshold = float(config.get("max_shape_distance", 0.15))
        achieved = {"reversed_shape_distance": reverse_distance, "threshold": threshold}
        if reverse_distance > threshold:
            return False, achieved, "reversed_shape_distance_too_large"
        return True, achieved, None


ReversedTrajectoryNegativeGenerator = ReversedNegativeGenerator
