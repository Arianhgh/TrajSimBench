"""Same-route negatives with deliberately altered temporal behavior."""

from .base import NegativeGenerator, shape_distance, temporal_behavior_distance


class SameRouteTemporalNegativeGenerator(NegativeGenerator):
    name = "same_route_temporal"
    version = "1.0"

    def __init__(
        self, *, max_shape_distance: float = 0.10, min_temporal_distance: float = 0.25, **defaults
    ) -> None:
        super().__init__(
            max_shape_distance=max_shape_distance,
            min_temporal_distance=min_temporal_distance,
            **defaults,
        )

    def _qualify(self, query, candidate, config):
        spatial = shape_distance(query, candidate)
        temporal = temporal_behavior_distance(query, candidate)
        achieved = {
            "shape_distance": spatial,
            "temporal_behavior_distance": temporal,
            "temporal_distance_definition": "duration_log_ratio_plus_normalized_interval_l1_v1",
        }
        if spatial > float(config["max_shape_distance"]):
            return False, achieved, "route_shape_too_different"
        if temporal != temporal or temporal < float(config["min_temporal_distance"]):
            return False, achieved, "temporal_behavior_not_different_enough"
        return True, achieved, None


SameRouteAlteredTemporalNegativeGenerator = SameRouteTemporalNegativeGenerator
