"""Geographically nearby but geometrically different negatives."""

from .base import NegativeGenerator, endpoints, shape_distance


class NearbyShapeNegativeGenerator(NegativeGenerator):
    name = "nearby_shape"
    version = "1.0"

    def __init__(
        self,
        *,
        geographic_nearness_radius_m: float = 1000.0,
        min_shape_distance: float = 0.15,
        max_shape_distance: float = 2.0,
        **defaults,
    ) -> None:
        super().__init__(
            geographic_nearness_radius_m=geographic_nearness_radius_m,
            min_shape_distance=min_shape_distance,
            max_shape_distance=max_shape_distance,
            **defaults,
        )

    def _qualify(self, query, candidate, config):
        qo, qd = endpoints(query)
        co, cd = endpoints(candidate)
        near = min(float(((qo - co) ** 2).sum() ** 0.5), float(((qd - cd) ** 2).sum() ** 0.5))
        shape = shape_distance(query, candidate)
        achieved = {
            "endpoint_nearness_m": near,
            "shape_distance": shape,
            "shape_distance_definition": "translation_scale_normalized_resampled_l2_v1",
        }
        if near > float(config["geographic_nearness_radius_m"]):
            return False, achieved, "geographic_nearness_not_met"
        if shape < float(config["min_shape_distance"]):
            return False, achieved, "shape_not_different_enough"
        if shape > float(config["max_shape_distance"]):
            return False, achieved, "shape_distance_above_band"
        return True, achieved, None


SpatiallyNearbyDifferentShapeNegativeGenerator = NearbyShapeNegativeGenerator
