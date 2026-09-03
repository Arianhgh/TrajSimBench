"""Partial route-overlap negatives with comparable endpoints."""

from .base import NegativeGenerator, endpoints, route_overlap


class PartialOverlapNegativeGenerator(NegativeGenerator):
    name = "partial_overlap"
    version = "1.0"

    def __init__(
        self,
        *,
        min_overlap: float = 0.20,
        max_overlap: float = 0.80,
        endpoint_radius_m: float = 500.0,
        overlap_tolerance_m: float = 25.0,
        **defaults,
    ) -> None:
        super().__init__(
            min_overlap=min_overlap,
            max_overlap=max_overlap,
            endpoint_radius_m=endpoint_radius_m,
            overlap_tolerance_m=overlap_tolerance_m,
            **defaults,
        )

    def _qualify(self, query, candidate, config):
        qo, qd = endpoints(query)
        co, cd = endpoints(candidate)
        endpoint_gap = max(
            float(((qo - co) ** 2).sum() ** 0.5), float(((qd - cd) ** 2).sum() ** 0.5)
        )
        overlap = route_overlap(query, candidate, tolerance_m=float(config["overlap_tolerance_m"]))
        achieved = {
            "endpoint_gap_m": endpoint_gap,
            "route_overlap": overlap,
            "route_overlap_definition": "resampled_path_coverage_v1",
        }
        if endpoint_gap > float(config["endpoint_radius_m"]):
            return False, achieved, "endpoint_gap_too_large"
        if overlap < float(config["min_overlap"]):
            return False, achieved, "overlap_too_small"
        if overlap > float(config["max_overlap"]):
            return False, achieved, "overlap_too_large"
        return True, achieved, None


PartialRouteOverlapNegativeGenerator = PartialOverlapNegativeGenerator
