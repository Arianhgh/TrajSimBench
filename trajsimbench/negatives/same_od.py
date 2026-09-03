"""Same-origin/destination, low-route-overlap negatives."""

from .base import NegativeGenerator, endpoints, route_overlap, trajectory_length


class SameODNegativeGenerator(NegativeGenerator):
    name = "same_od"
    version = "1.0"

    def __init__(
        self,
        *,
        origin_radius_m: float = 100.0,
        destination_radius_m: float = 100.0,
        max_route_overlap: float = 0.50,
        length_ratio_band: tuple[float, float] = (0.25, 4.0),
        overlap_tolerance_m: float = 25.0,
        **defaults,
    ) -> None:
        super().__init__(
            origin_radius_m=origin_radius_m,
            destination_radius_m=destination_radius_m,
            max_route_overlap=max_route_overlap,
            length_ratio_band=length_ratio_band,
            overlap_tolerance_m=overlap_tolerance_m,
            **defaults,
        )

    def _qualify(self, query, candidate, config):
        qo, qd = endpoints(query)
        co, cd = endpoints(candidate)
        origin = float(((qo - co) ** 2).sum() ** 0.5)
        destination = float(((qd - cd) ** 2).sum() ** 0.5)
        overlap = route_overlap(query, candidate, tolerance_m=float(config["overlap_tolerance_m"]))
        q_length = trajectory_length(query)
        c_length = trajectory_length(candidate)
        ratio = c_length / q_length if q_length > 0 else float("inf")
        lo, hi = map(float, config["length_ratio_band"])
        achieved = {
            "origin_distance_m": origin,
            "destination_distance_m": destination,
            "route_overlap": overlap,
            "length_ratio": ratio,
            "route_overlap_definition": "resampled_path_coverage_v1",
        }
        if origin > float(config["origin_radius_m"]):
            return False, achieved, "origin_radius_not_met"
        if destination > float(config["destination_radius_m"]):
            return False, achieved, "destination_radius_not_met"
        if overlap > float(config["max_route_overlap"]):
            return False, achieved, "route_overlap_too_high"
        if not lo <= ratio <= hi:
            return False, achieved, "length_ratio_outside_band"
        return True, achieved, None


SameOriginDestinationNegativeGenerator = SameODNegativeGenerator
