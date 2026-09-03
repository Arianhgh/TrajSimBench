"""Random and structured hard-negative generators."""

from .base import (
    NegativeCandidate,
    NegativeGenerationReport,
    NegativeGenerationResult,
    NegativeGenerator,
    route_overlap,
    shape_distance,
)
from .nearby_shape import (
    NearbyShapeNegativeGenerator,
    SpatiallyNearbyDifferentShapeNegativeGenerator,
)
from .partial_overlap import PartialOverlapNegativeGenerator, PartialRouteOverlapNegativeGenerator
from .random import RandomNegativeGenerator
from .registry import NEGATIVE_REGISTRY, NegativeGeneratorRegistry, get_negative_generator
from .reversed import ReversedNegativeGenerator, ReversedTrajectoryNegativeGenerator
from .same_od import SameODNegativeGenerator, SameOriginDestinationNegativeGenerator
from .temporal import SameRouteAlteredTemporalNegativeGenerator, SameRouteTemporalNegativeGenerator
from .translated import TranslatedShapeCopyNegativeGenerator, TranslatedShapeNegativeGenerator

__all__ = [
    "NegativeCandidate",
    "NegativeGenerationReport",
    "NegativeGenerationResult",
    "NegativeGenerator",
    "RandomNegativeGenerator",
    "SameODNegativeGenerator",
    "NearbyShapeNegativeGenerator",
    "SpatiallyNearbyDifferentShapeNegativeGenerator",
    "TranslatedShapeNegativeGenerator",
    "TranslatedShapeCopyNegativeGenerator",
    "ReversedNegativeGenerator",
    "ReversedTrajectoryNegativeGenerator",
    "PartialOverlapNegativeGenerator",
    "PartialRouteOverlapNegativeGenerator",
    "SameRouteTemporalNegativeGenerator",
    "SameRouteAlteredTemporalNegativeGenerator",
    "SameOriginDestinationNegativeGenerator",
    "NegativeGeneratorRegistry",
    "NEGATIVE_REGISTRY",
    "get_negative_generator",
    "shape_distance",
    "route_overlap",
]
