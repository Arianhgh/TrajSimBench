"""Deterministic, provenance-carrying trajectory perturbations.

The package deliberately has no dependency on the benchmark data layer.  A
trajectory is any object exposing ``points`` and, optionally,
``trajectory_id``/``metadata``.  This keeps perturbations useful for both the
canonical dataset reader and small test fixtures.
"""

from .base import Perturbation, PerturbationError, UnsupportedPerturbationError
from .registry import PERTURBATION_REGISTRY, PerturbationRegistry, get_perturbation
from .result import PerturbationProvenance, PerturbationResult
from .route import FreeSpaceDetourPerturbation, RoadNetworkDetourPerturbation
from .sampling import (
    ContiguousOutagePerturbation,
    RandomPointLossPerturbation,
    SamplingFrequencyReductionPerturbation,
    TruncationPerturbation,
)
from .spatial import (
    CorrelatedGPSDriftPerturbation,
    IndependentGPSNoisePerturbation,
    SpatialQuantizationPerturbation,
    SpatialTranslationPerturbation,
)
from .temporal import (
    ReversalPerturbation,
    SpeedDistortionPerturbation,
    TemporalJitterPerturbation,
)

# Friendly class aliases keep the public API readable while the registry uses
# the stable configuration names from the plan.
GPSNoisePerturbation = IndependentGPSNoisePerturbation
GPSDriftPerturbation = CorrelatedGPSDriftPerturbation
PointLossPerturbation = RandomPointLossPerturbation
GPSOutagePerturbation = ContiguousOutagePerturbation
DownsamplingPerturbation = SamplingFrequencyReductionPerturbation
QuantizationPerturbation = SpatialQuantizationPerturbation
TimeJitterPerturbation = TemporalJitterPerturbation
TranslationPerturbation = SpatialTranslationPerturbation
DetourPerturbation = FreeSpaceDetourPerturbation

__all__ = [
    "Perturbation",
    "PerturbationError",
    "UnsupportedPerturbationError",
    "PerturbationProvenance",
    "PerturbationResult",
    "PerturbationRegistry",
    "PERTURBATION_REGISTRY",
    "get_perturbation",
    "IndependentGPSNoisePerturbation",
    "CorrelatedGPSDriftPerturbation",
    "RandomPointLossPerturbation",
    "ContiguousOutagePerturbation",
    "SamplingFrequencyReductionPerturbation",
    "SpatialQuantizationPerturbation",
    "TemporalJitterPerturbation",
    "SpeedDistortionPerturbation",
    "TruncationPerturbation",
    "ReversalPerturbation",
    "SpatialTranslationPerturbation",
    "FreeSpaceDetourPerturbation",
    "RoadNetworkDetourPerturbation",
    "GPSNoisePerturbation",
    "GPSDriftPerturbation",
    "PointLossPerturbation",
    "GPSOutagePerturbation",
    "DownsamplingPerturbation",
    "QuantizationPerturbation",
    "TimeJitterPerturbation",
    "TranslationPerturbation",
    "DetourPerturbation",
]
