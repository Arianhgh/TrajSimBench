"""Raw-input loaders with explicit acquisition and license gates."""

from trajsimbench.data.loaders.base import BaseLoader, LoaderInspection, PreparationResult
from trajsimbench.data.loaders.geolife import GeoLifeLoader
from trajsimbench.data.loaders.germany import DatasetGateError, GermanyLoader
from trajsimbench.data.loaders.porto import PortoLoader
from trajsimbench.data.loaders.synthetic import generate_synthetic_trajectories, prepare_synthetic

__all__ = [
    "BaseLoader",
    "DatasetGateError",
    "GeoLifeLoader",
    "GermanyLoader",
    "LoaderInspection",
    "PortoLoader",
    "PreparationResult",
    "generate_synthetic_trajectories",
    "prepare_synthetic",
]
