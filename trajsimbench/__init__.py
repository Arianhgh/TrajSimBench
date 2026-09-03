"""CPU-first foundations for the TrajSimBench trajectory benchmark."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("trajsimbench")
except PackageNotFoundError:  # pragma: no cover - useful from an unpacked checkout
    __version__ = "0.1.0"

from trajsimbench.data.dataset import TrajectoryDataset, TrajectoryView

__all__ = ["TrajectoryDataset", "TrajectoryView", "__version__"]
