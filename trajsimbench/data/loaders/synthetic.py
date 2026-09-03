"""Deterministic synthetic trajectories used by tests and CPU smoke runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trajsimbench.data.dataset import TrajectoryDataset, write_canonical_dataset
from trajsimbench.data.schema import TrajectoryInput
from trajsimbench.data.splitting import make_split_bundle


def _line(
    start: tuple[float, float],
    end: tuple[float, float],
    count: int,
    times: np.ndarray | None = None,
) -> np.ndarray:
    coordinates = np.linspace(start, end, count, dtype=np.float64)
    if times is None:
        return coordinates
    return np.column_stack((coordinates, times))


def _record(
    identifier: str,
    coordinates: np.ndarray,
    *,
    user: str = "synthetic-user",
    mode: str | None = None,
) -> TrajectoryInput:
    return TrajectoryInput(
        identifier, coordinates, source_id=identifier, user_id=user, mobility_mode=mode
    )


def generate_synthetic_trajectories(seed: int = 0) -> list[TrajectoryInput]:
    """Create named geometry/time cases covering the foundation edge cases."""

    rng = np.random.default_rng(seed)
    base = _line((13.4000, 52.5000), (13.4100, 52.5000), 8)
    translated = base + np.array([0.0010, 0.0005])
    reversed_path = base[::-1].copy()
    variable = base[[0, 1, 3, 6, 7]].copy()
    small_detour = base.copy()
    small_detour[3:5, 1] += 0.0007
    large_detour = base.copy()
    large_detour[2:6, 1] += 0.003
    crossing = _line((13.4000, 52.4950), (13.4100, 52.5050), 8)
    route_a = np.array([[13.400, 52.500], [13.405, 52.500], [13.410, 52.500]], dtype=float)
    route_b = np.array([[13.400, 52.500], [13.405, 52.503], [13.410, 52.500]], dtype=float)
    temporal = np.column_stack((base, np.arange(len(base), dtype=float) ** 1.4))
    repeated = base.copy()
    repeated = np.insert(repeated, 4, repeated[4], axis=0)
    times = np.arange(len(repeated), dtype=float)
    repeated_timed = np.column_stack((repeated, times))
    jitter = base + rng.normal(0, 0.00002, size=base.shape)
    short = np.array([[13.400, 52.500]], dtype=float)
    return [
        _record("synthetic:straight", base, user="u0", mode="walk"),
        _record("synthetic:translated", translated, user="u1", mode="walk"),
        _record("synthetic:reversed", reversed_path, user="u2", mode="walk"),
        _record("synthetic:variable_sampling", variable, user="u3", mode="walk"),
        _record("synthetic:small_detour", small_detour, user="u4", mode="car"),
        _record("synthetic:large_detour", large_detour, user="u5", mode="car"),
        _record("synthetic:crossing", crossing, user="u6", mode="bike"),
        _record("synthetic:route_a", route_a, user="u7", mode="car"),
        _record("synthetic:route_b", route_b, user="u8", mode="car"),
        _record("synthetic:temporal_warp", temporal, user="u9", mode="walk"),
        _record("synthetic:repeated_points", repeated_timed, user="u10", mode="walk"),
        _record("synthetic:jittered", jitter, user="u11", mode="walk"),
        _record("synthetic:very_short", short, user="u12", mode=None),
    ]


def prepare_synthetic(
    output_path: str | Path,
    *,
    version: str = "v1",
    seed: int = 0,
    projected_crs: str = "EPSG:32633",
    min_points: int = 1,
) -> Path:
    records = generate_synthetic_trajectories(seed)
    ids = [record.trajectory_id for record in records]
    splits = make_split_bundle(ids, seed=seed, records=records, include_user_held_out=True)
    # Synthetic records intentionally include one point without a timestamp;
    # this exercises the documented NaN-for-genuinely-unavailable policy.
    path = write_canonical_dataset(
        output_path,
        records,
        dataset="synthetic",
        version=version,
        projected_crs=projected_crs,
        source_name="TrajSimBench generated fixture",
        source_license="MIT (generated fixture)",
        redistribution_policy="redistributable",
        preprocessing_config_hash=f"synthetic-seed-{seed}",
        code_version="trajsimbench-0.1.0",
        point_features={"columns": []},
        splits=splits,
        created_at_utc=None,
        min_points=min_points,
    )
    # Verify the public opening path here so fixture generation is end-to-end.
    TrajectoryDataset.open(path, mmap=True).validate().raise_if_invalid()
    return path
