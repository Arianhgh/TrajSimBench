"""Explicit WGS84 <-> projected-meter transformations."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pyproj import CRS, Transformer


@lru_cache(maxsize=32)
def _transformers(projected_crs: str) -> tuple[Transformer, Transformer]:
    crs = CRS.from_user_input(projected_crs)
    forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return forward, inverse


def project_coordinates(
    longitude: np.ndarray, latitude: np.ndarray, projected_crs: str
) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS84 arrays to meters using ``always_xy=True``."""

    lon = np.asarray(longitude, dtype=np.float64)
    lat = np.asarray(latitude, dtype=np.float64)
    if lon.shape != lat.shape:
        raise ValueError("longitude and latitude must have matching shapes")
    x, y = _transformers(projected_crs)[0].transform(lon, lat)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def inverse_project_coordinates(
    x: np.ndarray, y: np.ndarray, projected_crs: str
) -> tuple[np.ndarray, np.ndarray]:
    """Transform projected-meter arrays back to WGS84 longitude/latitude."""

    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have matching shapes")
    lon, lat = _transformers(projected_crs)[1].transform(x_values, y_values)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def choose_local_utm(longitude: np.ndarray, latitude: np.ndarray) -> str:
    """Choose a deterministic UTM CRS for a compact WGS84 extent."""

    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    if lon.size == 0 or lat.size == 0:
        raise ValueError("at least one coordinate is required to choose a UTM CRS")
    if np.nanmax(lon) - np.nanmin(lon) > 6:
        raise ValueError(
            "coordinates span more than one UTM zone; configure a local CRS explicitly"
        )
    mean_lon = float(np.nanmean(lon))
    mean_lat = float(np.nanmean(lat))
    zone = min(60, max(1, int((mean_lon + 180) // 6) + 1))
    return f"EPSG:{32600 + zone if mean_lat >= 0 else 32700 + zone}"


def projection_round_trip_error(lon: np.ndarray, lat: np.ndarray, projected_crs: str) -> float:
    x, y = project_coordinates(lon, lat, projected_crs)
    lon_back, lat_back = inverse_project_coordinates(x, y, projected_crs)
    return float(np.nanmax(np.maximum(np.abs(lon - lon_back), np.abs(lat - lat_back))))
