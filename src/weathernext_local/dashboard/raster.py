"""Deterministic precipitation raster rendering for Leaflet ImageOverlay."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from io import BytesIO
from itertools import pairwise

import matplotlib.image as mpl_image
import numpy as np
import xarray as xr

from .data import DashboardDataError, PrecipitationSpec

_BLUE_PALETTE = (
    "#f7fbff",
    "#e7f1fa",
    "#d4e6f4",
    "#bdd7ec",
    "#9ecae1",
    "#7db9da",
    "#5aa2cf",
    "#3787c0",
    "#1d6aad",
    "#0b4f91",
    "#08306b",
)


@dataclass(frozen=True)
class FixedColorScale:
    """Fixed lower bin edges and colors; never fitted to selected data."""

    edges: tuple[float, ...]
    colors: tuple[str, ...]
    labels: tuple[str, ...]
    units: str
    wet_threshold: float

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.colors) or len(self.edges) != len(self.labels):
            raise ValueError("scale edges, colors, and labels must have equal length")
        if not self.edges or self.edges[0] != 0.0:
            raise ValueError("scale must begin at zero")
        if any(right <= left for left, right in pairwise(self.edges)):
            raise ValueError("scale edges must be strictly increasing")


@dataclass(frozen=True)
class RasterOverlay:
    """A browser-ready PNG plus geographic placement metadata."""

    data_url: str
    bounds: tuple[tuple[float, float], tuple[float, float]]
    scale: FixedColorScale
    width: int
    height: int
    no_data_count: int
    dry_count: int


def _number(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(round(value))
    return f"{value:.3g}"


def _labels(edges: tuple[float, ...]) -> tuple[str, ...]:
    labels = [f">0-<{_number(edges[1])}"]
    labels.extend(f"{_number(lower)}-<{_number(upper)}" for lower, upper in pairwise(edges[1:]))
    labels.append(f">={_number(edges[-1])}")
    return tuple(labels)


_WN2_6H_EDGES = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
_HOURLY_EDGES = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0)

WN2_6H_SCALE = FixedColorScale(
    edges=_WN2_6H_EDGES,
    colors=_BLUE_PALETTE,
    labels=_labels(_WN2_6H_EDGES),
    units="mm/6 h",
    wet_threshold=0.1,
)

HOURLY_RATE_SCALE = FixedColorScale(
    edges=_HOURLY_EDGES,
    colors=_BLUE_PALETTE,
    labels=_labels(_HOURLY_EDGES),
    units="mm/h",
    wet_threshold=0.1,
)


def scale_for_precipitation(spec: PrecipitationSpec) -> FixedColorScale:
    """Choose a fixed physical scale from units/interval, never the data range."""

    if spec.display_units == "mm/6 h" and spec.interval_hours == 6.0:
        return WN2_6H_SCALE
    if spec.display_units == "mm/h" and spec.kind == "rate":
        return HOURLY_RATE_SCALE
    if spec.kind == "accumulation" and spec.interval_hours:
        multiplier = float(spec.interval_hours)
        edges = tuple(edge * multiplier for edge in _HOURLY_EDGES)
        # Keep exact zero and human-readable thresholds for this declared period.
        return FixedColorScale(
            edges=edges,
            colors=_BLUE_PALETTE,
            labels=_labels(edges),
            units=spec.display_units,
            wet_threshold=0.1 * multiplier,
        )
    raise DashboardDataError(f"no fixed precipitation scale is defined for {spec.display_units!r}")


def raster_bin_indices(values: np.ndarray, scale: FixedColorScale = WN2_6H_SCALE) -> np.ndarray:
    """Return color indices, using -1 for dry or missing cells."""

    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if np.any(array[finite] < 0):
        raise DashboardDataError("negative precipitation cannot be rasterized honestly")
    result = np.full(array.shape, -1, dtype=np.int16)
    wet = finite & (array > 0.0)
    result[wet] = np.searchsorted(scale.edges, array[wet], side="right") - 1
    result[wet] = np.minimum(result[wet], len(scale.colors) - 1)
    return result


def _hex_rgba(value: str, alpha: int) -> tuple[int, int, int, int]:
    stripped = value.lstrip("#")
    if len(stripped) != 6:
        raise ValueError(f"invalid six-digit color {value!r}")
    return (
        int(stripped[0:2], 16),
        int(stripped[2:4], 16),
        int(stripped[4:6], 16),
        alpha,
    )


def _coordinate(field: xr.DataArray, candidates: tuple[str, ...], role: str) -> str:
    name = next((candidate for candidate in candidates if candidate in field.coords), None)
    if not name or field.coords[name].ndim != 1:
        raise DashboardDataError(f"raster field has no 1-D {role} coordinate")
    return name


def _regular_spacing(values: np.ndarray, role: str) -> float:
    differences = np.diff(values)
    if values.size < 2 or np.any(differences <= 0):
        raise DashboardDataError(f"raster {role} cells are not strictly ordered")
    step = float(np.median(differences))
    tolerance = max(abs(step) * 1e-5, 1e-9)
    if not np.allclose(differences, step, rtol=1e-5, atol=tolerance):
        raise DashboardDataError(
            f"raster {role} cells are irregular; ImageOverlay would misplace them"
        )
    return step


def _prepare_grid(
    field: xr.DataArray,
) -> tuple[np.ndarray, tuple[tuple[float, float], tuple[float, float]]]:
    latitude_name = _coordinate(field, ("latitude", "lat"), "latitude")
    longitude_name = _coordinate(field, ("longitude", "lon"), "longitude")
    latitude_dimension = field.coords[latitude_name].dims[0]
    longitude_dimension = field.coords[longitude_name].dims[0]
    ordered = field.transpose(latitude_dimension, longitude_dimension)
    values = np.asarray(ordered.values, dtype=float)
    latitudes = np.asarray(ordered.coords[latitude_name].values, dtype=float)
    longitudes = np.asarray(ordered.coords[longitude_name].values, dtype=float)

    latitude_order = np.argsort(latitudes)[::-1]
    latitudes = latitudes[latitude_order]
    values = values[latitude_order, :]

    normalized_longitudes = ((longitudes + 180.0) % 360.0) - 180.0
    longitude_order = np.argsort(normalized_longitudes)
    normalized_longitudes = normalized_longitudes[longitude_order]
    values = values[:, longitude_order]
    if np.unique(normalized_longitudes).size != normalized_longitudes.size:
        raise DashboardDataError("longitude normalization creates duplicate cells at the dateline")

    # Spacing checks use ascending coordinates.
    lat_step = _regular_spacing(np.sort(latitudes), "latitude")
    lon_step = _regular_spacing(normalized_longitudes, "longitude")
    south = float(latitudes.min() - lat_step / 2.0)
    north = float(latitudes.max() + lat_step / 2.0)
    west = float(normalized_longitudes.min() - lon_step / 2.0)
    east = float(normalized_longitudes.max() + lon_step / 2.0)
    if np.ptp(latitudes) + lat_step >= 175.0:
        south, north = -90.0, 90.0
    else:
        south, north = max(-90.0, south), min(90.0, north)
    if np.ptp(normalized_longitudes) + lon_step >= 350.0:
        west, east = -180.0, 180.0
    if west < -180.0 or east > 180.0:
        raise DashboardDataError(
            "regional grid crosses the dateline and cannot use one ImageOverlay"
        )
    return values, ((south, west), (north, east))


def render_image_overlay(
    field: xr.DataArray,
    scale: FixedColorScale = WN2_6H_SCALE,
) -> RasterOverlay:
    """Encode one field as a transparent, fixed-bin PNG data URL."""

    field_units = field.attrs.get("units")
    if field_units != scale.units:
        raise DashboardDataError(
            f"field units {field_units!r} do not match raster scale {scale.units!r}"
        )
    values, bounds = _prepare_grid(field)
    indices = raster_bin_indices(values, scale)
    rgba = np.zeros((*indices.shape, 4), dtype=np.uint8)
    for index, color in enumerate(scale.colors):
        # Lighter low bins remain translucent enough to retain OSM context.
        alpha = 150 if index == 0 else min(170 + index * 7, 235)
        rgba[indices == index] = _hex_rgba(color, alpha)

    buffer = BytesIO()
    mpl_image.imsave(buffer, rgba, format="png", origin="upper")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return RasterOverlay(
        data_url=f"data:image/png;base64,{encoded}",
        bounds=bounds,
        scale=scale,
        width=int(values.shape[1]),
        height=int(values.shape[0]),
        no_data_count=int(np.count_nonzero(~np.isfinite(values))),
        dry_count=int(np.count_nonzero(np.isfinite(values) & (values == 0.0))),
    )
