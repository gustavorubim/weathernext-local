"""Pure dataset inspection and precipitation-selection helpers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

PRECIPITATION_VARIABLE = "total_precipitation_6hr"
NOWCAST_PRECIPITATION_VARIABLE = "precipitation_rate"
_LATITUDE_NAMES = ("latitude", "lat")
_LONGITUDE_NAMES = ("longitude", "lon")
_TIME_NAMES = ("valid_time", "prediction_timedelta", "lead_time", "time")
_REFERENCE_TIME_NAMES = (
    "forecast_reference_time",
    "init_time",
    "initial_time",
    "analysis_time",
    "time",
)


class DashboardDataError(ValueError):
    """Raised when a forecast cannot be displayed without guessing."""


@dataclass(frozen=True)
class PrecipitationSpec:
    """Explicit source-to-display conversion for a precipitation variable."""

    variable: str
    source_units: str
    display_units: str
    scale_factor: float
    interval_hours: float | None
    label: str
    kind: str


@dataclass(frozen=True)
class DatasetLayout:
    """Validated coordinate and variable layout used throughout the UI."""

    precipitation: PrecipitationSpec
    latitude_name: str
    longitude_name: str
    time_dimension: str
    time_coordinate: str
    valid_times: tuple[Any, ...]
    sample_dimension: str | None
    sample_size: int

    @property
    def time_size(self) -> int:
        return len(self.valid_times)


@dataclass(frozen=True)
class PrecipitationSummary:
    maximum: float
    area_weighted_mean: float
    wet_area_percent: float
    valid_area_percent: float
    is_global: bool


@dataclass(frozen=True)
class PointSeries:
    latitude: float
    longitude: float
    valid_times: tuple[Any, ...]
    values: tuple[float | None, ...]
    display_units: str
    variable: str


def open_forecast(path: str | Path) -> xr.Dataset:
    """Open a forecast lazily; array payloads load only when selected."""

    return xr.open_dataset(Path(path), chunks=None, cache=False)


def _mapping_at(value: object, key: str) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        child = value.get(key)
        return child if isinstance(child, Mapping) else None
    return None


def _precipitation_declaration(manifest: Mapping[str, Any] | None) -> object:
    if not isinstance(manifest, Mapping):
        return None
    if "primary_precipitation_variable" in manifest:
        return manifest["primary_precipitation_variable"]
    for section_name in ("dashboard", "artifact", "output"):
        section = _mapping_at(manifest, section_name)
        if section and "primary_precipitation_variable" in section:
            return section["primary_precipitation_variable"]
    return None


def _declaration_fields(declaration: object) -> tuple[str | None, Mapping[str, Any]]:
    if isinstance(declaration, str) and declaration.strip():
        return declaration.strip(), {}
    if isinstance(declaration, Mapping):
        value = declaration.get("variable") or declaration.get("name")
        if isinstance(value, str) and value.strip():
            return value.strip(), declaration
        raise DashboardDataError("primary_precipitation_variable object must contain 'variable'")
    if declaration is not None:
        raise DashboardDataError("primary_precipitation_variable must be a string or object")
    return None, {}


def _unit_key(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("\u2212", "-")
        .replace("\u207b", "-")
        .replace("\u00b2", "2")
        .replace("\u00b7", "")
        .replace("_", "")
        .replace(" ", "")
    )


def _parse_duration_hours(value: object) -> tuple[float, str] | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        hours = float(value)
        if hours <= 0:
            return None
        return hours, _format_interval(hours)
    if not isinstance(value, str):
        return None
    compact = value.strip().lower().replace(" ", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(h|hr|hrs|hour|hours)", compact)
    if match:
        hours = float(match.group(1))
        return (hours, _format_interval(hours)) if hours > 0 else None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(m|min|mins|minute|minutes)", compact)
    if match:
        minutes = float(match.group(1))
        hours = minutes / 60.0
        return (hours, _format_interval(hours)) if hours > 0 else None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(s|sec|secs|second|seconds)", compact)
    if match:
        seconds = float(match.group(1))
        hours = seconds / 3600.0
        return (hours, _format_interval(hours)) if hours > 0 else None
    return None


def _format_interval(hours: float) -> str:
    if math.isclose(hours, round(hours)):
        return f"{round(hours)} h"
    minutes = hours * 60.0
    if math.isclose(minutes, round(minutes)):
        return f"{round(minutes)} min"
    seconds = hours * 3600.0
    return f"{seconds:g} s"


def _declared_interval(
    variable: xr.DataArray, declaration: Mapping[str, Any], known_hours: float | None
) -> tuple[float, str] | None:
    values = (
        declaration.get("interval"),
        declaration.get("accumulation_interval"),
        variable.attrs.get("interval"),
        variable.attrs.get("accumulation_interval"),
        variable.attrs.get("accumulation_period"),
        variable.attrs.get("time_interval"),
    )
    for value in values:
        if value in (None, ""):
            continue
        parsed = _parse_duration_hours(value)
        if not parsed:
            raise DashboardDataError(f"unsupported precipitation interval {value!r}")
        if known_hours is not None and not math.isclose(parsed[0], known_hours):
            raise DashboardDataError(
                f"declared precipitation interval {value!r} conflicts with "
                f"the {known_hours:g}-hour variable contract"
            )
        if known_hours is None:
            return parsed
    return (known_hours, _format_interval(known_hours)) if known_hours is not None else None


def _raw_units(variable: xr.DataArray, declaration: Mapping[str, Any]) -> str:
    attr_units = variable.attrs.get("units")
    declared_units = declaration.get("units")
    if attr_units is not None and not isinstance(attr_units, str):
        raise DashboardDataError("precipitation units attribute must be text")
    if declared_units is not None and not isinstance(declared_units, str):
        raise DashboardDataError("manifest precipitation units must be text")
    if (
        isinstance(attr_units, str)
        and attr_units.strip()
        and isinstance(declared_units, str)
        and declared_units.strip()
        and _unit_key(attr_units) != _unit_key(declared_units)
    ):
        raise DashboardDataError(
            "manifest precipitation units conflict with the forecast.nc units attribute"
        )
    if isinstance(attr_units, str) and attr_units.strip():
        return attr_units.strip()
    if isinstance(declared_units, str) and declared_units.strip():
        return declared_units.strip()
    raise DashboardDataError(f"{variable.name!r} has no units; conversion would be ambiguous")


def _spec_for_variable(
    variable: xr.DataArray,
    declaration: Mapping[str, Any],
) -> PrecipitationSpec:
    name = str(variable.name)
    units = _raw_units(variable, declaration)
    unit_key = _unit_key(units)
    known_interval = 6.0 if name == PRECIPITATION_VARIABLE else None
    interval = _declared_interval(variable, declaration, known_interval)
    explicit_label = declaration.get("label") or variable.attrs.get("long_name")
    label = (
        explicit_label.strip()
        if isinstance(explicit_label, str) and explicit_label.strip()
        else name.replace("_", " ").title()
    )

    metre_units = {"m", "metre", "metres", "meter", "meters"}
    millimetre_units = {"mm", "millimetre", "millimetres", "millimeter", "millimeters"}
    hourly_mm = {
        "mm/h",
        "mm/hr",
        "mm/hour",
        "mmh-1",
        "mmhr-1",
        "mmhour-1",
        "kgm-2h-1",
    }
    accumulated_mm_equivalent = {"kgm-2"}
    per_second_kg = {
        "kgm-2s-1",
        "kg/m2/s",
        "kgm^-2s^-1",
    }
    per_second_m = {"m/s", "ms-1", "msec-1", "msecond-1"}
    per_second_mm = {"mm/s", "mms-1", "mmsec-1", "mmsecond-1"}

    # Units such as ``mm/15 min`` encode both quantity and interval.
    interval_unit = re.fullmatch(
        r"mm/([0-9]+(?:\.[0-9]+)?)(h|hr|hour|min|minute|s|sec|second)",
        unit_key,
    )
    if interval_unit:
        parsed_interval = _parse_duration_hours(f"{interval_unit.group(1)}{interval_unit.group(2)}")
        if not parsed_interval:
            raise DashboardDataError(f"unsupported precipitation units {units!r}")
        hours, interval_label = parsed_interval
        return PrecipitationSpec(
            variable=name,
            source_units=units,
            display_units=f"mm/{interval_label}",
            scale_factor=1.0,
            interval_hours=hours,
            label=label,
            kind="accumulation",
        )

    if unit_key in metre_units:
        if not interval:
            raise DashboardDataError(f"{name!r} is an accumulation in metres but has no interval")
        hours, interval_label = interval
        return PrecipitationSpec(
            variable=name,
            source_units=units,
            display_units=f"mm/{interval_label}",
            scale_factor=1000.0,
            interval_hours=hours,
            label=label,
            kind="accumulation",
        )
    if unit_key in millimetre_units:
        if not interval:
            raise DashboardDataError(
                f"{name!r} is an accumulation in millimetres but has no interval"
            )
        hours, interval_label = interval
        return PrecipitationSpec(
            variable=name,
            source_units=units,
            display_units=f"mm/{interval_label}",
            scale_factor=1.0,
            interval_hours=hours,
            label=label,
            kind="accumulation",
        )
    if unit_key in accumulated_mm_equivalent:
        if not interval:
            raise DashboardDataError(f"{name!r} is an accumulation in kg m-2 but has no interval")
        hours, interval_label = interval
        return PrecipitationSpec(
            variable=name,
            source_units=units,
            display_units=f"mm/{interval_label}",
            scale_factor=1.0,
            interval_hours=hours,
            label=label,
            kind="accumulation",
        )
    if unit_key in hourly_mm:
        return PrecipitationSpec(
            variable=name,
            source_units=units,
            display_units="mm/h",
            scale_factor=1.0,
            interval_hours=None,
            label=label,
            kind="rate",
        )
    if unit_key in per_second_kg:
        return PrecipitationSpec(
            variable=name,
            source_units=units,
            display_units="mm/h",
            scale_factor=3600.0,
            interval_hours=None,
            label=label,
            kind="rate",
        )
    if unit_key in per_second_m:
        return PrecipitationSpec(
            variable=name,
            source_units=units,
            display_units="mm/h",
            scale_factor=3_600_000.0,
            interval_hours=None,
            label=label,
            kind="rate",
        )
    if unit_key in per_second_mm:
        return PrecipitationSpec(
            variable=name,
            source_units=units,
            display_units="mm/h",
            scale_factor=3600.0,
            interval_hours=None,
            label=label,
            kind="rate",
        )
    raise DashboardDataError(f"unsupported precipitation units {units!r} for {name!r}")


def detect_precipitation(
    dataset: xr.Dataset, manifest: Mapping[str, Any] | None = None
) -> PrecipitationSpec:
    """Select precipitation only from an explicit declaration or known names."""

    declared_name, declaration = _declaration_fields(_precipitation_declaration(manifest))
    if declared_name:
        if declared_name not in dataset.data_vars:
            raise DashboardDataError(
                f"manifest precipitation variable {declared_name!r} is not in forecast.nc"
            )
        return _spec_for_variable(dataset[declared_name], declaration)

    candidates = [
        name
        for name in (PRECIPITATION_VARIABLE, NOWCAST_PRECIPITATION_VARIABLE)
        if name in dataset.data_vars
    ]
    if not candidates:
        raise DashboardDataError(
            "forecast.nc has no known precipitation variable and the manifest "
            "does not declare primary_precipitation_variable"
        )
    if len(candidates) > 1:
        raise DashboardDataError(
            "forecast.nc has multiple known precipitation variables; declare "
            "primary_precipitation_variable in manifest.json"
        )
    return _spec_for_variable(dataset[candidates[0]], {})


def _coordinate_name(variable: xr.DataArray, names: Sequence[str], role: str) -> str:
    for name in names:
        if name in variable.coords:
            coordinate = variable.coords[name]
            if coordinate.ndim == 1 and coordinate.dims[0] in variable.dims:
                return name
    raise DashboardDataError(
        f"precipitation variable needs a one-dimensional {role} coordinate "
        f"named {' or '.join(names)}"
    )


def _time_axis(variable: xr.DataArray, spatial_dimensions: set[str]) -> tuple[str, str]:
    for name in _TIME_NAMES:
        if name in variable.coords:
            coordinate = variable.coords[name]
            if (
                coordinate.ndim == 1
                and coordinate.dims[0] in variable.dims
                and coordinate.dims[0] not in spatial_dimensions
                and coordinate.dims[0] != "sample"
            ):
                return coordinate.dims[0], name

    inferred: list[tuple[str, str]] = []
    for name, coordinate in variable.coords.items():
        if coordinate.ndim != 1 or coordinate.dims[0] not in variable.dims:
            continue
        dimension = coordinate.dims[0]
        if dimension in spatial_dimensions or dimension == "sample":
            continue
        if np.issubdtype(coordinate.dtype, np.datetime64) or np.issubdtype(
            coordinate.dtype, np.timedelta64
        ):
            inferred.append((dimension, name))
    if len(inferred) == 1:
        return inferred[0]
    raise DashboardDataError(
        "precipitation variable needs one explicit valid-time or forecast-lead axis"
    )


def _scalar_datetime(dataset: xr.Dataset, excluded: str) -> np.datetime64 | None:
    for name in _REFERENCE_TIME_NAMES:
        if name == excluded:
            continue
        if name in dataset.coords:
            values = np.asarray(dataset.coords[name].values)
            if values.size == 1 and np.issubdtype(values.dtype, np.datetime64):
                value = values.reshape(-1)[0].astype("datetime64[ns]")
                if not np.isnat(value):
                    return value
        raw = dataset.attrs.get(name)
        if isinstance(raw, (str, datetime, np.datetime64)):
            try:
                value = np.datetime64(raw, "ns")
            except (TypeError, ValueError):
                continue
            if not np.isnat(value):
                return value
    return None


def _resolved_valid_times(dataset: xr.Dataset, time_coordinate: str) -> tuple[Any, ...]:
    coordinate = dataset.coords[time_coordinate]
    values = np.asarray(coordinate.values)
    if values.ndim != 1 or values.size == 0:
        raise DashboardDataError("forecast time coordinate must be non-empty and 1-D")
    if np.issubdtype(values.dtype, np.datetime64):
        cast = values.astype("datetime64[ns]")
        if np.isnat(cast).any():
            raise DashboardDataError("forecast time coordinate contains NaT")
        return tuple(cast)
    if np.issubdtype(values.dtype, np.timedelta64):
        reference = _scalar_datetime(dataset, time_coordinate)
        if reference is None:
            raise DashboardDataError(
                "forecast lead times require a scalar datetime reference coordinate"
            )
        return tuple(reference + values.astype("timedelta64[ns]"))
    # cftime and Python datetime coordinates can have object dtype.
    if values.dtype.kind == "O" and all(hasattr(item, "isoformat") for item in values.tolist()):
        return tuple(values.tolist())
    raise DashboardDataError(
        "forecast time coordinate is not datetime-valued and cannot yield valid times"
    )


def _validate_axis(coordinate: xr.DataArray, role: str) -> None:
    raw_units = coordinate.attrs.get("units")
    if not isinstance(raw_units, str) or not raw_units.strip():
        raise DashboardDataError(
            f"{role} coordinate must declare degree units; radians versus degrees cannot be guessed"
        )
    direction_tokens = re.split(r"[\s_-]+", raw_units.strip().casefold())
    if direction_tokens[-1] in {"s", "south", "w", "west"}:
        raise DashboardDataError(
            f"{role} coordinate units {raw_units!r} use an unsupported south/west-positive direction"
        )
    unit_key = re.sub(r"[\s_-]", "", raw_units.strip().casefold())
    generic_degrees = {"degree", "degrees"}
    role_degrees = (
        {"degreen", "degreesn", "degreenorth", "degreesnorth"}
        if role == "latitude"
        else {"degreee", "degreese", "degreeeast", "degreeseast"}
    )
    if unit_key not in generic_degrees | role_degrees:
        raise DashboardDataError(
            f"{role} coordinate units {raw_units!r} are not recognized degree units"
        )
    source_values = np.asarray(coordinate.values)
    if (
        not np.issubdtype(source_values.dtype, np.number)
        or np.issubdtype(source_values.dtype, np.bool_)
        or np.issubdtype(source_values.dtype, np.complexfloating)
    ):
        raise DashboardDataError(
            f"{role} coordinate must be real numeric data, found {source_values.dtype}"
        )
    values = source_values.astype(float)
    if values.ndim != 1 or values.size < 2:
        raise DashboardDataError(f"{role} coordinate must contain at least two cells")
    if not np.isfinite(values).all():
        raise DashboardDataError(f"{role} coordinate contains non-finite values")
    if np.unique(values).size != values.size:
        raise DashboardDataError(f"{role} coordinate contains duplicate cells")
    if role == "latitude" and (np.any(values < -90.0) or np.any(values > 90.0)):
        raise DashboardDataError("latitude coordinate is outside -90..90 degrees")
    if role == "longitude":
        signed = np.all((values >= -180.0) & (values <= 180.0))
        unsigned = np.all((values >= 0.0) & (values <= 360.0))
        if not (signed or unsigned):
            raise DashboardDataError(
                "longitude coordinate is outside a recognized -180..180 or 0..360 convention"
            )
        normalized = ((values + 180.0) % 360.0) - 180.0
        if np.unique(normalized).size != normalized.size:
            raise DashboardDataError(
                "longitude coordinate contains duplicate cells after dateline normalization"
            )
    differences = np.diff(values)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise DashboardDataError(f"{role} coordinate must be monotonic")
    absolute_differences = np.abs(differences)
    step = float(np.median(absolute_differences))
    tolerance = max(step * 1e-5, 1e-9)
    if not np.allclose(absolute_differences, step, rtol=1e-5, atol=tolerance):
        raise DashboardDataError(f"{role} coordinate must be regularly spaced for ImageOverlay")


def inspect_dataset(
    dataset: xr.Dataset, manifest: Mapping[str, Any] | None = None
) -> DatasetLayout:
    """Validate the precise subset of a forecast needed by the dashboard."""

    precipitation = detect_precipitation(dataset, manifest)
    variable = dataset[precipitation.variable]
    latitude_name = _coordinate_name(variable, _LATITUDE_NAMES, "latitude")
    longitude_name = _coordinate_name(variable, _LONGITUDE_NAMES, "longitude")
    _validate_axis(variable.coords[latitude_name], "latitude")
    _validate_axis(variable.coords[longitude_name], "longitude")
    latitude_dimension = variable.coords[latitude_name].dims[0]
    longitude_dimension = variable.coords[longitude_name].dims[0]
    if latitude_dimension == longitude_dimension:
        raise DashboardDataError(
            "latitude and longitude need distinct dimensions for a rectangular raster"
        )
    time_dimension, time_coordinate = _time_axis(
        variable, {latitude_dimension, longitude_dimension}
    )
    valid_times = _resolved_valid_times(dataset, time_coordinate)
    sample_dimension = "sample" if "sample" in variable.dims else None
    sample_size = int(variable.sizes["sample"]) if sample_dimension else 0
    if sample_dimension and sample_size < 1:
        raise DashboardDataError("sample dimension is empty")

    allowed = {
        latitude_dimension,
        longitude_dimension,
        time_dimension,
        *([sample_dimension] if sample_dimension else []),
    }
    unexpected = [dimension for dimension in variable.dims if dimension not in allowed]
    if unexpected:
        raise DashboardDataError(
            "precipitation variable has unsupported dimensions: " + ", ".join(unexpected)
        )
    if len(valid_times) != variable.sizes[time_dimension]:
        raise DashboardDataError("valid-time coordinate length does not match data")
    return DatasetLayout(
        precipitation=precipitation,
        latitude_name=latitude_name,
        longitude_name=longitude_name,
        time_dimension=time_dimension,
        time_coordinate=time_coordinate,
        valid_times=valid_times,
        sample_dimension=sample_dimension,
        sample_size=sample_size,
    )


def format_valid_time(value: Any) -> str:
    """Format a source valid time without asserting an unavailable timezone."""

    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value.astype("datetime64[m]"), unit="m").replace("T", " ")
    if hasattr(value, "isoformat"):
        return str(value.isoformat()).replace("T", " ")
    return str(value)


def sample_options(dataset: xr.Dataset, layout: DatasetLayout) -> tuple[tuple[str, int], ...]:
    if not layout.sample_dimension:
        return ()
    coordinate = dataset.coords.get(layout.sample_dimension)
    values = (
        np.asarray(coordinate.values).tolist()
        if coordinate is not None
        else list(range(layout.sample_size))
    )
    return tuple((f"Member {value}", index) for index, value in enumerate(values))


def _ensemble_selection(
    variable: xr.DataArray,
    layout: DatasetLayout,
    ensemble_mode: str,
    member_index: int,
) -> xr.DataArray:
    if not layout.sample_dimension:
        return variable
    if ensemble_mode == "member":
        if not 0 <= int(member_index) < layout.sample_size:
            raise DashboardDataError(
                f"member index {member_index} is outside 0..{layout.sample_size - 1}"
            )
        return variable.isel({layout.sample_dimension: int(member_index)})
    if ensemble_mode == "mean":
        return variable.mean(layout.sample_dimension, skipna=False)
    if ensemble_mode == "median":
        return variable.median(layout.sample_dimension, skipna=False)
    if ensemble_mode == "maximum":
        return variable.max(layout.sample_dimension, skipna=False)
    raise DashboardDataError(f"unsupported ensemble mode {ensemble_mode!r}")


def _convert_to_display_units(variable: xr.DataArray, spec: PrecipitationSpec) -> xr.DataArray:
    converted = variable.astype(float) * spec.scale_factor
    converted.attrs = dict(variable.attrs)
    converted.attrs.update(
        {
            "units": spec.display_units,
            "source_units": spec.source_units,
            "display_scale_factor": spec.scale_factor,
        }
    )
    return converted


def _floor_nonphysical_map_values(converted: xr.DataArray) -> xr.DataArray:
    """Apply an explicit display-only physical floor while retaining audit metadata."""

    values = np.asarray(converted.values, dtype=float)
    negative = np.isfinite(values) & (values < 0)
    if not np.any(negative):
        return converted
    floored = converted.copy(data=np.where(negative, 0.0, values))
    floored.attrs = dict(converted.attrs)
    floored.attrs.update(
        {
            "display_floor_applied": True,
            "display_floor_count": int(np.count_nonzero(negative)),
            "display_floor_minimum": float(np.min(values[negative])),
            "display_floor_policy": "physical precipitation floor at zero for map and KPIs",
            "raw_artifact_modified": False,
        }
    )
    return floored


def select_precipitation(
    dataset: xr.Dataset,
    time_index: int,
    ensemble_mode: str = "mean",
    member_index: int = 0,
    manifest: Mapping[str, Any] | None = None,
) -> xr.DataArray:
    """Load one source-backed map field and convert it to display units."""

    layout = inspect_dataset(dataset, manifest)
    if not 0 <= int(time_index) < layout.time_size:
        raise DashboardDataError(f"time index {time_index} is outside 0..{layout.time_size - 1}")
    variable = dataset[layout.precipitation.variable].isel({layout.time_dimension: int(time_index)})
    variable = _ensemble_selection(variable, layout, ensemble_mode, member_index)
    latitude_dimension = variable.coords[layout.latitude_name].dims[0]
    longitude_dimension = variable.coords[layout.longitude_name].dims[0]
    variable = variable.transpose(latitude_dimension, longitude_dimension).load()
    converted = _floor_nonphysical_map_values(
        _convert_to_display_units(variable, layout.precipitation)
    )
    converted.name = layout.precipitation.variable
    return converted


def _area_weights(latitude: np.ndarray, longitude_size: int) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitude.astype(float)))
    weights[np.abs(weights) < 1e-12] = 0.0
    return np.repeat(weights[:, None], longitude_size, axis=1)


def _is_global(latitude: np.ndarray, longitude: np.ndarray) -> bool:
    lat_step = float(np.median(np.abs(np.diff(np.sort(latitude)))))
    lon_normalized = np.sort(((longitude + 180.0) % 360.0) - 180.0)
    lon_step = float(np.median(np.abs(np.diff(lon_normalized))))
    return np.ptp(latitude) + lat_step >= 175.0 and np.ptp(lon_normalized) + lon_step >= 350.0


def summarize_precipitation(
    field: xr.DataArray, wet_threshold: float = 0.1
) -> PrecipitationSummary:
    """Compute map KPIs using latitude area weights over valid cells."""

    latitude_name = next((name for name in _LATITUDE_NAMES if name in field.coords), None)
    longitude_name = next((name for name in _LONGITUDE_NAMES if name in field.coords), None)
    if not latitude_name or not longitude_name:
        raise DashboardDataError("field has no recognizable latitude/longitude axes")
    latitude = np.asarray(field.coords[latitude_name].values, dtype=float)
    longitude = np.asarray(field.coords[longitude_name].values, dtype=float)
    values = np.asarray(field.values, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        raise DashboardDataError("selected precipitation field contains no valid cells")
    weights = _area_weights(latitude, longitude.size)
    total_domain_weight = float(weights.sum())
    valid_weight = float(weights[valid].sum())
    if valid_weight <= 0 or total_domain_weight <= 0:
        raise DashboardDataError("selected grid has no positive spatial area weight")
    mean = float(np.sum(values[valid] * weights[valid]) / valid_weight)
    wet = valid & (values >= wet_threshold)
    wet_percent = float(100.0 * weights[wet].sum() / valid_weight)
    valid_percent = float(100.0 * valid_weight / total_domain_weight)
    return PrecipitationSummary(
        maximum=float(np.nanmax(values)),
        area_weighted_mean=mean,
        wet_area_percent=wet_percent,
        valid_area_percent=valid_percent,
        is_global=_is_global(latitude, longitude),
    )


def _nearest_longitude_index(longitudes: np.ndarray, click_longitude: float) -> int:
    angular_distance = np.abs(
        ((longitudes.astype(float) - float(click_longitude) + 180.0) % 360.0) - 180.0
    )
    return int(np.argmin(angular_distance))


def nearest_cell_series(
    dataset: xr.Dataset,
    latitude: float,
    longitude: float,
    ensemble_mode: str = "mean",
    member_index: int = 0,
    manifest: Mapping[str, Any] | None = None,
) -> PointSeries:
    """Load the precipitation time series at the grid cell nearest a map click."""

    layout = inspect_dataset(dataset, manifest)
    variable = dataset[layout.precipitation.variable]
    latitudes = np.asarray(variable.coords[layout.latitude_name].values, dtype=float)
    longitudes = np.asarray(variable.coords[layout.longitude_name].values, dtype=float)
    lat_index = int(np.argmin(np.abs(latitudes - float(latitude))))
    lon_index = _nearest_longitude_index(longitudes, float(longitude))
    latitude_dimension = variable.coords[layout.latitude_name].dims[0]
    longitude_dimension = variable.coords[layout.longitude_name].dims[0]
    point = variable.isel({latitude_dimension: lat_index, longitude_dimension: lon_index})
    point = _ensemble_selection(point, layout, ensemble_mode, member_index).load()
    converted = _convert_to_display_units(point, layout.precipitation)
    values = np.asarray(converted.values, dtype=float).reshape(-1)
    if values.size != layout.time_size:
        raise DashboardDataError("nearest-cell series length does not match valid times")
    return PointSeries(
        latitude=float(latitudes[lat_index]),
        longitude=float(longitudes[lon_index]),
        valid_times=layout.valid_times,
        values=tuple(float(value) if np.isfinite(value) else None for value in values),
        display_units=layout.precipitation.display_units,
        variable=layout.precipitation.variable,
    )
