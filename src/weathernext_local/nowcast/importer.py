"""Import an authorized Global MetNet output into the local run catalog.

This is an ingestion path, not a model implementation.  Source identity and
the user's authorization attestation are recorded so the dashboard cannot
mistake imported data for locally executed inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from weathernext_local.nowcast.global_metnet import global_metnet_status
from weathernext_local.paths import runs_root

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
KNOWN_VARIABLES = (
    "precipitation_rate",
    "precipitation",
    "total_precipitation",
    "tp",
)
RATE_UNITS = {
    "mm/h",
    "mm h-1",
    "mm hr-1",
    "kg m-2 h-1",
    "kg m-2 s-1",
    "m/s",
    "mm/s",
}
ACCUMULATION_UNITS = {"mm", "m", "kg m-2"}
GENERIC_DEGREE_UNITS = {"degree", "degrees"}
LATITUDE_DEGREE_UNITS = {"degreen", "degreesn", "degreenorth", "degreesnorth"}
LONGITUDE_DEGREE_UNITS = {"degreee", "degreese", "degreeeast", "degreeseast"}


class NowcastImportError(RuntimeError):
    """Raised when supplied nowcast data does not meet the canonical contract."""


class AuthorizationRequiredError(NowcastImportError, PermissionError):
    """Raised unless the caller attests that the source may be ingested."""


class OriginAttestationRequiredError(NowcastImportError, ValueError):
    """Raised unless the caller separately attests and documents model origin."""


@dataclass(frozen=True, slots=True)
class ImportedNowcastRun:
    run_id: str
    run_dir: Path
    forecast_path: Path
    manifest_path: Path


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_source(
    source: Path, destination: Path, chunk_size: int = 8 * 1024 * 1024
) -> tuple[int, str]:
    """Copy once while hashing, binding validation to the exact imported bytes."""

    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        while chunk := input_stream.read(chunk_size):
            output_stream.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    return size, digest.hexdigest()


def _choose_name(
    dataset: xr.Dataset, explicit: str | None, candidates: tuple[str, ...], kind: str
) -> str:
    if explicit is not None:
        if explicit not in dataset.variables:
            raise NowcastImportError(
                f"Requested {kind} {explicit!r} is absent from the source dataset."
            )
        return explicit
    found = [candidate for candidate in candidates if candidate in dataset.variables]
    if len(found) == 1:
        return found[0]
    if not found:
        raise NowcastImportError(
            f"Could not identify a {kind}. Pass its name explicitly. Candidates: {', '.join(candidates)}."
        )
    raise NowcastImportError(
        f"Source contains multiple possible {kind}s ({', '.join(found)}); select one explicitly."
    )


@contextmanager
def _open_decoded_dataset(path: Path) -> Iterator[xr.Dataset]:
    """Decode CF metadata while retaining ownership of the raw backend handle."""

    with xr.open_dataset(path, decode_times=False) as encoded:
        try:
            decoded = xr.decode_cf(encoded)
        except Exception as exc:
            raise NowcastImportError(f"Could not decode source NetCDF coordinates: {exc}") from exc
        try:
            yield decoded
        finally:
            decoded.close()


def _normalized_units(value: Any) -> str:
    return " ".join(str(value).strip().replace("\N{MINUS SIGN}", "-").split())


def _validate_coordinate_units(data: xr.DataArray, name: str) -> str:
    raw = data.attrs.get("units")
    if not isinstance(raw, str) or not raw.strip():
        raise NowcastImportError(
            f"{name} coordinate must declare degree units; the importer never guesses radians versus degrees."
        )
    direction_tokens = re.split(r"[\s_-]+", raw.strip().casefold())
    if direction_tokens[-1] in {"s", "south", "w", "west"}:
        raise NowcastImportError(
            f"{name} coordinate units {raw!r} use an unsupported south/west-positive direction. "
            "Convert the coordinate values to conventional north/east-positive degrees before importing."
        )
    normalized = re.sub(r"[\s_-]", "", raw.strip().casefold())
    accepted = GENERIC_DEGREE_UNITS | (
        LATITUDE_DEGREE_UNITS if name == "latitude" else LONGITUDE_DEGREE_UNITS
    )
    if normalized not in accepted:
        raise NowcastImportError(
            f"{name} coordinate units {raw!r} are not recognized degree units. "
            "Convert the source explicitly before importing."
        )
    return raw.strip()


def _validate_axis(data: xr.DataArray, name: str) -> None:
    if data.ndim != 1:
        raise NowcastImportError(
            f"{name} must be a one-dimensional coordinate, found {data.dims!r}."
        )
    values = np.asarray(data.values)
    if values.size < 2:
        raise NowcastImportError(f"{name} coordinate must contain at least two cells.")
    if (
        not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.bool_)
        or np.issubdtype(values.dtype, np.complexfloating)
    ):
        raise NowcastImportError(
            f"{name} coordinate must be real numeric data, found {values.dtype}."
        )
    numeric_values = values.astype(np.float64)
    if not np.all(np.isfinite(numeric_values)):
        raise NowcastImportError(f"{name} coordinate contains non-finite values.")
    if np.unique(numeric_values).size != numeric_values.size:
        raise NowcastImportError(f"{name} coordinate contains duplicate values.")
    minimum = float(np.min(numeric_values))
    maximum = float(np.max(numeric_values))
    if name == "latitude" and (minimum < -90.0 or maximum > 90.0):
        raise NowcastImportError("latitude values must be within [-90, 90] degrees.")
    if name == "longitude":
        uses_signed_domain = minimum >= -180.0 and maximum <= 180.0
        uses_unsigned_domain = minimum >= 0.0 and maximum <= 360.0
        if not (uses_signed_domain or uses_unsigned_domain):
            raise NowcastImportError(
                "longitude values must consistently use either [-180, 180] or [0, 360] degrees."
            )
        wrapped = np.round(np.mod(numeric_values, 360.0), decimals=8)
        if np.unique(wrapped).size != wrapped.size:
            raise NowcastImportError(
                "longitude contains duplicate locations after modulo-360 normalization."
            )
    if numeric_values.size > 1:
        deltas = np.diff(numeric_values)
        if not (np.all(deltas > 0) or np.all(deltas < 0)):
            raise NowcastImportError(f"{name} coordinate must be strictly monotonic.")
        if not np.allclose(np.abs(deltas), abs(float(deltas[0])), rtol=1e-5, atol=1e-8):
            raise NowcastImportError(f"{name} coordinate must use regular spacing.")


def _validate_time(data: xr.DataArray) -> None:
    if data.ndim != 1 or data.size == 0:
        raise NowcastImportError(
            "The forecast time coordinate must be non-empty and one-dimensional."
        )
    values = np.asarray(data.values)
    if not np.issubdtype(values.dtype, np.datetime64):
        raise NowcastImportError(
            f"Forecast time values must be decoded absolute datetimes; found {values.dtype}."
        )
    if np.isnat(values).any():
        raise NowcastImportError("Forecast time coordinate contains NaT values.")
    if np.unique(values).size != values.size:
        raise NowcastImportError("Forecast time coordinate contains duplicate values.")


def _infer_step_minutes(time: xr.DataArray) -> int | None:
    values = np.asarray(time.values)
    if values.size < 2:
        return None
    try:
        nanoseconds = values.astype("datetime64[ns]").astype(np.int64)
    except (OverflowError, ValueError) as exc:
        raise NowcastImportError(
            "Forecast times cannot be represented exactly at nanosecond precision."
        ) from exc
    deltas = np.diff(nanoseconds)
    if np.any(deltas <= 0) or np.unique(deltas).size != 1:
        raise NowcastImportError(
            "Forecast times must be strictly increasing with a regular interval."
        )
    nanoseconds_per_minute = 60 * 1_000_000_000
    interval_nanoseconds = int(deltas[0])
    if interval_nanoseconds % nanoseconds_per_minute:
        raise NowcastImportError("Forecast interval must resolve to a whole number of minutes.")
    return interval_nanoseconds // nanoseconds_per_minute


def _validate_precipitation(data: xr.DataArray, time_dimension: str) -> None:
    if (
        not np.issubdtype(data.dtype, np.number)
        or np.issubdtype(data.dtype, np.bool_)
        or np.issubdtype(data.dtype, np.complexfloating)
    ):
        raise NowcastImportError(f"Precipitation must be real numeric data, found {data.dtype}.")
    if "sample" in data.dims and data.sizes["sample"] < 1:
        raise NowcastImportError("The optional sample dimension must contain at least one member.")
    sample_indices = range(data.sizes["sample"]) if "sample" in data.dims else (None,)
    for time_index in range(data.sizes[time_dimension]):
        for sample_index in sample_indices:
            indexers = {time_dimension: time_index}
            if sample_index is not None:
                indexers["sample"] = sample_index
            values = np.asarray(data.isel(indexers).values)
            if np.isinf(values).any():
                raise NowcastImportError("Precipitation contains infinite values.")
            finite = np.isfinite(values)
            if not np.any(finite):
                raise NowcastImportError(
                    "Every selectable sample and valid time must contain at least one finite "
                    "precipitation value."
                )
            if np.any(values[finite] < 0):
                raise NowcastImportError("Precipitation contains negative values.")


def _default_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"global-metnet-import-{timestamp}-{uuid.uuid4().hex[:8]}"


def import_global_metnet_output(
    source: str | os.PathLike[str],
    *,
    authorized: bool,
    origin_attested: bool,
    origin_evidence: str,
    variable: str | None = None,
    latitude: str | None = None,
    longitude: str | None = None,
    time: str | None = None,
    accumulation_interval_minutes: int | None = None,
    run_id: str | None = None,
    output_root: str | os.PathLike[str] | None = None,
) -> ImportedNowcastRun:
    """Validate and atomically catalog an authorized Global MetNet NetCDF file.

    Rate data must carry explicit rate units.  Accumulated data must additionally
    specify its interval; the importer never guesses precipitation semantics.
    """

    if authorized is not True:
        raise AuthorizationRequiredError(
            "Import refused: explicitly attest that you are authorized to use and store this source output."
        )
    if (
        origin_attested is not True
        or not isinstance(origin_evidence, str)
        or not origin_evidence.strip()
    ):
        raise OriginAttestationRequiredError(
            "Import refused: separately attest that this is a Global MetNet output and "
            "provide a non-empty origin evidence identifier. The repository records this "
            "as a caller claim; it cannot independently verify model origin."
        )
    source_path = Path(source).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise NowcastImportError(f"Source is not a regular file: {source_path}")
    selected_run_id = run_id or _default_run_id()
    if not RUN_ID_PATTERN.fullmatch(selected_run_id):
        raise NowcastImportError(
            "run_id must begin with an alphanumeric character and contain only letters, numbers, '.', '_', or '-'."
        )

    root = Path(output_root).expanduser().resolve() if output_root is not None else runs_root()
    root.mkdir(parents=True, exist_ok=True)
    final_dir = root / selected_run_id
    if final_dir.exists():
        raise NowcastImportError(f"Run already exists and will not be overwritten: {final_dir}")

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{selected_run_id}.", dir=root))
    snapshot_dir: Path | None = None
    try:
        forecast_path = temp_dir / "forecast.nc"
        manifest_path = temp_dir / "manifest.json"
        snapshot_dir = Path(tempfile.mkdtemp(prefix="weathernext-metnet-source-"))
        source_snapshot = snapshot_dir / "source.snapshot.nc"
        source_size, source_digest = _snapshot_source(source_path, source_snapshot)
        with _open_decoded_dataset(source_snapshot) as opened:
            variable_name = _choose_name(
                opened, variable, KNOWN_VARIABLES, "precipitation variable"
            )
            if variable_name not in opened.data_vars:
                raise NowcastImportError(
                    f"Precipitation {variable_name!r} must be a NetCDF data variable, not a coordinate."
                )
            latitude_name = _choose_name(
                opened, latitude, ("latitude", "lat"), "latitude coordinate"
            )
            longitude_name = _choose_name(
                opened, longitude, ("longitude", "lon"), "longitude coordinate"
            )
            time_name = _choose_name(
                opened,
                time,
                ("valid_time", "time", "lead_time", "prediction_timedelta"),
                "forecast time coordinate",
            )
            missing_coordinates = [
                name
                for name in (latitude_name, longitude_name, time_name)
                if name not in opened.coords
            ]
            if missing_coordinates:
                raise NowcastImportError(
                    "Selected axes must be NetCDF coordinates, not ordinary data variables: "
                    + ", ".join(missing_coordinates)
                )

            _validate_axis(opened[latitude_name], "latitude")
            _validate_axis(opened[longitude_name], "longitude")
            _validate_time(opened[time_name])
            step_minutes = _infer_step_minutes(opened[time_name])

            precipitation = opened[variable_name]
            axis_dims = (
                opened[latitude_name].dims[0],
                opened[longitude_name].dims[0],
                opened[time_name].dims[0],
            )
            if "sample" in axis_dims:
                raise NowcastImportError(
                    "The reserved 'sample' ensemble dimension cannot also represent time, "
                    "latitude, or longitude."
                )
            if len(set(axis_dims)) != 3:
                raise NowcastImportError(
                    "Time, latitude, and longitude must use three independent dimensions; "
                    f"found {axis_dims!r}."
                )
            latitude_units = _validate_coordinate_units(opened[latitude_name], "latitude")
            longitude_units = _validate_coordinate_units(opened[longitude_name], "longitude")
            required_dims = set(axis_dims)
            if not required_dims.issubset(precipitation.dims):
                raise NowcastImportError(
                    f"{variable_name!r} must depend on the selected time, latitude, and longitude axes; "
                    f"found dimensions {precipitation.dims!r}."
                )
            extra_dims = [
                dim for dim in precipitation.dims if dim not in required_dims and dim != "sample"
            ]
            non_singleton = [dim for dim in extra_dims if precipitation.sizes[dim] != 1]
            if non_singleton:
                raise NowcastImportError(
                    "Unsupported non-spatial precipitation dimensions: " + ", ".join(non_singleton)
                )
            _validate_precipitation(precipitation, axis_dims[2])

            units = _normalized_units(precipitation.attrs.get("units", ""))
            if not units:
                raise NowcastImportError(
                    f"{variable_name!r} has no units attribute. Add truthful units before importing."
                )
            if units in ACCUMULATION_UNITS:
                if (
                    not isinstance(accumulation_interval_minutes, Integral)
                    or isinstance(accumulation_interval_minutes, bool)
                    or accumulation_interval_minutes <= 0
                ):
                    raise NowcastImportError(
                        f"Accumulated precipitation units {units!r} require a positive, "
                        "non-boolean integer accumulation_interval_minutes value."
                    )
                interval_minutes = int(accumulation_interval_minutes)
                interval = f"{interval_minutes} min"
                canonical_name = "precipitation"
                label = f"{interval_minutes}-minute accumulated precipitation"
            elif units in RATE_UNITS:
                if accumulation_interval_minutes is not None:
                    raise NowcastImportError(
                        "Do not provide an accumulation interval for precipitation-rate data."
                    )
                interval = None
                canonical_name = "precipitation_rate"
                label = "precipitation rate"
            else:
                raise NowcastImportError(
                    f"Unsupported precipitation units {units!r}. Accepted rate units: "
                    f"{', '.join(sorted(RATE_UNITS))}; accumulation units: "
                    f"{', '.join(sorted(ACCUMULATION_UNITS))}."
                )

            squeeze_dims = {dim: 0 for dim in extra_dims}
            selected = opened[[variable_name]].isel(squeeze_dims, drop=True)
            rename: dict[str, str] = {}
            for old, new in (
                (variable_name, canonical_name),
                (latitude_name, "latitude"),
                (longitude_name, "longitude"),
                (time_name, "valid_time"),
            ):
                if old != new:
                    rename[old] = new
            selected = selected.rename(rename)
            if "sample" not in selected[canonical_name].dims:
                selected = selected.expand_dims(sample=[0])
            selected[canonical_name].attrs = dict(precipitation.attrs)
            selected[canonical_name].attrs["units"] = units
            selected["latitude"].attrs = dict(selected["latitude"].attrs)
            selected["latitude"].attrs.update(
                {"units": "degrees_north", "source_units": latitude_units}
            )
            selected["longitude"].attrs = dict(selected["longitude"].attrs)
            selected["longitude"].attrs.update(
                {"units": "degrees_east", "source_units": longitude_units}
            )
            selected.attrs = dict(opened.attrs)
            selected.attrs.update(
                {
                    "weathernext_local_import_mode": "authorized_output_import",
                    "source_sha256": source_digest,
                    "source_model": "Global MetNet (caller-attested; not independently verified)",
                }
            )
            selected.load()
            selected.to_netcdf(forecast_path, engine="h5netcdf")

        if (
            source_snapshot.stat().st_size != source_size
            or _sha256(source_snapshot) != source_digest
        ):
            raise NowcastImportError(
                "The private source snapshot changed while it was being imported; no run was installed."
            )
        source_snapshot.unlink()
        snapshot_dir.rmdir()
        snapshot_dir = None

        forecast_digest = _sha256(forecast_path)
        created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        status = global_metnet_status()
        primary: dict[str, Any] = {
            "variable": canonical_name,
            "units": units,
            "label": label,
        }
        if interval is not None:
            primary["interval"] = interval
        manifest = {
            "schema_id": "weathernext-local-run-manifest/v1",
            "status": "complete",
            "run_id": selected_run_id,
            "created_at": created_at,
            "primary_precipitation_variable": primary,
            "model": {
                "id": status.model_id,
                "name": status.display_name,
                "family": status.family,
                "implementation_status": status.implementation_status,
                "locally_executed": False,
                "paper_url": status.paper_url,
                "origin": {
                    "claim": "global-metnet",
                    "asserted_by": "caller",
                    "evidence": origin_evidence.strip(),
                    "verified_by_repository": False,
                },
            },
            "input": {
                "kind": "authorized_global_metnet_output",
                "source": {
                    "path": str(source_path),
                    "size_bytes": source_size,
                    "sha256": source_digest,
                    "identity_scope": "exact private snapshot decoded by importer",
                    "authorization_attested": True,
                    "global_metnet_origin_attested": True,
                },
            },
            "inference": {
                "execution_mode": "authorized_output_import",
                "performed_by_this_repository": False,
                "time_step_minutes": step_minutes,
            },
            "artifact": {
                "path": "forecast.nc",
                "format": "NetCDF4/HDF5",
                "size_bytes": forecast_path.stat().st_size,
                "sha256": forecast_digest,
                "precipitation_variable": canonical_name,
                "precipitation_units": units,
                "coordinate_units": {
                    "latitude": "degrees_north",
                    "longitude": "degrees_east",
                },
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_dir, final_dir)
    except BaseException as exc:
        cleanup_failures: list[str] = []
        for cleanup_path in (temp_dir, snapshot_dir):
            if cleanup_path is None or not cleanup_path.exists():
                continue
            try:
                shutil.rmtree(cleanup_path)
            except OSError as cleanup_exc:
                cleanup_failures.append(f"{cleanup_path}: {cleanup_exc}")
        if cleanup_failures:
            raise NowcastImportError(
                "Import failed and private temporary data could not be removed. "
                f"Remove it manually: {'; '.join(cleanup_failures)}. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        raise

    return ImportedNowcastRun(
        selected_run_id,
        final_dir,
        final_dir / "forecast.nc",
        final_dir / "manifest.json",
    )


__all__ = [
    "AuthorizationRequiredError",
    "ImportedNowcastRun",
    "NowcastImportError",
    "OriginAttestationRequiredError",
    "import_global_metnet_output",
]
