"""Lazy, inference-only adapter for official WeatherNext 2 Mini v0.3.0.

The execution path mirrors ``docs/weathernext2/wn2_demo.ipynb`` at upstream
tag v0.3.0: the fixed HRES sample is split with the upstream task config, the
ensemble wrapper is removed, GPU attention is set to ``triblockdiag_mha``, and
autoregressive inference is emitted in one-step chunks.  No training or
gradient code is constructed by this module.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import tempfile
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

from weathernext_local.assets import (
    SAMPLE_SPECS,
    WEIGHTS_SPEC,
    AssetPaths,
    ensure_assets,
    normalize_steps,
)
from weathernext_local.doctor import RuntimeReport, assert_runtime
from weathernext_local.paths import runs_root

UPSTREAM_REPOSITORY = "https://github.com/google-deepmind/weathernext"
UPSTREAM_VERSION = "v0.3.0"
UPSTREAM_DISTRIBUTION_VERSION = "0.3.0"
UPSTREAM_COMMIT = "89c4b2a77a1c57b328b909c575550fd2e5aadc9c"
MODEL_NAME = "WeatherNextCyclones_Mini"
MODEL_SPLIT = "2024"
MODEL_ID = f"{MODEL_NAME}_<{MODEL_SPLIT}"
CONFIG_NAME = f"weathernext2/configs/{MODEL_NAME}"
ATTENTION_TYPE = "triblockdiag_mha"
MANIFEST_SCHEMA_ID = "weathernext-local-run-manifest/v1"
FORECAST_FILENAME = "forecast.nc"
MANIFEST_FILENAME = "manifest.json"
PRECIPITATION_VARIABLE = "total_precipitation_6hr"
DASHBOARD_VARIABLES = (
    PRECIPITATION_VARIABLE,
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
)


class ModelAdapterError(RuntimeError):
    """Base class for model adapter failures."""


class ModelDependenciesError(ModelAdapterError):
    """Raised when the pinned upstream execution dependencies are unavailable."""


class UnsupportedConfigurationError(ModelAdapterError, ValueError):
    """Raised when a run requests an unvalidated execution mode."""


class SampleContractError(ModelAdapterError):
    """Raised when an official sample does not have the expected demo schema."""


class OutputContractError(ModelAdapterError):
    """Raised when predictions cannot satisfy the canonical dashboard schema."""


class InferenceOutOfMemoryError(ModelAdapterError, MemoryError):
    """Raised with focused guidance when XLA reports device memory exhaustion."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Configuration for an inference-only local Mini run.

    Multiple members are supported sequentially on the selected device, but a
    single member and one device are the conservative defaults for consumer
    GPU memory.  ``steps`` is restricted to samples released with the demo.
    """

    steps: int = 1
    output_dir: Path | None = None
    cache_dir: Path | None = None
    weights_path: Path | None = None
    sample_path: Path | None = None
    seed: int = 0
    ensemble_members: int = 1
    device_index: int = 0
    download_assets: bool = True
    repair_assets: bool = True
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", normalize_steps(self.steps))
        for name in ("output_dir", "cache_dir", "weights_path", "sample_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser())
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise UnsupportedConfigurationError("seed must be a non-negative integer.")
        if (
            isinstance(self.ensemble_members, bool)
            or not isinstance(self.ensemble_members, int)
            or self.ensemble_members < 1
        ):
            raise UnsupportedConfigurationError("ensemble_members must be a positive integer.")
        if (
            isinstance(self.device_index, bool)
            or not isinstance(self.device_index, int)
            or self.device_index < 0
        ):
            raise UnsupportedConfigurationError("device_index must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class RunResult:
    run_dir: Path
    forecast_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Dependencies:
    haiku: ModuleType
    jax: ModuleType
    numpy: ModuleType
    xarray: ModuleType
    xarray_jax: ModuleType
    rollout: ModuleType
    checkpoint: ModuleType
    data_utils: ModuleType
    fiddle_config_io: ModuleType
    fgn: ModuleType


def _load_dependencies() -> _Dependencies:
    modules = {
        "haiku": "haiku",
        "jax": "jax",
        "numpy": "numpy",
        "xarray": "xarray",
        "xarray_jax": "xarray_jax",
        "rollout": "weathernext.utils.rollout",
        "checkpoint": "weathernext.utils.checkpoint",
        "data_utils": "weathernext.utils.data_utils",
        "fiddle_config_io": "weathernext.utils.fiddle_config_io",
        "fgn": "weathernext.weathernext2.fgn",
    }
    imported: dict[str, ModuleType] = {}
    try:
        for key, module_name in modules.items():
            imported[key] = importlib.import_module(module_name)
    except Exception as exc:
        raise ModelDependenciesError(
            "WeatherNext Mini model dependencies are incomplete. Install the project's "
            "'model' extra inside Linux/WSL2; the adapter requires the pinned upstream "
            f"WeatherNext {UPSTREAM_VERSION} package. Failed import: {exc}."
        ) from exc
    try:
        installed_version = metadata.version("weathernext")
    except metadata.PackageNotFoundError as exc:
        raise ModelDependenciesError(
            "The upstream 'weathernext' distribution metadata is missing. Install the "
            f"official {UPSTREAM_VERSION} model package inside Linux/WSL2."
        ) from exc
    if installed_version != UPSTREAM_DISTRIBUTION_VERSION:
        raise ModelDependenciesError(
            f"Unsupported upstream weathernext package {installed_version!r}; this adapter is "
            f"pinned to {UPSTREAM_DISTRIBUTION_VERSION!r} ({UPSTREAM_VERSION})."
        )
    return _Dependencies(**imported)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _default_run_dir() -> Path:
    token = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return runs_root() / f"mini-{token}-{uuid.uuid4().hex[:8]}"


def _is_oom_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    indicators = (
        "resource_exhausted",
        "out of memory",
        "out-of-memory",
        "cuda_error_out_of_memory",
        "failed to allocate",
        "memory allocation failed",
        "oom",
    )
    return any(indicator in text for indicator in indicators)


def _raise_inference_error(exc: BaseException) -> None:
    if _is_oom_error(exc):
        raise InferenceOutOfMemoryError(
            "WeatherNext Mini exhausted accelerator memory during inference or XLA compilation. "
            "The adapter already uses triblockdiag_mha, one device, and one-step chunks. "
            "Retry with one ensemble member, close other GPU workloads, and begin with the "
            "1-step sample. A 12 GB GPU is an experimental configuration and may not fit; "
            "use the bundled/public forecast artifacts if it does not."
        ) from exc
    raise exc


def _select_device(deps: _Dependencies, report: RuntimeReport, device_index: int) -> Any:
    devices = list(deps.jax.local_devices())
    if device_index >= len(devices):
        raise UnsupportedConfigurationError(
            f"device_index {device_index} is unavailable; JAX reported {len(devices)} local "
            f"GPU device(s): {[device.kind for device in report.devices]}."
        )
    selected = devices[device_index]
    if str(getattr(selected, "platform", "")) != "gpu":
        raise UnsupportedConfigurationError(
            f"Selected device {device_index} is not a GPU: {selected!s}. CPU inference is disabled."
        )
    return selected


def _load_official_inputs(
    deps: _Dependencies, sample_path: Path, upstream_config: Any, expected_steps: int
) -> tuple[Any, Any, Any, Any]:
    try:
        example_batch = deps.xarray.load_dataset(sample_path, engine="h5netcdf").compute()
    except Exception as exc:
        raise SampleContractError(
            f"The verified sample NetCDF could not be decoded by xarray: {sample_path}: {exc}"
        ) from exc

    if "time" not in example_batch.sizes:
        raise SampleContractError("The official sample has no 'time' dimension.")
    available_steps = int(example_batch.sizes["time"]) - 2
    if available_steps != expected_steps:
        raise SampleContractError(
            f"Sample time layout provides {available_steps} forecast steps after its two input "
            f"frames, but this run requested {expected_steps}."
        )
    try:
        inputs, targets, forcings = deps.data_utils.extract_inputs_targets_forcings(
            example_batch,
            target_lead_times=slice("6h", f"{available_steps * 6}h"),
            **dataclasses.asdict(upstream_config.task),
        )
    except Exception as exc:
        raise SampleContractError(
            "The fixed HRES sample did not satisfy the pinned Mini task schema. "
            f"Confirm that both the sample and WeatherNext package are from {UPSTREAM_VERSION}: {exc}"
        ) from exc
    if int(targets.sizes.get("time", -1)) != expected_steps:
        raise SampleContractError(
            f"Upstream extraction returned {targets.sizes.get('time')} target times, "
            f"expected {expected_steps}."
        )
    return example_batch, inputs, targets, forcings


def _configure_forward(deps: _Dependencies, upstream_config: Any, loaded_checkpoint: Any) -> Any:
    try:
        transformer_kwargs = upstream_config.predictor_kwargs["noisy_function_kwargs"][
            "mesh_model_ctor"
        ].keywords["transformer_kwargs"]
        transformer_kwargs["attention_type"] = ATTENTION_TYPE
        wrappers = upstream_config.predictor_wrappers
        if not wrappers:
            raise ValueError("predictor_wrappers is empty")
        inference_config = deps.fgn.PredictorConfig(
            task=upstream_config.task,
            predictor_constructor=upstream_config.predictor_constructor,
            predictor_kwargs=upstream_config.predictor_kwargs,
            predictor_wrappers=wrappers[:-1],  # Official notebook removes ensemble wrapper.
        )
    except Exception as exc:
        raise ModelDependenciesError(
            f"WeatherNext {UPSTREAM_VERSION} Mini config did not match the official demo structure: {exc}"
        ) from exc

    def run_forward(inputs: Any, targets_template: Any, forcings: Any) -> Any:
        predictor = deps.fgn.construct_predictor(inference_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    transformed = deps.haiku.transform(run_forward)

    def apply(rng: Any, inputs: Any, targets_template: Any, forcings: Any) -> Any:
        return transformed.apply(loaded_checkpoint.params, rng, inputs, targets_template, forcings)

    return deps.jax.jit(apply)


def _run_rollout(
    deps: _Dependencies,
    predictor_jit: Any,
    selected_device: Any,
    inputs: Any,
    targets: Any,
    forcings: Any,
    *,
    seed: int,
    ensemble_members: int,
) -> Any:
    predictor_pmap = deps.xarray_jax.pmap(predictor_jit, dim="sample")
    base_rng = deps.jax.random.PRNGKey(seed)
    rngs = deps.numpy.stack(
        [deps.jax.random.fold_in(base_rng, index) for index in range(ensemble_members)], axis=0
    )
    chunks: list[Any] = []
    generator = deps.rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=predictor_pmap,
        rngs=rngs,
        inputs=inputs,
        targets_template=targets * deps.numpy.nan,
        forcings=forcings,
        num_steps_per_chunk=1,
        num_samples=ensemble_members,
        pmap_devices=[selected_device],
    )
    try:
        for chunk in generator:
            missing = [name for name in DASHBOARD_VARIABLES if name not in chunk.data_vars]
            if missing:
                raise OutputContractError(
                    f"WeatherNext predictions are missing dashboard variables: {', '.join(missing)}."
                )
            # Move only the canonical output fields to host memory immediately;
            # retaining all model fields on device across chunks can cause OOM.
            chunks.append(deps.jax.device_get(chunk[list(DASHBOARD_VARIABLES)]))
    except Exception as exc:
        _raise_inference_error(exc)
    if not chunks:
        raise OutputContractError("WeatherNext produced no forecast chunks.")
    try:
        return deps.xarray.combine_by_coords(chunks)
    except Exception as exc:
        raise OutputContractError(
            f"Could not combine forecast chunks by sample/time coordinates: {exc}"
        ) from exc


def _initialization_time(example_batch: Any) -> Any:
    if "datetime" not in example_batch.coords:
        raise SampleContractError(
            "The official sample has no 'datetime' coordinate for provenance."
        )
    init = example_batch["datetime"]
    if "batch" in init.dims:
        if init.sizes["batch"] != 1:
            raise SampleContractError("The official demo adapter requires a singleton input batch.")
        init = init.isel(batch=0, drop=True)
    if "time" not in init.dims or init.sizes["time"] < 2:
        raise SampleContractError(
            "The sample datetime coordinate lacks the two required input frames."
        )
    return init.isel(time=1, drop=True).values


def _canonicalize_predictions(
    predictions: Any, example_batch: Any, expected_steps: int
) -> tuple[Any, str]:
    output = predictions[list(DASHBOARD_VARIABLES)].copy()
    if "batch" in output.dims:
        if output.sizes["batch"] != 1:
            raise OutputContractError(
                f"Canonical output requires one batch, found {output.sizes['batch']}."
            )
        output = output.isel(batch=0, drop=True)
    if "sample" not in output.dims:
        output = output.expand_dims(sample=[0])

    init_value = _initialization_time(example_batch)
    output = output.assign_coords(
        forecast_reference_time=init_value,
        init_time=init_value,
    )
    if "datetime" in output.coords and output["datetime"].dims == ("time",):
        output = output.assign_coords(valid_time=("time", output["datetime"].values))
    elif "time" in output.coords:
        try:
            valid_values = init_value + output["time"].values
        except Exception as exc:
            raise OutputContractError(
                "Forecast time is not a valid timedelta coordinate and no 1-D datetime coordinate exists."
            ) from exc
        output = output.assign_coords(valid_time=("time", valid_values))

    precip = output[PRECIPITATION_VARIABLE]
    original_units = str(precip.attrs.get("units", "m")).strip().casefold()
    metre_units = {"m", "meter", "meters", "metre", "metres"}
    if original_units not in metre_units:
        raise OutputContractError(
            f"{PRECIPITATION_VARIABLE} has units {precip.attrs.get('units')!r}; "
            "the canonical artifact must preserve the upstream values in metres."
        )
    output[PRECIPITATION_VARIABLE].attrs = {
        **precip.attrs,
        "units": "m",
        "accumulation_period": "6 hours",
    }
    output.attrs = {
        **output.attrs,
        "schema_id": "weathernext-local-forecast/v1",
        "model_id": MODEL_ID,
        "upstream_release": UPSTREAM_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "attention_type": ATTENTION_TYPE,
        "inference_only": "true",
        "precipitation_values_preserved_in_metres": "true",
    }
    _validate_canonical_dataset(output, expected_steps)
    return output, str(init_value)


def _validate_canonical_dataset(dataset: Any, expected_steps: int) -> None:
    missing = [name for name in DASHBOARD_VARIABLES if name not in dataset.data_vars]
    if missing:
        raise OutputContractError(f"Canonical forecast is missing variables: {', '.join(missing)}.")
    if "time" not in dataset.dims or int(dataset.sizes["time"]) != expected_steps:
        raise OutputContractError(
            f"Canonical forecast must contain {expected_steps} time steps; sizes={dict(dataset.sizes)}."
        )
    if "sample" not in dataset.dims or int(dataset.sizes["sample"]) < 1:
        raise OutputContractError("Canonical forecast must retain a non-empty 'sample' dimension.")
    for coordinate in ("lat", "lon"):
        if coordinate not in dataset.coords or dataset[coordinate].ndim != 1:
            raise OutputContractError(
                f"Canonical forecast requires an explicit 1-D {coordinate!r} coordinate."
            )
    if "valid_time" not in dataset.coords or dataset["valid_time"].dims != ("time",):
        raise OutputContractError("Canonical forecast requires a 1-D 'valid_time' coordinate.")
    for coordinate in ("forecast_reference_time", "init_time"):
        if coordinate not in dataset.coords or dataset[coordinate].ndim != 0:
            raise OutputContractError(f"Canonical forecast requires scalar {coordinate!r}.")
    if dataset[PRECIPITATION_VARIABLE].attrs.get("units") != "m":
        raise OutputContractError(f"{PRECIPITATION_VARIABLE} must have canonical units 'm'.")


def _write_netcdf_atomic(
    dataset: Any, destination: Path, expected_steps: int, xarray_module: ModuleType
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp.nc", dir=destination.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        encoding = {
            variable: {"zlib": True, "complevel": 4, "shuffle": True}
            for variable in dataset.data_vars
        }
        dataset.to_netcdf(temp_path, engine="h5netcdf", encoding=encoding)
        with xarray_module.open_dataset(temp_path, engine="h5netcdf") as reopened:
            _validate_canonical_dataset(reopened, expected_steps)
        with temp_path.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temp_path, destination)
    except Exception:
        with suppress(OSError):
            temp_path.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _build_manifest(
    *,
    report: RuntimeReport,
    assets: AssetPaths,
    config: RunConfig,
    selected_device: Any,
    forecast_path: Path,
    started_at: str,
    completed_at: str,
    runtime_seconds: float,
    initialization_time: str,
    artifact_dimensions: dict[str, int],
) -> dict[str, Any]:
    sample = SAMPLE_SPECS[config.steps]
    manifest: dict[str, Any] = {
        "schema_id": MANIFEST_SCHEMA_ID,
        "status": "complete",
        "run_id": forecast_path.parent.name,
        "created_at": completed_at,
        "model": {
            "id": MODEL_ID,
            "name": MODEL_NAME,
            "family": "medium-range",
            "split": f"<{MODEL_SPLIT}",
            "config_name": CONFIG_NAME,
            "resolution_degrees": 1.0,
            "upstream": {
                "repository": UPSTREAM_REPOSITORY,
                "release": UPSTREAM_VERSION,
                "commit": UPSTREAM_COMMIT,
                "distribution_version": UPSTREAM_DISTRIBUTION_VERSION,
            },
            "weights_license": "CC BY 4.0",
            "locally_executed": True,
            "weights": assets.weights_verification.to_manifest(WEIGHTS_SPEC),
        },
        "input": {
            "kind": "official_fixed_hres_forecast_sample",
            "initialization_time": initialization_time,
            "sample": assets.sample_verification.to_manifest(sample),
        },
        "inference": {
            "steps": config.steps,
            "forecast_horizon_hours": config.steps * 6,
            "step_hours": 6,
            "seed": config.seed,
            "ensemble_members": config.ensemble_members,
            "member_indices": list(range(config.ensemble_members)),
            "device_index": config.device_index,
            "device": str(selected_device),
            "backend": "gpu",
            "attention_type": ATTENTION_TYPE,
            "steps_per_chunk": 1,
            "gradients_enabled": False,
        },
        "environment": report.to_dict(),
        "artifact": {
            "path": forecast_path.name,
            "format": "NetCDF4/HDF5",
            "size_bytes": forecast_path.stat().st_size,
            "sha256": _sha256(forecast_path),
            "variables": list(DASHBOARD_VARIABLES),
            "dimensions": artifact_dimensions,
            "precipitation_variable": PRECIPITATION_VARIABLE,
            "precipitation_units": "m",
        },
        "primary_precipitation_variable": {
            "variable": PRECIPITATION_VARIABLE,
            "units": "m",
            "interval": "6 h",
            "label": "6-hour accumulated precipitation",
        },
        "timing": {
            "started_at": started_at,
            "completed_at": completed_at,
            "runtime_seconds": round(runtime_seconds, 6),
        },
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate required provenance fields; raise rather than guess defaults."""

    def require(mapping: Any, key: str, path: str) -> Any:
        if not isinstance(mapping, dict) or key not in mapping:
            raise OutputContractError(f"Manifest is missing required field {path}.{key}.")
        return mapping[key]

    if require(manifest, "schema_id", "$manifest") != MANIFEST_SCHEMA_ID:
        raise OutputContractError("Manifest schema_id is not the supported v1 contract.")
    if require(manifest, "status", "$manifest") != "complete":
        raise OutputContractError("Only complete runs may be published as canonical artifacts.")
    run_id = require(manifest, "run_id", "$manifest")
    created_at = require(manifest, "created_at", "$manifest")
    if not isinstance(run_id, str) or not run_id.strip():
        raise OutputContractError("Manifest run_id must be non-empty text.")
    if not isinstance(created_at, str) or not created_at.endswith("Z"):
        raise OutputContractError(
            "Manifest created_at must be an explicit UTC timestamp ending in 'Z'."
        )
    model = require(manifest, "model", "$manifest")
    if require(model, "id", "$manifest.model") != MODEL_ID:
        raise OutputContractError("Manifest model.id does not identify the pinned Mini checkpoint.")
    if require(model, "family", "$manifest.model") != "medium-range":
        raise OutputContractError("Manifest model.family must be 'medium-range'.")
    upstream = require(model, "upstream", "$manifest.model")
    expected_upstream = {
        "repository": UPSTREAM_REPOSITORY,
        "release": UPSTREAM_VERSION,
        "commit": UPSTREAM_COMMIT,
        "distribution_version": UPSTREAM_DISTRIBUTION_VERSION,
    }
    if upstream != expected_upstream:
        raise OutputContractError(
            "Manifest upstream identity does not match the pinned v0.3.0 release."
        )
    if require(model, "locally_executed", "$manifest.model") is not True:
        raise OutputContractError("Manifest model.locally_executed must be true for adapter runs.")
    weights = require(model, "weights", "$manifest.model")
    input_section = require(manifest, "input", "$manifest")
    if require(input_section, "kind", "$manifest.input") != "official_fixed_hres_forecast_sample":
        raise OutputContractError(
            "Manifest input.kind does not identify the official fixed HRES sample."
        )
    initialization_time = require(input_section, "initialization_time", "$manifest.input")
    if not isinstance(initialization_time, str) or not initialization_time:
        raise OutputContractError("Manifest input.initialization_time must be non-empty text.")
    sample = require(input_section, "sample", "$manifest.input")
    for name, provenance in (("weights", weights), ("sample", sample)):
        for key in ("path", "source_uri", "size_bytes", "md5"):
            require(provenance, key, f"$manifest.{name}")
        md5 = require(provenance, "md5", f"$manifest.{name}")
        if require(md5, "algorithm", f"$manifest.{name}.md5") != "md5":
            raise OutputContractError(f"Manifest {name} digest algorithm must be 'md5'.")
        require(md5, "base64", f"$manifest.{name}.md5")
    inference = require(manifest, "inference", "$manifest")
    steps = require(inference, "steps", "$manifest.inference")
    try:
        steps = normalize_steps(steps)
    except ValueError as exc:
        raise OutputContractError(f"Manifest inference.steps is unsupported: {steps!r}.") from exc
    selected_sample = SAMPLE_SPECS[steps]
    expected_weight_identity = (
        WEIGHTS_SPEC.gcs_uri,
        WEIGHTS_SPEC.size_bytes,
        WEIGHTS_SPEC.md5_base64,
    )
    actual_weight_identity = (
        weights["source_uri"],
        weights["size_bytes"],
        weights["md5"]["base64"],
    )
    if actual_weight_identity != expected_weight_identity:
        raise OutputContractError(
            "Manifest weights identity does not match the official <2024 object."
        )
    expected_sample_identity = (
        selected_sample.gcs_uri,
        selected_sample.size_bytes,
        selected_sample.md5_base64,
    )
    actual_sample_identity = (
        sample["source_uri"],
        sample["size_bytes"],
        sample["md5"]["base64"],
    )
    if actual_sample_identity != expected_sample_identity:
        raise OutputContractError(
            f"Manifest sample identity does not match the official {steps}-step object."
        )
    if require(inference, "backend", "$manifest.inference") != "gpu":
        raise OutputContractError("Manifest inference.backend must be 'gpu'.")
    if require(inference, "attention_type", "$manifest.inference") != ATTENTION_TYPE:
        raise OutputContractError(f"Manifest must record attention_type={ATTENTION_TYPE!r}.")
    if require(inference, "gradients_enabled", "$manifest.inference") is not False:
        raise OutputContractError("Manifest must record gradients_enabled=false.")
    ensemble_members = require(inference, "ensemble_members", "$manifest.inference")
    if (
        not isinstance(ensemble_members, int)
        or isinstance(ensemble_members, bool)
        or ensemble_members < 1
    ):
        raise OutputContractError("Manifest ensemble_members must be a positive integer.")
    if require(inference, "member_indices", "$manifest.inference") != list(range(ensemble_members)):
        raise OutputContractError("Manifest member_indices must enumerate every produced member.")
    artifact = require(manifest, "artifact", "$manifest")
    for key in (
        "path",
        "size_bytes",
        "sha256",
        "variables",
        "dimensions",
        "precipitation_units",
    ):
        require(artifact, key, "$manifest.artifact")
    if artifact["path"] != FORECAST_FILENAME:
        raise OutputContractError(f"Manifest artifact.path must be {FORECAST_FILENAME!r}.")
    if not isinstance(artifact["size_bytes"], int) or artifact["size_bytes"] <= 0:
        raise OutputContractError("Manifest artifact.size_bytes must be a positive integer.")
    dimensions = artifact["dimensions"]
    if not isinstance(dimensions, dict):
        raise OutputContractError("Manifest artifact.dimensions must be an object.")
    if dimensions.get("time") != steps or dimensions.get("sample") != ensemble_members:
        raise OutputContractError(
            "Manifest artifact dimensions disagree with the inference step/member counts."
        )
    if artifact["precipitation_units"] != "m":
        raise OutputContractError("Manifest precipitation_units must be 'm'.")
    if artifact["variables"] != list(DASHBOARD_VARIABLES):
        raise OutputContractError(
            "Manifest variables do not match the canonical dashboard selection."
        )
    primary_precipitation = require(manifest, "primary_precipitation_variable", "$manifest")
    expected_precipitation = {
        "variable": PRECIPITATION_VARIABLE,
        "units": "m",
        "interval": "6 h",
        "label": "6-hour accumulated precipitation",
    }
    if primary_precipitation != expected_precipitation:
        raise OutputContractError(
            "Manifest primary_precipitation_variable does not match the canonical "
            "6-hour metre-valued field."
        )
    sha256 = artifact["sha256"]
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise OutputContractError("Manifest artifact.sha256 is not a 64-character hex digest.")
    try:
        int(sha256, 16)
    except ValueError as exc:
        raise OutputContractError("Manifest artifact.sha256 is not hexadecimal.") from exc
    environment = require(manifest, "environment", "$manifest")
    if require(environment, "ok", "$manifest.environment") is not True:
        raise OutputContractError(
            "Manifest environment.ok must be true for a complete adapter run."
        )
    timing = require(manifest, "timing", "$manifest")
    for key in ("started_at", "completed_at", "runtime_seconds"):
        require(timing, key, "$manifest.timing")


def _write_manifest_atomic(manifest: dict[str, Any], destination: Path) -> None:
    validate_manifest(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, destination)
    except Exception:
        with suppress(OSError):
            temp_path.unlink(missing_ok=True)
        raise


def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate a run manifest without importing model dependencies."""

    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutputContractError(
            f"Could not read provenance manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise OutputContractError(f"Provenance manifest {manifest_path} is not a JSON object.")
    validate_manifest(value)
    return value


class WeatherNextMiniAdapter:
    """Execute one official Mini sample rollout and materialize its artifact."""

    def __init__(self, config: RunConfig | None = None):
        self.config = RunConfig() if config is None else config
        if not isinstance(self.config, RunConfig):
            raise TypeError("config must be a RunConfig instance.")

    def run(self) -> RunResult:
        # These are effective only before JAX initializes. They make the
        # programmatic API as conservative as the CLI on consumer GPUs.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        report = assert_runtime()
        run_dir = self.config.output_dir or _default_run_dir()
        if not run_dir.name:
            raise UnsupportedConfigurationError(
                "output_dir must identify a dedicated run directory, not a filesystem root."
            )
        if run_dir.exists() and not run_dir.is_dir():
            raise FileExistsError(f"Run output path exists and is not a directory: {run_dir}")
        if not self.config.overwrite and run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Run output directory is not empty: {run_dir}. Choose a new directory or "
                "set overwrite=True explicitly."
            )
        deps = _load_dependencies()
        selected_device = _select_device(deps, report, self.config.device_index)
        assets = ensure_assets(
            self.config.cache_dir,
            self.config.steps,
            download=self.config.download_assets,
            weights_path=self.config.weights_path,
            sample_path=self.config.sample_path,
            repair=self.config.repair_assets,
        )
        forecast_path = run_dir / FORECAST_FILENAME
        manifest_path = run_dir / MANIFEST_FILENAME
        run_dir.mkdir(parents=True, exist_ok=True)

        started_at = _utc_now()
        started_clock = time.perf_counter()
        upstream_config = deps.fiddle_config_io.get_fiddle_config_by_name(CONFIG_NAME)
        with assets.weights.open("rb") as stream:
            loaded_checkpoint = deps.checkpoint.load(stream, deps.fgn.CheckPoint)
        example_batch, inputs, targets, forcings = _load_official_inputs(
            deps, assets.sample, upstream_config, self.config.steps
        )
        predictor_jit = _configure_forward(deps, upstream_config, loaded_checkpoint)
        predictions = _run_rollout(
            deps,
            predictor_jit,
            selected_device,
            inputs,
            targets,
            forcings,
            seed=self.config.seed,
            ensemble_members=self.config.ensemble_members,
        )
        canonical, initialization_time = _canonicalize_predictions(
            predictions, example_batch, self.config.steps
        )
        _write_netcdf_atomic(canonical, forecast_path, self.config.steps, deps.xarray)

        completed_at = _utc_now()
        runtime_seconds = time.perf_counter() - started_clock
        manifest = _build_manifest(
            report=report,
            assets=assets,
            config=self.config,
            selected_device=selected_device,
            forecast_path=forecast_path,
            started_at=started_at,
            completed_at=completed_at,
            runtime_seconds=runtime_seconds,
            initialization_time=initialization_time,
            artifact_dimensions={name: int(size) for name, size in canonical.sizes.items()},
        )
        _write_manifest_atomic(manifest, manifest_path)
        return RunResult(run_dir, forecast_path, manifest_path, manifest)


def run_forecast(config: RunConfig | None = None, **overrides: Any) -> RunResult:
    """Convenience entry point used by the CLI and programmatic callers."""

    if config is not None and overrides:
        raise TypeError("Pass either a RunConfig or keyword overrides, not both.")
    selected = RunConfig(**overrides) if config is None else config
    return WeatherNextMiniAdapter(selected).run()


__all__ = [
    "ATTENTION_TYPE",
    "CONFIG_NAME",
    "DASHBOARD_VARIABLES",
    "FORECAST_FILENAME",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_ID",
    "MODEL_ID",
    "MODEL_NAME",
    "PRECIPITATION_VARIABLE",
    "UPSTREAM_COMMIT",
    "UPSTREAM_DISTRIBUTION_VERSION",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_VERSION",
    "InferenceOutOfMemoryError",
    "ModelAdapterError",
    "ModelDependenciesError",
    "OutputContractError",
    "RunConfig",
    "RunResult",
    "SampleContractError",
    "UnsupportedConfigurationError",
    "WeatherNextMiniAdapter",
    "load_manifest",
    "run_forecast",
    "validate_manifest",
]
