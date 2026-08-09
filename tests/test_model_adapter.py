from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from weathernext_local.assets import (
    SAMPLE_SPECS,
    WEIGHTS_SPEC,
    AssetPaths,
    AssetVerification,
)
from weathernext_local.doctor import DeviceInfo, RuntimeReport
from weathernext_local.model import adapter


def test_model_module_is_lazy_imported() -> None:
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    script = (
        "import sys; import weathernext_local.model; "
        "print(','.join(sorted({'jax','haiku','xarray','numpy','weathernext'} & set(sys.modules))))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.stdout.strip() == ""


def test_pinned_model_identity_and_dashboard_contract() -> None:
    assert adapter.MODEL_ID == "WeatherNextCyclones_Mini_<2024"
    assert adapter.UPSTREAM_VERSION == "v0.3.0"
    assert adapter.UPSTREAM_COMMIT == "89c4b2a77a1c57b328b909c575550fd2e5aadc9c"
    assert adapter.ATTENTION_TYPE == "triblockdiag_mha"
    assert adapter.DASHBOARD_VARIABLES == (
        "total_precipitation_6hr",
        "2m_temperature",
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    )


def test_dependency_loader_rejects_unpinned_upstream_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter.importlib, "import_module", lambda name: ModuleType(name))
    monkeypatch.setattr(adapter.metadata, "version", lambda name: "0.4.0")

    with pytest.raises(adapter.ModelDependenciesError, match=r"pinned to '0.3.0'"):
        adapter._load_dependencies()


def test_run_config_defaults_to_single_member_and_device() -> None:
    config = adapter.RunConfig()
    assert config.steps == 1
    assert config.ensemble_members == 1
    assert config.device_index == 0
    assert config.seed == 0
    assert config.download_assets


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"ensemble_members": 0}, "ensemble_members"),
        ({"ensemble_members": True}, "ensemble_members"),
        ({"device_index": -1}, "device_index"),
        ({"device_index": True}, "device_index"),
    ],
)
def test_run_config_rejects_invalid_execution_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(adapter.UnsupportedConfigurationError, match=message):
        adapter.RunConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "message",
    [
        "RESOURCE_EXHAUSTED while allocating",
        "CUDA_ERROR_OUT_OF_MEMORY",
        "Failed to allocate 123 bytes",
        "XlaRuntimeError: OOM",
    ],
)
def test_oom_detection_is_focused(message: str) -> None:
    assert adapter._is_oom_error(RuntimeError(message))
    assert not adapter._is_oom_error(RuntimeError("invalid tensor shape"))


def test_oom_translation_has_consumer_gpu_guidance() -> None:
    with pytest.raises(adapter.InferenceOutOfMemoryError, match="12 GB GPU"):
        adapter._raise_inference_error(RuntimeError("RESOURCE_EXHAUSTED: out of memory"))


def test_forward_configuration_matches_official_gpu_notebook() -> None:
    transformer_kwargs: dict[str, object] = {"attention_type": "splash_mha"}
    mesh_ctor = SimpleNamespace(keywords={"transformer_kwargs": transformer_kwargs})
    wrappers = [{"name": "normalization"}, {"name": "ensemble"}]
    upstream_config = SimpleNamespace(
        task="task",
        predictor_constructor="constructor",
        predictor_kwargs={"noisy_function_kwargs": {"mesh_model_ctor": mesh_ctor}},
        predictor_wrappers=wrappers,
    )
    observed: dict[str, object] = {}

    class FakeFgn:
        @staticmethod
        def PredictorConfig(**kwargs: object) -> SimpleNamespace:
            observed["config"] = kwargs
            return SimpleNamespace(**kwargs)

        @staticmethod
        def construct_predictor(config: object) -> object:
            observed["constructed"] = config

            def predictor(
                inputs: object, *, targets_template: object, forcings: object
            ) -> tuple[object, object, object]:
                return inputs, targets_template, forcings

            return predictor

    class FakeTransformed:
        def __init__(self, function: object):
            self.function = function

        def apply(
            self,
            params: object,
            rng: object,
            inputs: object,
            targets_template: object,
            forcings: object,
        ) -> object:
            observed["params"] = params
            observed["rng"] = rng
            return self.function(inputs, targets_template, forcings)  # type: ignore[operator]

    fake_haiku = SimpleNamespace(transform=lambda function: FakeTransformed(function))
    fake_jax = SimpleNamespace(jit=lambda function: function)
    deps = SimpleNamespace(fgn=FakeFgn, haiku=fake_haiku, jax=fake_jax)

    forward = adapter._configure_forward(
        deps,
        upstream_config,
        SimpleNamespace(params="official-params"),  # type: ignore[arg-type]
    )
    result = forward("rng", "inputs", "targets", "forcings")

    assert transformer_kwargs["attention_type"] == "triblockdiag_mha"
    assert observed["config"]["predictor_wrappers"] == wrappers[:-1]  # type: ignore[index]
    assert observed["params"] == "official-params"
    assert result == ("inputs", "targets", "forcings")


def _report() -> RuntimeReport:
    return RuntimeReport(
        platform="linux",
        platform_release="test",
        is_wsl=True,
        python_version="3.11",
        jax_version="0.test",
        jaxlib_version="0.test",
        backend="gpu",
        devices=(DeviceInfo(0, "gpu", "Test GPU", 0),),
        issues=(),
    )


def test_manifest_is_strict_round_trip(tmp_path: Path) -> None:
    forecast = tmp_path / adapter.FORECAST_FILENAME
    forecast.write_bytes(b"netcdf-placeholder")
    weights_check = AssetVerification(
        tmp_path / "weights.npz", WEIGHTS_SPEC.size_bytes, WEIGHTS_SPEC.md5_base64
    )
    sample_check = AssetVerification(
        tmp_path / "sample.nc", SAMPLE_SPECS[1].size_bytes, SAMPLE_SPECS[1].md5_base64
    )
    assets = AssetPaths(
        weights=weights_check.path,
        sample=sample_check.path,
        weights_verification=weights_check,
        sample_verification=sample_check,
    )
    config = adapter.RunConfig(steps=1, output_dir=tmp_path)
    manifest = adapter._build_manifest(
        report=_report(),
        assets=assets,
        config=config,
        selected_device=SimpleNamespace(__str__=lambda self: "Test GPU"),
        forecast_path=forecast,
        started_at="2026-08-09T12:00:00.000Z",
        completed_at="2026-08-09T12:00:01.000Z",
        runtime_seconds=1.0,
        initialization_time="2024-10-07T00:00:00.000000000",
        artifact_dimensions={"sample": 1, "time": 1, "lat": 181, "lon": 360},
    )
    manifest_path = tmp_path / adapter.MANIFEST_FILENAME
    adapter._write_manifest_atomic(manifest, manifest_path)

    loaded = adapter.load_manifest(manifest_path)

    assert loaded == manifest
    assert loaded["model"]["id"] == "WeatherNextCyclones_Mini_<2024"
    assert loaded["inference"]["gradients_enabled"] is False
    assert loaded["inference"]["attention_type"] == "triblockdiag_mha"
    assert loaded["artifact"]["precipitation_units"] == "m"
    assert loaded["artifact"]["sha256"] == hashlib_sha256(forecast.read_bytes())
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert not list(tmp_path.glob("*.tmp"))


def hashlib_sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def test_manifest_validation_rejects_missing_or_false_provenance(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(adapter.OutputContractError, match="schema_id"):
        adapter.load_manifest(path)

    bad = {"schema_id": adapter.MANIFEST_SCHEMA_ID, "status": "running"}
    with pytest.raises(adapter.OutputContractError, match="complete runs"):
        adapter.validate_manifest(bad)


def test_run_forecast_rejects_config_and_overrides() -> None:
    with pytest.raises(TypeError, match="either a RunConfig or keyword overrides"):
        adapter.run_forecast(adapter.RunConfig(), steps=4)


def test_select_device_rejects_unavailable_index() -> None:
    deps = SimpleNamespace(
        jax=SimpleNamespace(local_devices=lambda: [SimpleNamespace(platform="gpu")])
    )
    with pytest.raises(adapter.UnsupportedConfigurationError, match="reported 1 local GPU"):
        adapter._select_device(deps, _report(), 1)  # type: ignore[arg-type]


def test_select_device_rejects_cpu_even_after_report() -> None:
    cpu = SimpleNamespace(platform="cpu")
    deps = SimpleNamespace(jax=SimpleNamespace(local_devices=lambda: [cpu]))
    with pytest.raises(adapter.UnsupportedConfigurationError, match="CPU inference is disabled"):
        adapter._select_device(deps, _report(), 0)  # type: ignore[arg-type]
