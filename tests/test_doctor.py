from __future__ import annotations

from types import SimpleNamespace

import pytest

from weathernext_local import doctor


def test_native_windows_fails_before_importing_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.sys, "platform", "win32")
    monkeypatch.setattr(doctor.platform, "release", lambda: "Windows-11")

    def forbidden_import(name: str) -> object:
        raise AssertionError(f"must not import {name}")

    monkeypatch.setattr(doctor.importlib, "import_module", forbidden_import)
    report = doctor.inspect_environment()

    assert not report.ok
    assert report.backend is None
    assert report.errors[0].code == "native_windows_unsupported"
    with pytest.raises(doctor.NativeWindowsError, match="WSL2"):
        doctor.assert_runtime(report)


def test_linux_reports_missing_optional_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    monkeypatch.setattr(doctor.platform, "release", lambda: "6.8.0-microsoft-standard-WSL2")

    def missing(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(doctor.importlib, "import_module", missing)
    report = doctor.inspect_environment()

    assert report.is_wsl
    assert report.errors[0].code == "jax_missing"
    with pytest.raises(doctor.JaxUnavailableError, match=r"model.*extra"):
        report.raise_for_errors()


def test_cpu_jax_is_rejected_with_actionable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_jax = SimpleNamespace(
        __version__="0.test",
        default_backend=lambda: "cpu",
        local_devices=lambda: [
            SimpleNamespace(id=0, platform="cpu", device_kind="cpu", process_index=0)
        ],
    )
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    monkeypatch.setattr(doctor.platform, "release", lambda: "6.8.0")
    monkeypatch.setattr(doctor.importlib, "import_module", lambda name: fake_jax)
    monkeypatch.setattr(doctor.metadata, "version", lambda name: "0.test")

    report = doctor.inspect_environment()

    assert report.backend == "cpu"
    assert report.errors[0].code == "gpu_backend_required"
    with pytest.raises(
        doctor.GPUBackendRequiredError, match="CPU execution is intentionally disabled"
    ):
        doctor.assert_runtime(report)


def test_gpu_jax_report_is_valid_and_warns_for_multiple_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = [
        SimpleNamespace(id=index, platform="gpu", device_kind=f"GPU {index}", process_index=0)
        for index in range(2)
    ]
    fake_jax = SimpleNamespace(
        __version__="0.test", default_backend=lambda: "gpu", local_devices=lambda: devices
    )
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    monkeypatch.setattr(doctor.platform, "release", lambda: "6.8.0")
    monkeypatch.setattr(doctor.importlib, "import_module", lambda name: fake_jax)
    monkeypatch.setattr(doctor.metadata, "version", lambda name: "0.test-lib")

    report = doctor.inspect_environment()

    assert report.ok
    assert report.device_count == 2
    assert report.warnings[0].code == "single_device_default"
    assert report.to_dict()["devices"][0]["kind"] == "GPU 0"
    assert doctor.assert_runtime(report) is report


def test_backend_initialization_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_backend() -> str:
        raise RuntimeError("CUDA plugin exploded")

    fake_jax = SimpleNamespace(
        __version__="0.test", default_backend=fail_backend, local_devices=lambda: []
    )
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    monkeypatch.setattr(doctor.platform, "release", lambda: "6.8.0")
    monkeypatch.setattr(doctor.importlib, "import_module", lambda name: fake_jax)
    monkeypatch.setattr(doctor.metadata, "version", lambda name: "0.test-lib")

    report = doctor.inspect_environment()

    assert report.errors[0].code == "jax_backend_failed"
    with pytest.raises(doctor.GPUBackendRequiredError, match="CUDA plugin exploded"):
        report.raise_for_errors()
