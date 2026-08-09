"""Fail-closed checks for the supported WeatherNext Mini runtime.

JAX is intentionally imported only inside :func:`inspect_environment`.  This
keeps downloads, the dashboard, and ordinary package imports usable on native
Windows while model execution remains Linux/WSL + NVIDIA GPU only.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any, Literal


class RuntimeEnvironmentError(RuntimeError):
    """Base class for unsupported or incomplete inference environments."""


class NativeWindowsError(RuntimeEnvironmentError):
    """Raised when model execution is attempted in native Windows Python."""


class UnsupportedPlatformError(RuntimeEnvironmentError):
    """Raised when model execution is attempted outside Linux/WSL."""


class JaxUnavailableError(RuntimeEnvironmentError):
    """Raised when the optional model/JAX dependencies are unavailable."""


class GPUBackendRequiredError(RuntimeEnvironmentError):
    """Raised when JAX is installed without a usable GPU backend."""


@dataclass(frozen=True, slots=True)
class RuntimeIssue:
    severity: Literal["error", "warning"]
    code: str
    message: str
    remediation: str


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    id: int | None
    platform: str
    kind: str
    process_index: int | None


@dataclass(frozen=True, slots=True)
class RuntimeReport:
    platform: str
    platform_release: str
    is_wsl: bool
    python_version: str
    jax_version: str | None
    jaxlib_version: str | None
    backend: str | None
    devices: tuple[DeviceInfo, ...]
    issues: tuple[RuntimeIssue, ...]

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def device_count(self) -> int:
        return len(self.devices)

    @property
    def errors(self) -> tuple[RuntimeIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[RuntimeIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "platform": self.platform,
            "platform_release": self.platform_release,
            "is_wsl": self.is_wsl,
            "python_version": self.python_version,
            "jax_version": self.jax_version,
            "jaxlib_version": self.jaxlib_version,
            "backend": self.backend,
            "device_count": self.device_count,
            "devices": [asdict(device) for device in self.devices],
            "issues": [asdict(issue) for issue in self.issues],
        }

    def raise_for_errors(self) -> None:
        if not self.errors:
            return
        detail = "\n".join(
            f"- [{issue.code}] {issue.message} Remedy: {issue.remediation}" for issue in self.errors
        )
        message = f"WeatherNext Mini runtime checks failed:\n{detail}"
        first_code = self.errors[0].code
        exception_type: type[RuntimeEnvironmentError]
        if first_code == "native_windows_unsupported":
            exception_type = NativeWindowsError
        elif first_code == "unsupported_platform":
            exception_type = UnsupportedPlatformError
        elif first_code in {"jax_missing", "jax_import_failed"}:
            exception_type = JaxUnavailableError
        else:
            exception_type = GPUBackendRequiredError
        raise exception_type(message)


def _is_wsl(release: str) -> bool:
    lowered = release.casefold()
    return "microsoft" in lowered or "wsl" in lowered or bool(os.environ.get("WSL_INTEROP"))


def _device_info(device: Any) -> DeviceInfo:
    process_index = getattr(device, "process_index", None)
    if callable(process_index):
        process_index = process_index()
    return DeviceInfo(
        id=getattr(device, "id", None),
        platform=str(getattr(device, "platform", "unknown")),
        kind=str(getattr(device, "device_kind", type(device).__name__)),
        process_index=process_index,
    )


def inspect_environment() -> RuntimeReport:
    """Inspect the local runtime without throwing on unsupported machines.

    Native Windows is reported before attempting to import JAX.  That ordering
    gives users the actionable WSL/Linux requirement even when the Windows
    environment intentionally lacks the optional model dependencies.
    """

    platform_name = sys.platform
    release = platform.release()
    wsl = platform_name.startswith("linux") and _is_wsl(release)
    issues: list[RuntimeIssue] = []

    if platform_name == "win32":
        issues.append(
            RuntimeIssue(
                "error",
                "native_windows_unsupported",
                "JAX NVIDIA GPU execution is not supported by this adapter in native Windows Python.",
                "Open the repository in WSL2, install the Linux model dependencies there, and run inference inside WSL2.",
            )
        )
        return RuntimeReport(
            platform_name,
            release,
            False,
            platform.python_version(),
            None,
            None,
            None,
            (),
            tuple(issues),
        )

    if not platform_name.startswith("linux"):
        issues.append(
            RuntimeIssue(
                "error",
                "unsupported_platform",
                f"Model execution is supported only on Linux or WSL2, not {platform_name!r}.",
                "Use a Linux host or WSL2 with an NVIDIA driver exposed to the guest.",
            )
        )
        return RuntimeReport(
            platform_name,
            release,
            False,
            platform.python_version(),
            None,
            None,
            None,
            (),
            tuple(issues),
        )

    try:
        jax = importlib.import_module("jax")
    except ModuleNotFoundError:
        issues.append(
            RuntimeIssue(
                "error",
                "jax_missing",
                "The optional JAX/WeatherNext model environment is not installed.",
                "Install this project with its 'model' extra inside Linux/WSL2.",
            )
        )
        return RuntimeReport(
            platform_name,
            release,
            wsl,
            platform.python_version(),
            None,
            None,
            None,
            (),
            tuple(issues),
        )
    except Exception as exc:  # A broken CUDA plugin can fail during JAX import.
        issues.append(
            RuntimeIssue(
                "error",
                "jax_import_failed",
                f"JAX could not be imported: {type(exc).__name__}: {exc}",
                "Repair the Linux JAX CUDA installation and verify that the NVIDIA driver is visible in WSL/Linux.",
            )
        )
        return RuntimeReport(
            platform_name,
            release,
            wsl,
            platform.python_version(),
            None,
            None,
            None,
            (),
            tuple(issues),
        )

    jax_version = str(getattr(jax, "__version__", "unknown"))
    try:
        jaxlib_version = metadata.version("jaxlib")
    except metadata.PackageNotFoundError:
        jaxlib_version = None

    backend: str | None = None
    devices: tuple[DeviceInfo, ...] = ()
    try:
        backend = str(jax.default_backend())
        devices = tuple(_device_info(device) for device in jax.local_devices())
    except Exception as exc:
        issues.append(
            RuntimeIssue(
                "error",
                "jax_backend_failed",
                f"JAX could not initialize its accelerator backend: {type(exc).__name__}: {exc}",
                "Check the NVIDIA driver, WSL GPU passthrough, and the installed JAX CUDA wheel.",
            )
        )

    if backend is not None and backend != "gpu":
        issues.append(
            RuntimeIssue(
                "error",
                "gpu_backend_required",
                f"JAX selected backend {backend!r}; WeatherNext Mini inference requires the NVIDIA GPU backend.",
                "Install the Linux JAX CUDA wheel and confirm jax.default_backend() returns 'gpu'. CPU execution is intentionally disabled.",
            )
        )
    elif backend == "gpu" and not devices:
        issues.append(
            RuntimeIssue(
                "error",
                "no_gpu_devices",
                "JAX selected the GPU backend but reported no local devices.",
                "Confirm the NVIDIA device is visible inside Linux/WSL2 and restart the Python process.",
            )
        )
    elif backend == "gpu" and any(device.platform != "gpu" for device in devices):
        issues.append(
            RuntimeIssue(
                "error",
                "non_gpu_device",
                "JAX returned a non-GPU local device while the GPU backend was selected.",
                "Repair the JAX CUDA installation before running inference.",
            )
        )

    if backend == "gpu" and len(devices) > 1:
        issues.append(
            RuntimeIssue(
                "warning",
                "single_device_default",
                f"JAX found {len(devices)} local GPU devices; this adapter uses one device by default.",
                "Select a device index explicitly only if the default device is not appropriate.",
            )
        )

    return RuntimeReport(
        platform_name,
        release,
        wsl,
        platform.python_version(),
        jax_version,
        jaxlib_version,
        backend,
        devices,
        tuple(issues),
    )


def assert_runtime(report: RuntimeReport | None = None) -> RuntimeReport:
    """Return a valid report or raise an actionable typed exception."""

    checked = inspect_environment() if report is None else report
    checked.raise_for_errors()
    return checked


__all__ = [
    "DeviceInfo",
    "GPUBackendRequiredError",
    "JaxUnavailableError",
    "NativeWindowsError",
    "RuntimeEnvironmentError",
    "RuntimeIssue",
    "RuntimeReport",
    "UnsupportedPlatformError",
    "assert_runtime",
    "inspect_environment",
]
