"""Command-line interface for local forecasting, ingestion, and visualization."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from weathernext_local.paths import assets_root, runs_root

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Run WeatherNext 2 Mini, catalog authorized nowcasts, and view precipitation maps.",
)

WINDOWS_PATH_PATTERN = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<tail>.*)$")


def _abort(message: str, *, code: int = 1) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code)


def _translate_absolute_windows_path(raw: str) -> Path:
    """Ask WSL to translate a drive path using its configured automount root."""

    try:
        completed = subprocess.run(
            ["/usr/bin/wslpath", "-u", raw],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError(
            f"WSL could not translate Windows path {raw!r}; verify /usr/bin/wslpath and the "
            "distribution's automount configuration."
        ) from exc
    translated = completed.stdout.strip()
    if not translated.startswith("/"):
        raise ValueError(f"WSL returned an invalid translation for Windows path {raw!r}.")
    return Path(translated)


def _portable_path(path: Path | None) -> Path | None:
    """Translate a drive-letter path when the Windows wrapper invokes WSL."""

    if path is None or os.name == "nt":
        return path
    raw = os.fspath(path)
    if raw.startswith(("\\\\", "//")):
        raise ValueError(
            "UNC paths are not translated automatically. Copy the file into this checkout "
            "or pass an absolute drive-letter path such as C:\\path\\file.nc."
        )
    if raw.startswith("\\"):
        raise ValueError(
            f"Rooted Windows path {raw!r} has no drive letter; use an absolute drive path."
        )
    if re.match(r"^[A-Za-z]:(?:$|[^\\/])", raw):
        raise ValueError(
            f"Drive-relative Windows path {raw!r} is ambiguous; use an absolute drive path."
        )
    match = WINDOWS_PATH_PATTERN.fullmatch(raw)
    if match is None:
        return Path(raw.replace("\\", "/"))
    return _translate_absolute_windows_path(raw)


def _is_loopback_host(host: str) -> bool:
    if host.strip().casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip()).is_loopback
    except ValueError:
        return False


@app.command()
def doctor(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the complete machine-readable runtime report."),
    ] = False,
) -> None:
    """Check that JAX sees an NVIDIA GPU in the supported Linux/WSL runtime."""

    from weathernext_local.doctor import inspect_environment

    report = inspect_environment()
    payload = report.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        state = "READY" if report.ok else "NOT READY"
        typer.echo(f"WeatherNext Mini runtime: {state}")
        typer.echo(f"  platform: {report.platform} {report.platform_release}")
        typer.echo(f"  WSL: {report.is_wsl}")
        typer.echo(f"  Python: {report.python_version}")
        typer.echo(f"  JAX: {report.jax_version or 'not available'}")
        typer.echo(f"  backend: {report.backend or 'not initialized'}")
        for device in report.devices:
            typer.echo(f"  device {device.id}: {device.kind} ({device.platform})")
        for issue in report.issues:
            typer.echo(f"  {issue.severity.upper()} [{issue.code}]: {issue.message}")
            typer.echo(f"    Remedy: {issue.remediation}")
    if not report.ok:
        raise typer.Exit(1)


@app.command("fetch")
def fetch_assets(
    steps: Annotated[
        int,
        typer.Option("--steps", min=1, help="Official sample lead count: 1, 4, 12, 20, or 30."),
    ] = 1,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Override the verified asset cache directory."),
    ] = None,
) -> None:
    """Download and integrity-check the official Mini weights and demo input."""

    from weathernext_local.assets import AssetError, ensure_assets

    try:
        selected_cache = _portable_path(cache_dir) or assets_root()
        typer.echo(f"Fetching verified public artifacts into {selected_cache} ...")
        result = ensure_assets(selected_cache, steps=steps)
    except (AssetError, OSError, ValueError) as exc:
        _abort(str(exc))
    typer.echo(
        f"Weights: {result.weights} ({result.weights_verification.size_bytes:,} bytes, verified)"
    )
    typer.echo(
        f"Input:   {result.sample} ({result.sample_verification.size_bytes:,} bytes, verified)"
    )


@app.command("run")
def run_weather_next(
    steps: Annotated[
        int,
        typer.Option("--steps", min=1, help="Official sample lead count: 1, 4, 12, 20, or 30."),
    ] = 1,
    members: Annotated[
        int,
        typer.Option(
            "--members", min=1, help="Sequential ensemble members; start with one on 12 GB VRAM."
        ),
    ] = 1,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 0,
    device: Annotated[int, typer.Option("--device", min=0, help="JAX local GPU index.")] = 0,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir", help="Explicit new run directory; generated automatically by default."
        ),
    ] = None,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Override the verified asset cache directory."),
    ] = None,
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Require already downloaded, verified artifacts."),
    ] = False,
) -> None:
    """Run official WeatherNext 2 Mini inference on one NVIDIA GPU."""

    # These must be set before importing JAX.  They avoid reserving almost all
    # VRAM up front and leave no ambiguity that this is inference-only.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    try:
        from weathernext_local.model import ModelAdapterError, RunConfig, run_forecast

        result = run_forecast(
            RunConfig(
                steps=steps,
                output_dir=_portable_path(output_dir),
                cache_dir=_portable_path(cache_dir),
                seed=seed,
                ensemble_members=members,
                device_index=device,
                download_assets=not offline,
            )
        )
    except (ModelAdapterError, OSError, RuntimeError, ValueError) as exc:
        _abort(str(exc))
    typer.echo(f"Run complete: {result.run_dir}")
    typer.echo(f"Forecast: {result.forecast_path}")
    typer.echo(f"Provenance: {result.manifest_path}")


@app.command("dashboard")
def dashboard(
    data_root: Annotated[
        Path | None,
        typer.Option("--data-root", help="Directory containing run subdirectories."),
    ] = None,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65_535)] = 8050,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    confirm_network_exposure: Annotated[
        bool,
        typer.Option(
            "--confirm-network-exposure",
            help="Required to bind the unauthenticated dashboard beyond loopback.",
        ),
    ] = False,
) -> None:
    """Launch the source-backed precipitation map dashboard."""

    from weathernext_local.dashboard import run_dashboard

    if not _is_loopback_host(host) and debug:
        _abort("Debug mode is allowed only on a loopback host.")
    if not _is_loopback_host(host) and not confirm_network_exposure:
        _abort(
            "A non-loopback bind exposes an unauthenticated local-data dashboard. "
            "Pass --confirm-network-exposure only if that is intentional."
        )
    try:
        selected_root = _portable_path(data_root) or runs_root()
    except ValueError as exc:
        _abort(str(exc))
    typer.echo(f"Serving catalog {selected_root} at http://{host}:{port}")
    run_dashboard(data_root=selected_root, host=host, port=port, debug=debug)


@app.command("metnet-status")
def metnet_status(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable capability metadata."),
    ] = False,
) -> None:
    """Show the verified Global MetNet release and execution state."""

    from weathernext_local.nowcast import global_metnet_status

    status = global_metnet_status()
    if json_output:
        typer.echo(json.dumps(status.to_dict(), indent=2))
        return
    typer.echo(f"{status.display_name}: {status.implementation_status}")
    typer.echo(status.reason)
    typer.echo(
        f"Published capability: ~{status.nominal_resolution_degrees}°, "
        f"{status.time_step_minutes}-minute steps, {status.forecast_horizon_hours}-hour horizon."
    )
    typer.echo(f"Checked: {status.checked_at}")
    typer.echo(f"Paper: {status.paper_url}")


@app.command("metnet-run")
def metnet_run() -> None:
    """Fail closed until Google releases executable Global MetNet artifacts."""

    from weathernext_local.nowcast import GlobalMetNetAdapter, GlobalMetNetUnavailableError

    try:
        GlobalMetNetAdapter().run()
    except GlobalMetNetUnavailableError as exc:
        _abort(str(exc), code=2)


@app.command("import-metnet")
def import_metnet(
    source: Annotated[
        Path,
        typer.Argument(dir_okay=False, help="Authorized Global MetNet NetCDF output."),
    ],
    confirm_authorized: Annotated[
        bool,
        typer.Option(
            "--confirm-authorized",
            help="Attest that you are permitted to use and store this source output.",
        ),
    ] = False,
    confirm_global_metnet_origin: Annotated[
        bool,
        typer.Option(
            "--confirm-global-metnet-origin",
            help="Attest separately that the file is a Global MetNet output.",
        ),
    ] = False,
    origin_evidence: Annotated[
        str,
        typer.Option(
            "--origin-evidence",
            help="Export/job/dataset identifier supporting the caller's origin claim.",
        ),
    ] = "",
    variable: Annotated[
        str | None,
        typer.Option(
            "--variable", help="Precipitation variable name when auto-detection is ambiguous."
        ),
    ] = None,
    latitude: Annotated[
        str | None, typer.Option("--latitude", help="Latitude coordinate name.")
    ] = None,
    longitude: Annotated[
        str | None, typer.Option("--longitude", help="Longitude coordinate name.")
    ] = None,
    time: Annotated[
        str | None, typer.Option("--time", help="Forecast time coordinate name.")
    ] = None,
    accumulation_interval_minutes: Annotated[
        int | None,
        typer.Option(
            "--accumulation-interval-minutes",
            min=1,
            help="Required only when the supplied precipitation is accumulated rather than a rate.",
        ),
    ] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", help="Override the run catalog root."),
    ] = None,
) -> None:
    """Catalog a legitimately obtained Global MetNet output without relabelling it as local inference."""

    from weathernext_local.nowcast import NowcastImportError, import_global_metnet_output

    try:
        result = import_global_metnet_output(
            _portable_path(source) or source,
            authorized=confirm_authorized,
            origin_attested=confirm_global_metnet_origin,
            origin_evidence=origin_evidence,
            variable=variable,
            latitude=latitude,
            longitude=longitude,
            time=time,
            accumulation_interval_minutes=accumulation_interval_minutes,
            run_id=run_id,
            output_root=_portable_path(output_root),
        )
    except (NowcastImportError, OSError, ValueError) as exc:
        _abort(str(exc))
    typer.echo(f"Imported run: {result.run_dir}")
    typer.echo("Execution mode: authorized_output_import (not locally generated)")


if __name__ == "__main__":
    app()
