"""Focused tests for CLI portability and dashboard exposure guards."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from weathernext_local import cli


@pytest.fixture
def linux_path_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the WSL branch without mutating the process-wide ``os.name``."""

    monkeypatch.setattr(
        cli,
        "os",
        SimpleNamespace(name="posix", fspath=os.fspath),
    )

    def translate(command: list[str], **_kwargs: object) -> SimpleNamespace:
        raw = command[-1]
        drive = raw[0].lower()
        tail = raw[3:].replace("\\", "/")
        return SimpleNamespace(stdout=f"/mnt/{drive}/{tail}\n")

    monkeypatch.setattr(cli.subprocess, "run", translate)


@pytest.mark.usefixtures("linux_path_semantics")
def test_portable_path_translates_absolute_windows_drive_path() -> None:
    translated = cli._portable_path(Path(r"C:\Users\gusta\forecast.nc"))

    assert translated is not None
    assert translated.as_posix() == "/mnt/c/Users/gusta/forecast.nc"


@pytest.mark.usefixtures("linux_path_semantics")
def test_portable_path_respects_configurable_wsl_automount_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []

    def translate(command: list[str], **_kwargs: object) -> SimpleNamespace:
        observed.append(command)
        return SimpleNamespace(stdout="/windir/c/Users/gusta/forecast.nc\n")

    monkeypatch.setattr(cli.subprocess, "run", translate)

    translated = cli._portable_path(Path(r"C:\Users\gusta\forecast.nc"))

    assert translated is not None
    assert translated.as_posix() == "/windir/c/Users/gusta/forecast.nc"
    assert observed == [["/usr/bin/wslpath", "-u", r"C:\Users\gusta\forecast.nc"]]


@pytest.mark.usefixtures("linux_path_semantics")
def test_portable_path_fails_closed_when_wslpath_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("wslpath")

    monkeypatch.setattr(cli.subprocess, "run", unavailable)

    with pytest.raises(ValueError, match="WSL could not translate"):
        cli._portable_path(Path(r"C:\Users\gusta\forecast.nc"))


@pytest.mark.usefixtures("linux_path_semantics")
def test_portable_path_normalizes_relative_backslashes() -> None:
    translated = cli._portable_path(Path(r"data\runs\forecast.nc"))

    assert translated is not None
    assert translated.as_posix() == "data/runs/forecast.nc"


@pytest.mark.usefixtures("linux_path_semantics")
@pytest.mark.parametrize(
    ("raw_path", "message_fragment"),
    [
        (r"\\server\share\forecast.nc", "UNC paths"),
        (r"\rooted\without-drive.nc", "has no drive letter"),
        ("C:", "Drive-relative Windows path"),
        (r"C:forecast.nc", "Drive-relative Windows path"),
    ],
)
def test_portable_path_rejects_ambiguous_windows_paths(
    raw_path: str,
    message_fragment: str,
) -> None:
    with pytest.raises(ValueError, match=message_fragment):
        cli._portable_path(Path(raw_path))


def test_portable_path_is_unchanged_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    source = Path(r"C:\Users\gusta\forecast.nc")
    monkeypatch.setattr(
        cli,
        "os",
        SimpleNamespace(name="nt", fspath=os.fspath),
    )

    assert cli._portable_path(source) is source


@pytest.mark.parametrize("host", ["localhost", " LOCALHOST ", "127.0.0.1", "127.9.8.7", "::1"])
def test_is_loopback_host_accepts_only_loopback_addresses(host: str) -> None:
    assert cli._is_loopback_host(host)


@pytest.mark.parametrize("host", ["", "0.0.0.0", "::", "192.168.1.10", "example.test"])
def test_is_loopback_host_rejects_non_loopback_hosts(host: str) -> None:
    assert not cli._is_loopback_host(host)


def test_dashboard_refuses_remote_debug_before_starting_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weathernext_local.dashboard as dashboard_package

    started = False

    def unexpected_start(**_kwargs: object) -> None:
        nonlocal started
        started = True

    monkeypatch.setattr(dashboard_package, "run_dashboard", unexpected_start)

    with pytest.raises(typer.Exit) as raised:
        cli.dashboard(
            data_root=None,
            host="0.0.0.0",
            port=8050,
            debug=True,
            confirm_network_exposure=True,
        )

    assert raised.value.exit_code == 1
    assert not started


def test_dashboard_requires_confirmation_before_remote_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weathernext_local.dashboard as dashboard_package

    started = False

    def unexpected_start(**_kwargs: object) -> None:
        nonlocal started
        started = True

    monkeypatch.setattr(dashboard_package, "run_dashboard", unexpected_start)

    with pytest.raises(typer.Exit) as raised:
        cli.dashboard(
            data_root=None,
            host="0.0.0.0",
            port=8050,
            debug=False,
            confirm_network_exposure=False,
        )

    assert raised.value.exit_code == 1
    assert not started
