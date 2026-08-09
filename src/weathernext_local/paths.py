"""Repository-local storage paths.

Large assets and generated runs deliberately live outside the Python package so
that a checkout can be upgraded without moving or accidentally packaging them.
"""

from __future__ import annotations

import os
from pathlib import Path


def _user_data_root() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base.expanduser().resolve() / "weathernext-local"


def project_root() -> Path:
    configured = os.environ.get("WNX_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    editable_candidate = Path(__file__).resolve().parents[2]
    if (editable_candidate / "pyproject.toml").is_file():
        return editable_candidate
    return _user_data_root()


def assets_root() -> Path:
    configured = os.environ.get("WNX_ASSETS_DIR")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else project_root() / "data" / "assets"
    )


def runs_root() -> Path:
    configured = os.environ.get("WNX_RUNS_DIR")
    return (
        Path(configured).expanduser().resolve() if configured else project_root() / "data" / "runs"
    )
