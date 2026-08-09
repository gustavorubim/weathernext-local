"""Local, source-backed forecast dashboard.

The package keeps catalog, data selection, and raster rendering separate from
Dash so those pieces can be tested without starting a web server.
"""

from .catalog import CatalogIssue, RunCatalog, RunRecord, discover_runs


def create_app(*args, **kwargs):
    """Import Dash only when the interactive application is requested."""

    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


def run_dashboard(*args, **kwargs):
    """Import Dash only when starting the interactive application."""

    from .app import run_dashboard as _run_dashboard

    return _run_dashboard(*args, **kwargs)


__all__ = [
    "CatalogIssue",
    "RunCatalog",
    "RunRecord",
    "create_app",
    "discover_runs",
    "run_dashboard",
]
