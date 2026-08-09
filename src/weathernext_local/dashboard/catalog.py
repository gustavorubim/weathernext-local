"""Run discovery for the local dashboard.

Only complete run directories containing both an object-valued manifest and
the canonical ``forecast.nc`` artifact enter the selectable catalog. Invalid
directories are retained as issues so the UI can explain why they are absent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CatalogIssue:
    """A run directory that could not safely enter the catalog."""

    path: Path
    message: str


@dataclass(frozen=True)
class RunRecord:
    """Paths and immutable manifest metadata for one complete run."""

    run_id: str
    run_dir: Path
    manifest_path: Path
    forecast_path: Path
    manifest: Mapping[str, Any]

    @property
    def model_family(self) -> str | None:
        """Return an explicitly declared model family, if present."""

        direct = self.manifest.get("model_family")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        model = self.manifest.get("model")
        if isinstance(model, Mapping):
            family = model.get("family")
            if isinstance(family, str) and family.strip():
                return family.strip()
        return None

    @property
    def model_id(self) -> str | None:
        """Return an explicitly declared model identifier, if present."""

        model = self.manifest.get("model")
        if isinstance(model, Mapping):
            value = model.get("id") or model.get("name")
            if isinstance(value, str) and value.strip():
                return value.strip()
        value = self.manifest.get("model_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @property
    def display_label(self) -> str:
        """Build a filesystem/manifest-backed selector label."""

        model = self.model_id
        family = self.model_family
        qualifier = model or family
        return f"{self.run_id} · {qualifier}" if qualifier else self.run_id


@dataclass(frozen=True)
class RunCatalog:
    """Complete selectable runs plus non-fatal discovery issues."""

    root: Path
    runs: tuple[RunRecord, ...]
    issues: tuple[CatalogIssue, ...]

    def get(self, run_id: str) -> RunRecord | None:
        return next((run for run in self.runs if run.run_id == run_id), None)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({run.model_family for run in self.runs if run.model_family}))


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifest could not be read: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be a JSON object")
    return manifest


def discover_runs(root: str | Path = "data/runs") -> RunCatalog:
    """Discover immediate child run directories without opening NetCDF data.

    Discovery is intentionally metadata-only. ``forecast.nc`` is checked for
    existence but not opened, preserving lazy dataset access until selection.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return RunCatalog(root=root_path, runs=(), issues=())
    if not root_path.is_dir():
        return RunCatalog(
            root=root_path,
            runs=(),
            issues=(CatalogIssue(root_path, "run root is not a directory"),),
        )

    runs: list[RunRecord] = []
    issues: list[CatalogIssue] = []
    for manifest_path in sorted(root_path.glob("*/manifest.json")):
        run_dir = manifest_path.parent
        forecast_path = run_dir / "forecast.nc"
        try:
            manifest = _read_manifest(manifest_path)
        except ValueError as exc:
            issues.append(CatalogIssue(manifest_path, str(exc)))
            continue
        if not forecast_path.is_file():
            issues.append(CatalogIssue(run_dir, "forecast.nc is missing"))
            continue
        runs.append(
            RunRecord(
                run_id=run_dir.name,
                run_dir=run_dir.resolve(),
                manifest_path=manifest_path.resolve(),
                forecast_path=forecast_path.resolve(),
                manifest=manifest,
            )
        )

    # Newer timestamp-like run IDs naturally appear first, while arbitrary IDs
    # remain deterministic. No manifest timestamps are inferred or rewritten.
    runs.sort(key=lambda run: run.run_id, reverse=True)
    return RunCatalog(root=root_path, runs=tuple(runs), issues=tuple(issues))
