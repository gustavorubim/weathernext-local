"""Truthful execution boundary for Google's Global MetNet.

Google has described an operational Global MetNet system, but has not released
the executable source code or trained weights.  This module intentionally does
not substitute a different model or fabricate an implementation.  It exposes a
stable capability record and a fail-closed adapter which can be replaced only
after an official, licensable release is verified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class GlobalMetNetUnavailableError(RuntimeError):
    """Raised when local Global MetNet inference is requested without a release."""


@dataclass(frozen=True, slots=True)
class GlobalMetNetReleaseStatus:
    """Verified public capability and release state for Global MetNet."""

    model_id: str = "global-metnet"
    display_name: str = "Global MetNet"
    family: str = "nowcast"
    executable: bool = False
    implementation_status: str = "blocked_upstream_artifacts"
    checked_at: str = "2026-08-09"
    forecast_horizon_hours: int = 12
    time_step_minutes: int = 15
    nominal_resolution_degrees: float = 0.05
    source_code_released: bool = False
    trained_weights_released: bool = False
    reason: str = (
        "Google has published the Global MetNet paper and product description, "
        "but no executable source release or trained checkpoint is publicly available."
    )
    paper_url: str = "https://arxiv.org/abs/2510.13050"
    product_url: str = (
        "https://research.google/blog/google-research-2025-bolder-breakthroughs-bigger-impact/"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def global_metnet_status() -> GlobalMetNetReleaseStatus:
    """Return the source-verified public release state pinned for this project."""

    return GlobalMetNetReleaseStatus()


class GlobalMetNetAdapter:
    """Reserved adapter for a future official Global MetNet release.

    Keeping the refusal in executable code prevents a downstream caller from
    accidentally interpreting an unrelated nowcasting model as Global MetNet.
    """

    @property
    def status(self) -> GlobalMetNetReleaseStatus:
        return global_metnet_status()

    def run(self, *_: Any, **__: Any) -> None:
        status = self.status
        raise GlobalMetNetUnavailableError(
            f"{status.display_name} local execution is {status.implementation_status}: "
            f"{status.reason} Checked {status.checked_at}. See {status.paper_url}. "
            "Use the authorized-output importer only if you possess a legitimate "
            "Global MetNet result; it will be labelled as imported, not locally generated."
        )


__all__ = [
    "GlobalMetNetAdapter",
    "GlobalMetNetReleaseStatus",
    "GlobalMetNetUnavailableError",
    "global_metnet_status",
]
