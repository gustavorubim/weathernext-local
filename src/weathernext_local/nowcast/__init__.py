"""Fail-closed Global MetNet integration and authorized output ingestion."""

from weathernext_local.nowcast.global_metnet import (
    GlobalMetNetAdapter,
    GlobalMetNetReleaseStatus,
    GlobalMetNetUnavailableError,
    global_metnet_status,
)
from weathernext_local.nowcast.importer import (
    AuthorizationRequiredError,
    ImportedNowcastRun,
    NowcastImportError,
    OriginAttestationRequiredError,
    import_global_metnet_output,
)

__all__ = [
    "AuthorizationRequiredError",
    "GlobalMetNetAdapter",
    "GlobalMetNetReleaseStatus",
    "GlobalMetNetUnavailableError",
    "ImportedNowcastRun",
    "NowcastImportError",
    "OriginAttestationRequiredError",
    "global_metnet_status",
    "import_global_metnet_output",
]
