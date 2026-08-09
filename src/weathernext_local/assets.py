"""Verified public artifacts used by the WeatherNext 2 Mini demo.

The Google Cloud Storage objects are public.  Downloads are deliberately
implemented with the standard library so this module remains usable before the
large, Linux-only model dependency set is installed.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import os
import tempfile
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import closing, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol
from urllib.parse import quote

from weathernext_local.paths import assets_root

GCS_BUCKET = "dm_graphcast"
GCS_HTTPS_ROOT = f"https://storage.googleapis.com/{GCS_BUCKET}"
GCS_URI_ROOT = f"gs://{GCS_BUCKET}"
SUPPORTED_SAMPLE_STEPS = (1, 4, 12, 20, 30)
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024


class AssetError(RuntimeError):
    """Base class for asset acquisition and verification failures."""


class UnsupportedStepsError(AssetError, ValueError):
    """Raised when no official sample exists for the requested lead count."""


class AssetMissingError(AssetError, FileNotFoundError):
    """Raised when an expected local artifact is absent."""


class AssetIntegrityError(AssetError):
    """Raised when an artifact does not match its official size and MD5."""


class AssetDownloadError(AssetError):
    """Raised when a public object cannot be downloaded and verified."""


@dataclass(frozen=True, slots=True)
class AssetSpec:
    """Immutable identity for one official GCS object."""

    key: str
    object_path: str
    local_filename: str
    size_bytes: int
    md5_base64: str

    @property
    def url(self) -> str:
        # GCS object names in this release contain spaces and ``<``.
        return f"{GCS_HTTPS_ROOT}/{quote(self.object_path, safe='/')}"

    @property
    def gcs_uri(self) -> str:
        return f"{GCS_URI_ROOT}/{self.object_path}"


WEIGHTS_SPEC = AssetSpec(
    key="weights",
    object_path="weathernext2/params/WeatherNextCyclones_Mini_<2024.npz",
    local_filename="WeatherNextCyclones_Mini_lt2024.npz",
    size_bytes=226_897_594,
    md5_base64="R1qGPfvNoqNl62xzJfz6XA==",
)


def _sample_spec(steps: int, size_bytes: int, md5_base64: str) -> AssetSpec:
    token = f"{steps:02d}"
    object_path = (
        "weathernext2/dataset/"
        "source-hres_forecast_init-2024-10-07 00:00:00_"
        f"res-1.0_levels-13_steps-{token}.nc"
    )
    return AssetSpec(
        key=f"sample-{token}",
        object_path=object_path,
        local_filename=f"hres_init-2024-10-07T00-00-00Z_res-1.0_levels-13_steps-{token}.nc",
        size_bytes=size_bytes,
        md5_base64=md5_base64,
    )


SAMPLE_SPECS: dict[int, AssetSpec] = {
    1: _sample_spec(1, 79_552_583, "KDz7si0OQ8hLxTAQSuyH5Q=="),
    4: _sample_spec(4, 158_535_239, "eI8kzsg5wr/2Y6LD5aFFiA=="),
    12: _sample_spec(12, 369_155_655, "xvOO3Ov4SXQvXMNzn+xuDA=="),
    20: _sample_spec(20, 579_776_071, "iUe5AKoiKcuEmNGp44unCg=="),
    30: _sample_spec(30, 843_051_591, "2vT1aAWyL2Foe+/AkMitFA=="),
}


def normalize_steps(steps: int | str) -> int:
    """Return an official sample step count, rejecting ambiguous values."""

    if isinstance(steps, bool):
        value = -1
    elif isinstance(steps, int):
        value = steps
    elif isinstance(steps, str) and steps.strip().isdigit():
        value = int(steps)
    else:
        value = -1
    if value not in SAMPLE_SPECS:
        supported = ", ".join(str(item) for item in SUPPORTED_SAMPLE_STEPS)
        raise UnsupportedStepsError(
            f"Unsupported sample step count {steps!r}. Choose one of: {supported}."
        )
    return value


def sample_spec(steps: int | str) -> AssetSpec:
    """Return the immutable public sample identity for ``steps``."""

    return SAMPLE_SPECS[normalize_steps(steps)]


@dataclass(frozen=True, slots=True)
class AssetVerification:
    path: Path
    size_bytes: int
    md5_base64: str

    def to_manifest(self, spec: AssetSpec) -> dict[str, object]:
        return {
            "path": str(self.path.resolve()),
            "source_uri": spec.gcs_uri,
            "size_bytes": self.size_bytes,
            "md5": {"algorithm": "md5", "base64": self.md5_base64},
        }


@dataclass(frozen=True, slots=True)
class AssetPaths:
    weights: Path
    sample: Path
    weights_verification: AssetVerification
    sample_verification: AssetVerification


def _md5_base64(chunks: Iterator[bytes]) -> tuple[int, str]:
    digest = hashlib.md5(usedforsecurity=False)
    total = 0
    for chunk in chunks:
        total += len(chunk)
        digest.update(chunk)
    return total, base64.b64encode(digest.digest()).decode("ascii")


def _file_chunks(source: BinaryIO, chunk_size: int) -> Iterator[bytes]:
    while chunk := source.read(chunk_size):
        yield chunk


def verify_asset(
    path: str | os.PathLike[str],
    spec: AssetSpec,
    *,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
) -> AssetVerification:
    """Verify one local object against the release metadata.

    Size is checked before hashing so truncated files fail quickly.  MD5 is
    used because that is the digest published by GCS for these exact objects;
    it is an identity check, not a password-security primitive.
    """

    target = Path(path)
    if not target.is_file():
        raise AssetMissingError(
            f"Required {spec.key} asset is missing: {target}. "
            "Run the asset download command while online."
        )
    actual_size = target.stat().st_size
    if actual_size != spec.size_bytes:
        raise AssetIntegrityError(
            f"Corrupt {spec.key} asset {target}: expected {spec.size_bytes} bytes, "
            f"found {actual_size}. Re-download the asset."
        )
    with target.open("rb") as source:
        hashed_size, actual_md5 = _md5_base64(_file_chunks(source, chunk_size))
    if hashed_size != actual_size:
        raise AssetIntegrityError(
            f"The {spec.key} asset changed while it was being verified: {target}."
        )
    if not hmac.compare_digest(actual_md5, spec.md5_base64):
        raise AssetIntegrityError(
            f"Corrupt {spec.key} asset {target}: expected base64 MD5 "
            f"{spec.md5_base64}, found {actual_md5}. Re-download the asset."
        )
    return AssetVerification(target, actual_size, actual_md5)


class _Response(Protocol):
    def read(self, amount: int = -1) -> bytes: ...

    def close(self) -> None: ...


class _Opener(Protocol):
    def __call__(self, request: urllib.request.Request, timeout: float) -> _Response: ...


ProgressCallback = Callable[[int, int], None]


def download_asset(
    spec: AssetSpec,
    destination: str | os.PathLike[str],
    *,
    repair: bool = True,
    timeout: float = 60.0,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
    progress: ProgressCallback | None = None,
    opener: _Opener = urllib.request.urlopen,
) -> AssetVerification:
    """Stream, verify, and atomically install an official public object.

    An existing valid file is reused.  An existing corrupt file is not touched
    until a complete replacement has passed both checks, so interruption can
    never turn a recoverable cache entry into a partially downloaded artifact.
    """

    target = Path(destination)
    if target.exists():
        try:
            return verify_asset(target, spec, chunk_size=chunk_size)
        except (AssetMissingError, AssetIntegrityError):
            if not repair:
                raise

    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        spec.url,
        headers={"User-Agent": "weathernext-local/0.1 (+verified-stream-download)"},
    )
    temp_path: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".part", dir=target.parent
        )
        temp_path = Path(temp_name)
        digest = hashlib.md5(usedforsecurity=False)
        received = 0
        with os.fdopen(fd, "wb") as sink, closing(opener(request, timeout=timeout)) as source:
            while chunk := source.read(chunk_size):
                received += len(chunk)
                if received > spec.size_bytes:
                    raise AssetIntegrityError(
                        f"Download for {spec.key} exceeded its expected size of "
                        f"{spec.size_bytes} bytes. The public object did not match the release metadata."
                    )
                sink.write(chunk)
                digest.update(chunk)
                if progress is not None:
                    progress(received, spec.size_bytes)
            sink.flush()
            os.fsync(sink.fileno())

        actual_md5 = base64.b64encode(digest.digest()).decode("ascii")
        if received != spec.size_bytes or not hmac.compare_digest(actual_md5, spec.md5_base64):
            raise AssetIntegrityError(
                f"Downloaded {spec.key} failed verification: expected {spec.size_bytes} bytes / "
                f"{spec.md5_base64}, received {received} bytes / {actual_md5}. "
                "The existing cache entry, if any, was left unchanged."
            )
        os.replace(temp_path, target)
        temp_path = None
        return AssetVerification(target, received, actual_md5)
    except AssetIntegrityError:
        raise
    except (OSError, http.client.HTTPException) as exc:
        raise AssetDownloadError(
            f"Could not download public {spec.key} asset from {spec.url}: {exc}. "
            "Check network access and free disk space, then retry."
        ) from exc
    finally:
        if temp_path is not None:
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)


def ensure_assets(
    cache_dir: str | os.PathLike[str] | None = None,
    steps: int | str = 1,
    *,
    download: bool = True,
    weights_path: str | os.PathLike[str] | None = None,
    sample_path: str | os.PathLike[str] | None = None,
    repair: bool = True,
    progress: ProgressCallback | None = None,
) -> AssetPaths:
    """Resolve and strictly verify the checkpoint and requested sample.

    Explicit paths are still checked against the official object identities.
    Set ``download=False`` for an offline/fail-closed check.
    """

    selected_sample = sample_spec(steps)
    root = Path(cache_dir) if cache_dir is not None else assets_root()
    weights_target = (
        Path(weights_path) if weights_path is not None else root / WEIGHTS_SPEC.local_filename
    )
    sample_target = (
        Path(sample_path) if sample_path is not None else root / selected_sample.local_filename
    )

    if download:
        weights_check = download_asset(
            WEIGHTS_SPEC, weights_target, repair=repair, progress=progress
        )
        sample_check = download_asset(
            selected_sample, sample_target, repair=repair, progress=progress
        )
    else:
        weights_check = verify_asset(weights_target, WEIGHTS_SPEC)
        sample_check = verify_asset(sample_target, selected_sample)
    return AssetPaths(weights_target, sample_target, weights_check, sample_check)


__all__ = [
    "SAMPLE_SPECS",
    "SUPPORTED_SAMPLE_STEPS",
    "WEIGHTS_SPEC",
    "AssetDownloadError",
    "AssetError",
    "AssetIntegrityError",
    "AssetMissingError",
    "AssetPaths",
    "AssetSpec",
    "AssetVerification",
    "UnsupportedStepsError",
    "download_asset",
    "ensure_assets",
    "normalize_steps",
    "sample_spec",
    "verify_asset",
]
