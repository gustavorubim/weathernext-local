from __future__ import annotations

import base64
import hashlib
import io
import urllib.error
from pathlib import Path

import pytest

from weathernext_local import assets


def _spec(payload: bytes, *, key: str = "test") -> assets.AssetSpec:
    digest = hashlib.md5(payload, usedforsecurity=False).digest()
    return assets.AssetSpec(
        key=key,
        object_path="folder/an object_<2024.bin",
        local_filename=f"{key}.bin",
        size_bytes=len(payload),
        md5_base64=base64.b64encode(digest).decode("ascii"),
    )


def test_release_asset_metadata_is_exact() -> None:
    assert assets.WEIGHTS_SPEC.object_path == (
        "weathernext2/params/WeatherNextCyclones_Mini_<2024.npz"
    )
    assert (assets.WEIGHTS_SPEC.size_bytes, assets.WEIGHTS_SPEC.md5_base64) == (
        226_897_594,
        "R1qGPfvNoqNl62xzJfz6XA==",
    )
    assert {
        step: (spec.size_bytes, spec.md5_base64) for step, spec in assets.SAMPLE_SPECS.items()
    } == {
        1: (79_552_583, "KDz7si0OQ8hLxTAQSuyH5Q=="),
        4: (158_535_239, "eI8kzsg5wr/2Y6LD5aFFiA=="),
        12: (369_155_655, "xvOO3Ov4SXQvXMNzn+xuDA=="),
        20: (579_776_071, "iUe5AKoiKcuEmNGp44unCg=="),
        30: (843_051_591, "2vT1aAWyL2Foe+/AkMitFA=="),
    }


@pytest.mark.parametrize(("provided", "expected"), [(1, 1), ("01", 1), (" 20 ", 20), (30, 30)])
def test_normalize_steps_accepts_only_published_samples(provided: int | str, expected: int) -> None:
    assert assets.normalize_steps(provided) == expected


@pytest.mark.parametrize("provided", [0, 2, 31, True, 1.0, "1.0", "nope", None])
def test_normalize_steps_rejects_unsupported_values(provided: object) -> None:
    with pytest.raises(assets.UnsupportedStepsError, match="Choose one of: 1, 4, 12, 20, 30"):
        assets.normalize_steps(provided)  # type: ignore[arg-type]


def test_public_url_percent_encodes_object_name() -> None:
    url = assets.WEIGHTS_SPEC.url
    assert url.startswith("https://storage.googleapis.com/dm_graphcast/")
    assert "%3C2024" in url
    assert "<2024" not in url
    sample_url = assets.sample_spec(1).url
    assert "%20" in sample_url


def test_verify_asset_streams_and_returns_identity(tmp_path: Path) -> None:
    payload = b"weather" * 10_000
    spec = _spec(payload)
    path = tmp_path / spec.local_filename
    path.write_bytes(payload)

    verified = assets.verify_asset(path, spec, chunk_size=37)

    assert verified.path == path
    assert verified.size_bytes == len(payload)
    assert verified.md5_base64 == spec.md5_base64


def test_verify_asset_reports_missing_size_and_digest(tmp_path: Path) -> None:
    good = b"abcd"
    spec = _spec(good)
    missing = tmp_path / "missing.bin"
    with pytest.raises(assets.AssetMissingError, match="missing"):
        assets.verify_asset(missing, spec)

    missing.write_bytes(b"abc")
    with pytest.raises(assets.AssetIntegrityError, match="expected 4 bytes, found 3"):
        assets.verify_asset(missing, spec)

    missing.write_bytes(b"abce")
    with pytest.raises(assets.AssetIntegrityError, match="expected base64 MD5"):
        assets.verify_asset(missing, spec)


class _Response(io.BytesIO):
    pass


def test_download_is_streamed_verified_and_atomic(tmp_path: Path) -> None:
    payload = b"0123456789" * 100
    spec = _spec(payload)
    calls: list[tuple[str, float]] = []
    progress: list[tuple[int, int]] = []

    def opener(request: object, *, timeout: float) -> _Response:
        calls.append((request.full_url, timeout))  # type: ignore[attr-defined]
        return _Response(payload)

    target = tmp_path / "cache" / spec.local_filename
    result = assets.download_asset(
        spec,
        target,
        timeout=12.5,
        chunk_size=17,
        progress=lambda received, total: progress.append((received, total)),
        opener=opener,
    )

    assert result.path == target
    assert target.read_bytes() == payload
    assert calls == [(spec.url, 12.5)]
    assert progress[-1] == (len(payload), len(payload))
    assert not list(target.parent.glob("*.part"))

    # A second call reuses the verified object and does not open the network.
    assets.download_asset(spec, target, opener=opener)
    assert len(calls) == 1


def test_failed_repair_preserves_existing_file(tmp_path: Path) -> None:
    old_payload = b"old-but-corrupt"
    expected = b"expected"
    spec = _spec(expected)
    target = tmp_path / spec.local_filename
    target.write_bytes(old_payload)

    def opener(request: object, *, timeout: float) -> _Response:
        del request, timeout
        return _Response(b"bad-data")

    with pytest.raises(assets.AssetIntegrityError, match=r"existing cache entry.*left unchanged"):
        assets.download_asset(spec, target, repair=True, opener=opener)
    assert target.read_bytes() == old_payload


def test_download_wraps_network_error_and_leaves_no_partial(tmp_path: Path) -> None:
    spec = _spec(b"payload")
    target = tmp_path / spec.local_filename

    def opener(request: object, *, timeout: float) -> _Response:
        del request, timeout
        raise urllib.error.URLError("offline")

    with pytest.raises(assets.AssetDownloadError, match="Check network access"):
        assets.download_asset(spec, target, opener=opener)
    assert not target.exists()
    assert not list(tmp_path.glob("*.part"))


def test_ensure_assets_can_fail_closed_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weight_payload = b"weight"
    sample_payload = b"sample"
    weight_spec = _spec(weight_payload, key="weights")
    sample = _spec(sample_payload, key="sample-01")
    monkeypatch.setattr(assets, "WEIGHTS_SPEC", weight_spec)
    monkeypatch.setitem(assets.SAMPLE_SPECS, 1, sample)
    (tmp_path / weight_spec.local_filename).write_bytes(weight_payload)
    (tmp_path / sample.local_filename).write_bytes(sample_payload)

    resolved = assets.ensure_assets(tmp_path, 1, download=False)

    assert resolved.weights.read_bytes() == weight_payload
    assert resolved.sample.read_bytes() == sample_payload
