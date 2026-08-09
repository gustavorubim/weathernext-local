from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest
import xarray as xr

from weathernext_local.nowcast import (
    AuthorizationRequiredError,
    GlobalMetNetAdapter,
    GlobalMetNetUnavailableError,
    NowcastImportError,
    OriginAttestationRequiredError,
    global_metnet_status,
    import_global_metnet_output,
)


def _write_dataset(dataset: xr.Dataset, path) -> None:
    for name in ("latitude", "lat"):
        if name in dataset.coords:
            dataset.coords[name].attrs.setdefault("units", "degrees_north")
    for name in ("longitude", "lon"):
        if name in dataset.coords:
            dataset.coords[name].attrs.setdefault("units", "degrees_east")
    dataset.to_netcdf(path, engine="h5netcdf")


def _write_nowcast(path, *, units: str = "mm/h", accumulated: bool = False) -> None:
    name = "precipitation" if accumulated else "precipitation_rate"
    dataset = xr.Dataset(
        {
            name: (
                ("time", "lat", "lon"),
                np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                {"units": units, "long_name": "test precipitation"},
            )
        },
        coords={
            "time": np.array(["2026-08-09T12:15", "2026-08-09T12:30"], dtype="datetime64[m]"),
            "lat": np.array([-1.0, 0.0, 1.0]),
            "lon": np.array([10.0, 11.0, 12.0, 13.0]),
        },
    )
    _write_dataset(dataset, path)


def test_global_metnet_is_truthfully_fail_closed() -> None:
    status = global_metnet_status()
    assert not status.executable
    assert not status.source_code_released
    assert not status.trained_weights_released
    assert status.family == "nowcast"

    with pytest.raises(GlobalMetNetUnavailableError, match="no executable source release"):
        GlobalMetNetAdapter().run()


def test_import_requires_authorization_attestation(tmp_path) -> None:
    source = tmp_path / "nowcast.nc"
    _write_nowcast(source)

    with pytest.raises(AuthorizationRequiredError, match="explicitly attest"):
        import_global_metnet_output(
            source,
            authorized=False,
            origin_attested=True,
            origin_evidence="test-fixture",
            output_root=tmp_path / "runs",
        )


def test_import_requires_separate_origin_attestation(tmp_path) -> None:
    source = tmp_path / "nowcast.nc"
    _write_nowcast(source)

    with pytest.raises(OriginAttestationRequiredError, match="cannot independently verify"):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=False,
            origin_evidence="",
            output_root=tmp_path / "runs",
        )


@pytest.mark.parametrize("authorized", ["true", "false", 1, object()])
def test_import_requires_literal_boolean_authorization(tmp_path, authorized) -> None:
    source = tmp_path / "nowcast.nc"
    _write_nowcast(source)

    with pytest.raises(AuthorizationRequiredError, match="explicitly attest"):
        import_global_metnet_output(
            source,
            authorized=authorized,
            origin_attested=True,
            origin_evidence="test-fixture",
            output_root=tmp_path / "runs",
        )


@pytest.mark.parametrize("origin_attested", ["true", "false", 1, object()])
def test_import_requires_literal_boolean_origin_attestation(tmp_path, origin_attested) -> None:
    source = tmp_path / "nowcast.nc"
    _write_nowcast(source)

    with pytest.raises(OriginAttestationRequiredError, match="cannot independently verify"):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=origin_attested,
            origin_evidence="test-fixture",
            output_root=tmp_path / "runs",
        )


def test_import_rate_creates_provenance_preserving_run(tmp_path) -> None:
    source = tmp_path / "nowcast.nc"
    _write_nowcast(source)

    result = import_global_metnet_output(
        source,
        authorized=True,
        origin_attested=True,
        origin_evidence="test-fixture://global-metnet-output",
        run_id="metnet-authorized-001",
        output_root=tmp_path / "runs",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["model"]["id"] == "global-metnet"
    assert manifest["model"]["family"] == "nowcast"
    assert manifest["model"]["locally_executed"] is False
    assert manifest["model"]["origin"] == {
        "claim": "global-metnet",
        "asserted_by": "caller",
        "evidence": "test-fixture://global-metnet-output",
        "verified_by_repository": False,
    }
    assert manifest["input"]["source"]["authorization_attested"] is True
    assert manifest["inference"] == {
        "execution_mode": "authorized_output_import",
        "performed_by_this_repository": False,
        "time_step_minutes": 15,
    }
    assert manifest["primary_precipitation_variable"] == {
        "variable": "precipitation_rate",
        "units": "mm/h",
        "label": "precipitation rate",
    }

    with xr.open_dataset(result.forecast_path) as dataset:
        assert dataset["precipitation_rate"].dims == (
            "sample",
            "valid_time",
            "latitude",
            "longitude",
        )
        assert dataset["precipitation_rate"].attrs["units"] == "mm/h"
        assert dataset.coords["latitude"].attrs["units"] == "degrees_north"
        assert dataset.coords["longitude"].attrs["units"] == "degrees_east"
        assert dataset.sizes == {"sample": 1, "valid_time": 2, "latitude": 3, "longitude": 4}


def test_accumulation_never_guesses_interval(tmp_path) -> None:
    source = tmp_path / "accumulated.nc"
    _write_nowcast(source, units="mm", accumulated=True)

    with pytest.raises(NowcastImportError, match="require a positive"):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://global-metnet-output",
            output_root=tmp_path / "runs",
        )

    result = import_global_metnet_output(
        source,
        authorized=True,
        origin_attested=True,
        origin_evidence="test-fixture://global-metnet-output",
        accumulation_interval_minutes=15,
        run_id="metnet-accumulated-001",
        output_root=tmp_path / "runs",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["primary_precipitation_variable"]["interval"] == "15 min"


@pytest.mark.parametrize("interval", [True, False, 15.0, "15"])
def test_accumulation_requires_positive_non_boolean_integer_interval(tmp_path, interval) -> None:
    source = tmp_path / "accumulated.nc"
    _write_nowcast(source, units="mm", accumulated=True)

    with pytest.raises(NowcastImportError, match="non-boolean integer"):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://global-metnet-output",
            accumulation_interval_minutes=interval,
            output_root=tmp_path / "runs",
        )


def test_import_does_not_overwrite_existing_run(tmp_path) -> None:
    source = tmp_path / "nowcast.nc"
    _write_nowcast(source)
    kwargs = {
        "authorized": True,
        "origin_attested": True,
        "origin_evidence": "test-fixture://global-metnet-output",
        "run_id": "same-run",
        "output_root": tmp_path / "runs",
    }
    import_global_metnet_output(source, **kwargs)
    with pytest.raises(NowcastImportError, match="will not be overwritten"):
        import_global_metnet_output(source, **kwargs)


def test_import_rejects_shared_coordinate_dimensions(tmp_path) -> None:
    source = tmp_path / "invalid-grid.nc"
    dataset = xr.Dataset(
        {
            "precipitation_rate": (
                ("x",),
                np.array([0.0, 1.0], dtype=np.float32),
                {"units": "mm/h"},
            )
        },
        coords={
            "time": (
                "x",
                np.array(["2026-08-09T12:15", "2026-08-09T12:30"], dtype="datetime64[m]"),
            ),
            "lat": ("x", [-1.0, 1.0]),
            "lon": ("x", [10.0, 11.0]),
        },
    )
    _write_dataset(dataset, source)

    with pytest.raises(NowcastImportError, match="three independent dimensions"):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://invalid-grid",
            output_root=tmp_path / "runs",
        )


def test_import_rejects_time_on_reserved_sample_dimension(tmp_path) -> None:
    source = tmp_path / "sample-is-time.nc"
    dataset = xr.Dataset(
        {
            "precipitation_rate": (
                ("sample", "lat", "lon"),
                np.ones((2, 2, 2), dtype=np.float32),
                {"units": "mm/h"},
            )
        },
        coords={
            "valid_time": (
                "sample",
                np.array(["2026-08-09T12:15", "2026-08-09T12:30"], dtype="datetime64[m]"),
            ),
            "lat": [-1.0, 1.0],
            "lon": [10.0, 11.0],
        },
    )
    _write_dataset(dataset, source)

    with pytest.raises(NowcastImportError, match="reserved 'sample'"):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://sample-is-time",
            output_root=tmp_path / "runs",
        )


def test_import_rejects_relative_timedelta_without_reference_time(tmp_path) -> None:
    source = tmp_path / "relative-time.nc"
    dataset = xr.Dataset(
        {
            "precipitation_rate": (
                ("lead_time", "lat", "lon"),
                np.zeros((2, 2, 2), dtype=np.float32),
                {"units": "mm/h"},
            )
        },
        coords={
            "lead_time": np.array([15, 30], dtype="timedelta64[m]"),
            "lat": [-1.0, 1.0],
            "lon": [10.0, 11.0],
        },
    )
    _write_dataset(dataset, source)

    with pytest.raises(NowcastImportError, match="absolute datetimes"):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://relative-time",
            output_root=tmp_path / "runs",
        )


@pytest.mark.parametrize(
    ("latitudes", "values", "message"),
    [
        ([-1.0, 0.0, 999.0], np.ones((2, 3, 2), dtype=np.float32), "within"),
        ([-1.0, 0.25, 1.0], np.ones((2, 3, 2), dtype=np.float32), "regular spacing"),
        (
            [-1.0, 0.0, 1.0],
            np.full((2, 3, 2), np.nan, dtype=np.float32),
            "at least one finite",
        ),
        ([-1.0, 0.0, 1.0], -np.ones((2, 3, 2), dtype=np.float32), "negative"),
    ],
)
def test_import_rejects_dashboard_incompatible_values(tmp_path, latitudes, values, message) -> None:
    source = tmp_path / "invalid-values.nc"
    dataset = xr.Dataset(
        {
            "precipitation_rate": (
                ("time", "lat", "lon"),
                values,
                {"units": "mm/h"},
            )
        },
        coords={
            "time": np.array(["2026-08-09T12:15", "2026-08-09T12:30"], dtype="datetime64[m]"),
            "lat": latitudes,
            "lon": [10.0, 11.0],
        },
    )
    _write_dataset(dataset, source)

    with pytest.raises(NowcastImportError, match=message):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://invalid-values",
            output_root=tmp_path / "runs",
        )


def test_import_rejects_nat_valid_time_and_removes_private_snapshot(tmp_path, monkeypatch) -> None:
    private_temp = tmp_path / "private-temp"
    private_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private_temp))
    source = tmp_path / "nat-time.nc"
    dataset = xr.Dataset(
        {
            "precipitation_rate": (
                ("time", "lat", "lon"),
                np.ones((1, 2, 2), dtype=np.float32),
                {"units": "mm/h"},
            )
        },
        coords={
            "time": np.array(["NaT"], dtype="datetime64[ns]"),
            "lat": [-1.0, 1.0],
            "lon": [10.0, 11.0],
        },
    )
    _write_dataset(dataset, source)

    with pytest.raises(NowcastImportError, match=r"NaT|decode"):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://nat-time",
            output_root=tmp_path / "runs",
        )
    assert list(private_temp.iterdir()) == []
    assert list((tmp_path / "runs").iterdir()) == []


def test_import_rejects_one_empty_selectable_time(tmp_path) -> None:
    source = tmp_path / "mixed-finite.nc"
    values = np.ones((2, 2, 2), dtype=np.float32)
    values[1, :, :] = np.nan
    dataset = xr.Dataset(
        {
            "precipitation_rate": (
                ("time", "lat", "lon"),
                values,
                {"units": "mm/h"},
            )
        },
        coords={
            "time": np.array(["2026-08-09T12:15", "2026-08-09T12:30"], dtype="datetime64[m]"),
            "lat": [-1.0, 1.0],
            "lon": [10.0, 11.0],
        },
    )
    _write_dataset(dataset, source)

    with pytest.raises(NowcastImportError, match="Every selectable"):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://mixed-finite",
            output_root=tmp_path / "runs",
        )


@pytest.mark.parametrize(
    ("latitudes", "longitudes", "message"),
    [
        ([0.0], [10.0, 11.0], "at least two"),
        ([-1.0, 1.0], [0.0, 360.0], "modulo-360"),
    ],
)
def test_import_rejects_unrenderable_spatial_axes(tmp_path, latitudes, longitudes, message) -> None:
    source = tmp_path / "invalid-axis.nc"
    dataset = xr.Dataset(
        {
            "precipitation_rate": (
                ("time", "lat", "lon"),
                np.ones((1, len(latitudes), len(longitudes)), dtype=np.float32),
                {"units": "mm/h"},
            )
        },
        coords={
            "time": np.array(["2026-08-09T12:15"], dtype="datetime64[m]"),
            "lat": latitudes,
            "lon": longitudes,
        },
    )
    _write_dataset(dataset, source)

    with pytest.raises(NowcastImportError, match=message):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://invalid-axis",
            output_root=tmp_path / "runs",
        )


def test_import_rejects_unsigned_integer_axis_wraparound(tmp_path) -> None:
    source = tmp_path / "uint8-wraparound.nc"
    dataset = xr.Dataset(
        {
            "precipitation_rate": (
                ("time", "lat", "lon"),
                np.ones((1, 2, 3), dtype=np.float32),
                {"units": "mm/h"},
            )
        },
        coords={
            "time": np.array(["2026-08-09T12:15"], dtype="datetime64[m]"),
            "lat": np.array([0, 1], dtype=np.uint8),
            "lon": np.array([254, 255, 0], dtype=np.uint8),
        },
    )
    _write_dataset(dataset, source)

    with pytest.raises(NowcastImportError, match="strictly monotonic"):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://unsigned-wraparound",
            output_root=tmp_path / "runs",
        )


@pytest.mark.parametrize(
    ("coordinate", "values"),
    [
        ("lat", np.array([1 + 1j, 2 + 2j], dtype=np.complex128)),
        ("lon", np.array([10 + 1j, 11 + 2j], dtype=np.complex128)),
    ],
)
def test_import_rejects_complex_spatial_coordinates(tmp_path, coordinate, values) -> None:
    source = tmp_path / "complex-axis.nc"
    coordinates = {
        "time": np.array(["2026-08-09T12:15"], dtype="datetime64[m]"),
        "lat": np.array([1.0, 2.0]),
        "lon": np.array([10.0, 11.0]),
    }
    coordinates[coordinate] = values
    dataset = xr.Dataset(
        {
            "precipitation_rate": (
                ("time", "lat", "lon"),
                np.ones((1, 2, 2), dtype=np.float32),
                {"units": "mm/h"},
            )
        },
        coords=coordinates,
    )
    _write_dataset(dataset, source)

    with pytest.raises(NowcastImportError, match="real numeric"):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://complex-axis",
            output_root=tmp_path / "runs",
        )


@pytest.mark.parametrize(
    ("units", "message"),
    [
        (None, "must declare"),
        ("radians", "not recognized"),
        ("degree_s", "south/west-positive"),
    ],
)
def test_import_rejects_ambiguous_or_radian_coordinate_units(tmp_path, units, message) -> None:
    source = tmp_path / "coordinate-units.nc"
    dataset = xr.Dataset(
        {
            "precipitation_rate": (
                ("time", "lat", "lon"),
                np.ones((1, 2, 2), dtype=np.float32),
                {"units": "mm/h"},
            )
        },
        coords={
            "time": np.array(["2026-08-09T12:15"], dtype="datetime64[m]"),
            "lat": [-1.0, 1.0],
            "lon": [10.0, 11.0],
        },
    )
    if units is not None:
        dataset.coords["lat"].attrs["units"] = units
        dataset.coords["lon"].attrs["units"] = "degrees_east"
    dataset.to_netcdf(source, engine="h5netcdf")

    with pytest.raises(NowcastImportError, match=message):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://coordinate-units",
            output_root=tmp_path / "runs",
        )


def test_import_requires_selected_axes_to_be_coordinates(tmp_path) -> None:
    source = tmp_path / "axes-as-data-vars.nc"
    dataset = xr.Dataset(
        {
            "precipitation_rate": (
                ("t", "y", "x"),
                np.ones((1, 2, 2), dtype=np.float32),
                {"units": "mm/h"},
            ),
            "valid_time": ("t", np.array(["2026-08-09T12:15"], dtype="datetime64[m]")),
            "latitude": ("y", [-1.0, 1.0], {"units": "degrees_north"}),
            "longitude": ("x", [10.0, 11.0], {"units": "degrees_east"}),
        }
    )
    dataset.to_netcdf(source, engine="h5netcdf")

    with pytest.raises(NowcastImportError, match="must be NetCDF coordinates"):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://axes-as-data-vars",
            output_root=tmp_path / "runs",
        )


def test_import_requires_precipitation_to_be_a_data_variable(tmp_path) -> None:
    source = tmp_path / "precipitation-as-coordinate.nc"
    dataset = xr.Dataset(
        coords={
            "valid_time": np.array(["2026-08-09T12:15"], dtype="datetime64[m]"),
            "latitude": ("latitude", [-1.0, 1.0], {"units": "degrees_north"}),
            "longitude": ("longitude", [10.0, 11.0], {"units": "degrees_east"}),
            "precipitation_rate": (
                ("valid_time", "latitude", "longitude"),
                np.ones((1, 2, 2), dtype=np.float32),
                {"units": "mm/h"},
            ),
        }
    )
    dataset.to_netcdf(source, engine="h5netcdf")

    with pytest.raises(NowcastImportError, match="must be a NetCDF data variable"):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://coordinate-only-precipitation",
            output_root=tmp_path / "runs",
        )


def test_import_rejects_irregular_subsecond_time_steps(tmp_path) -> None:
    source = tmp_path / "subsecond-time.nc"
    dataset = xr.Dataset(
        {
            "precipitation_rate": (
                ("time", "lat", "lon"),
                np.ones((3, 2, 2), dtype=np.float32),
                {"units": "mm/h"},
            )
        },
        coords={
            "time": np.array(
                [
                    "2026-08-09T12:00:00.000",
                    "2026-08-09T12:15:00.100",
                    "2026-08-09T12:30:00.900",
                ],
                dtype="datetime64[ms]",
            ),
            "lat": [-1.0, 1.0],
            "lon": [10.0, 11.0],
        },
    )
    _write_dataset(dataset, source)

    with pytest.raises(NowcastImportError, match="regular interval"):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://subsecond-time",
            output_root=tmp_path / "runs",
        )


def test_import_rejects_empty_sample_dimension(tmp_path) -> None:
    source = tmp_path / "empty-sample.nc"
    dataset = xr.Dataset(
        {
            "precipitation_rate": (
                ("sample", "time", "lat", "lon"),
                np.empty((0, 1, 2, 2), dtype=np.float32),
                {"units": "mm/h"},
            )
        },
        coords={
            "sample": np.array([], dtype=np.int64),
            "time": np.array(["2026-08-09T12:15"], dtype="datetime64[m]"),
            "lat": [-1.0, 1.0],
            "lon": [10.0, 11.0],
        },
    )
    _write_dataset(dataset, source)

    with pytest.raises(NowcastImportError, match="at least one member"):
        import_global_metnet_output(
            source,
            authorized=True,
            origin_attested=True,
            origin_evidence="test-fixture://empty-sample",
            output_root=tmp_path / "runs",
        )
