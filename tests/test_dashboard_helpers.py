from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from weathernext_local.dashboard.catalog import discover_runs
from weathernext_local.dashboard.data import (
    inspect_dataset,
    nearest_cell_series,
    open_forecast,
    select_precipitation,
)
from weathernext_local.dashboard.raster import (
    WN2_6H_SCALE,
    raster_bin_indices,
    render_image_overlay,
)


@pytest.fixture
def forecast_dataset() -> xr.Dataset:
    times = np.array(["2026-08-09T00:00", "2026-08-09T06:00"], dtype="datetime64[ns]")
    values_m = np.array(
        [
            [
                [[0.001, 0.002], [0.003, 0.004]],
                [[0.005, 0.006], [0.007, 0.008]],
            ],
            [
                [[0.011, 0.012], [0.013, 0.014]],
                [[0.015, 0.016], [0.017, 0.018]],
            ],
        ],
        dtype=np.float32,
    )
    dataset = xr.Dataset(
        data_vars={
            "total_precipitation_6hr": (
                ("sample", "time", "latitude", "longitude"),
                values_m,
                {"units": "m"},
            )
        },
        coords={
            "sample": [0, 1],
            "time": times,
            "latitude": [10.0, 20.0],
            "longitude": [100.0, 110.0],
        },
    )
    dataset.coords["latitude"].attrs["units"] = "degrees_north"
    dataset.coords["longitude"].attrs["units"] = "degrees_east"
    return dataset


def test_discover_runs_requires_object_manifest_and_forecast(tmp_path) -> None:
    runs_root = tmp_path / "data" / "runs"
    complete = runs_root / "complete"
    complete.mkdir(parents=True)
    manifest = {"run_id": "complete", "model": "WeatherNext 2 Mini"}
    (complete / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (complete / "forecast.nc").write_bytes(b"catalog discovery does not open this file")

    missing_forecast = runs_root / "missing-forecast"
    missing_forecast.mkdir()
    (missing_forecast / "manifest.json").write_text("{}", encoding="utf-8")

    malformed = runs_root / "malformed"
    malformed.mkdir()
    (malformed / "manifest.json").write_text("[]", encoding="utf-8")
    (malformed / "forecast.nc").write_bytes(b"not a real NetCDF file")

    catalog = discover_runs(runs_root)

    assert len(catalog.runs) == 1
    assert catalog.runs[0].forecast_path == complete / "forecast.nc"
    assert catalog.runs[0].manifest == manifest
    assert len(catalog.issues) == 2
    issue_text = " ".join(map(str, catalog.issues))
    assert "missing-forecast" in issue_text
    assert "malformed" in issue_text


def test_open_forecast_keeps_data_lazy(tmp_path, forecast_dataset) -> None:
    forecast_path = tmp_path / "forecast.nc"
    forecast_dataset.to_netcdf(forecast_path, engine="h5netcdf")

    opened = open_forecast(forecast_path)
    try:
        variable = opened["total_precipitation_6hr"].variable
        assert not variable._in_memory
        np.testing.assert_allclose(
            variable.isel(sample=0, time=0).values,
            [[0.001, 0.002], [0.003, 0.004]],
        )
    finally:
        opened.close()


def test_inspect_dataset_identifies_wn2_axes_and_conversion(forecast_dataset) -> None:
    layout = inspect_dataset(forecast_dataset)

    assert layout.latitude_name == "latitude"
    assert layout.longitude_name == "longitude"
    assert layout.time_dimension == "time"
    assert layout.time_coordinate == "time"
    assert layout.sample_dimension == "sample"
    assert layout.sample_size == 2
    assert layout.time_size == 2
    assert layout.precipitation.variable == "total_precipitation_6hr"
    assert layout.precipitation.source_units == "m"
    assert layout.precipitation.display_units == "mm/6 h"
    assert layout.precipitation.scale_factor == pytest.approx(1000.0)


def test_select_precipitation_uses_time_mean_and_converts_metres_to_mm(
    forecast_dataset,
) -> None:
    field = select_precipitation(forecast_dataset, time_index=1, ensemble_mode="mean")

    assert field.dims == ("latitude", "longitude")
    assert field.attrs["units"] == "mm/6 h"
    np.testing.assert_allclose(field.values, [[10.0, 11.0], [12.0, 13.0]])


def test_select_precipitation_can_choose_one_sample(forecast_dataset) -> None:
    field = select_precipitation(
        forecast_dataset,
        time_index=0,
        ensemble_mode="member",
        member_index=1,
    )

    np.testing.assert_allclose(field.values, [[11.0, 12.0], [13.0, 14.0]])


def test_map_selection_explicitly_floors_negative_values_without_changing_source(
    forecast_dataset,
) -> None:
    forecast_dataset["total_precipitation_6hr"][0, 0, 0, 0] = -0.00005

    field = select_precipitation(
        forecast_dataset,
        time_index=0,
        ensemble_mode="member",
        member_index=0,
    )

    assert field.values[0, 0] == 0.0
    assert field.attrs["display_floor_count"] == 1
    assert field.attrs["display_floor_minimum"] == pytest.approx(-0.05)
    assert field.attrs["raw_artifact_modified"] is False
    assert forecast_dataset["total_precipitation_6hr"].values[0, 0, 0, 0] == pytest.approx(-0.00005)


def test_nearest_cell_series_returns_selected_coordinates_and_mm_values(
    forecast_dataset,
) -> None:
    series = nearest_cell_series(
        forecast_dataset,
        latitude=19.7,
        longitude=109.8,
        ensemble_mode="mean",
    )

    assert series.latitude == pytest.approx(20.0)
    assert series.longitude == pytest.approx(110.0)
    assert series.variable == "total_precipitation_6hr"
    assert series.display_units == "mm/6 h"
    np.testing.assert_array_equal(
        series.valid_times,
        forecast_dataset.coords["time"].values,
    )
    np.testing.assert_allclose(series.values, [9.0, 13.0])


def test_wn2_raster_bins_are_fixed_physical_thresholds() -> None:
    assert WN2_6H_SCALE.edges == (
        0.0,
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        200.0,
    )
    assert WN2_6H_SCALE.units == "mm/6 h"

    indices = raster_bin_indices(
        np.array([np.nan, 0.0, 0.05, 0.1, 0.499, 0.5, 200.0, 201.0]),
        WN2_6H_SCALE,
    )

    np.testing.assert_array_equal(indices, [-1, -1, 0, 1, 1, 2, 10, 10])


def test_render_image_overlay_preserves_grid_geometry() -> None:
    field = xr.DataArray(
        [[1.0, np.nan], [10.0, 100.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [10.0, 20.0], "longitude": [100.0, 110.0]},
        attrs={"units": "mm/6 h"},
    )

    overlay = render_image_overlay(field, WN2_6H_SCALE)

    assert overlay.data_url.startswith("data:image/png;base64,")
    assert overlay.bounds == ((5.0, 95.0), (25.0, 115.0))
    assert overlay.width == 2
    assert overlay.height == 2
    assert overlay.no_data_count == 1
    assert overlay.scale is WN2_6H_SCALE
