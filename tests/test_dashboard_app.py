from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from weathernext_local.dashboard.app import create_app
from weathernext_local.dashboard.data import (
    DashboardDataError,
    inspect_dataset,
    select_precipitation,
    summarize_precipitation,
)
from weathernext_local.dashboard.raster import HOURLY_RATE_SCALE, scale_for_precipitation


def _nowcast_dataset(units: str = "kg m-2 h-1") -> xr.Dataset:
    values = np.array(
        [[[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]],
        dtype=np.float32,
    )
    dataset = xr.Dataset(
        {
            "precipitation_rate": (
                ("sample", "valid_time", "latitude", "longitude"),
                values,
                {"units": units},
            )
        },
        coords={
            "sample": [0],
            "valid_time": np.array(
                ["2026-08-09T00:00", "2026-08-09T00:15"], dtype="datetime64[ns]"
            ),
            "latitude": [25.0, 26.0],
            "longitude": [-81.0, -80.0],
        },
    )
    dataset.coords["latitude"].attrs["units"] = "degrees_north"
    dataset.coords["longitude"].attrs["units"] = "degrees_east"
    return dataset


def test_empty_app_is_explicit_and_contains_no_weather_placeholders(tmp_path) -> None:
    app = create_app(tmp_path / "runs")
    client = app.server.test_client()

    response = client.get("/_dash-layout")

    assert response.status_code == 200
    payload = response.get_json()
    serialized = json.dumps(payload)
    assert "Generate or import an authorized forecast first" in serialized
    assert "never substitutes sample or synthetic weather data" in serialized
    assert "not an official weather forecast" in serialized
    assert "data/runs/&lt;run-id&gt;/manifest.json" not in serialized
    assert "manifest.json" in serialized


def test_authorized_nowcast_rate_uses_explicit_hourly_units_and_fixed_scale() -> None:
    dataset = _nowcast_dataset()
    manifest = {
        "primary_precipitation_variable": {
            "variable": "precipitation_rate",
            "units": "kg m-2 h-1",
            "label": "precipitation rate",
        }
    }

    layout = inspect_dataset(dataset, manifest)
    field = select_precipitation(dataset, 1, manifest=manifest)

    assert layout.precipitation.display_units == "mm/h"
    assert layout.precipitation.scale_factor == pytest.approx(1.0)
    assert scale_for_precipitation(layout.precipitation) is HOURLY_RATE_SCALE
    np.testing.assert_allclose(field.values, [[4.0, 5.0], [6.0, 7.0]])


def test_declared_fifteen_minute_accumulation_gets_interval_specific_scale() -> None:
    dataset = _nowcast_dataset(units="kg m-2").rename({"precipitation_rate": "precipitation"})
    manifest = {
        "primary_precipitation_variable": {
            "variable": "precipitation",
            "units": "kg m-2",
            "interval": "15 min",
            "label": "15-minute accumulated precipitation",
        }
    }

    layout = inspect_dataset(dataset, manifest)
    scale = scale_for_precipitation(layout.precipitation)

    assert layout.precipitation.display_units == "mm/15 min"
    assert scale.units == "mm/15 min"
    assert scale.edges[1] == pytest.approx(0.025)


def test_render_callback_uses_run_values_for_map_kpis_and_clicked_series(tmp_path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "authorized-nowcast"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "authorized-nowcast",
        "model": {"id": "test-contract", "family": "nowcast"},
        "primary_precipitation_variable": {
            "variable": "precipitation_rate",
            "units": "kg m-2 h-1",
            "label": "precipitation rate",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _nowcast_dataset().to_netcdf(run_dir / "forecast.nc", engine="h5netcdf")
    app = create_app(runs_root)
    callback = next(
        metadata["callback"].__wrapped__
        for output, metadata in app.callback_map.items()
        if "forecast-map-layers.children" in output
    )

    result = callback(
        "authorized-nowcast",
        1,
        "mean",
        0,
        {"latlng": {"lat": 25.8, "lng": -80.2}},
    )

    assert len(result[0]) == 2  # ImageOverlay plus selected-cell marker.
    assert result[2] == "precipitation rate · mm/h"
    assert result[3] == "2026-08-09 00:15"
    assert result[4] == "7.00 mm/h"
    assert result[9].data[0].y[-1] == pytest.approx(7.0)
    assert result[10] == "Nearest cell | 26.000 deg, -80.000 deg"
    assert result[11] == ""
    assert result[12] == ""


@pytest.mark.parametrize(
    ("coordinate", "values"),
    [("latitude", [91.0, 92.0]), ("longitude", [999.0, 1_000.0])],
)
def test_dashboard_rejects_out_of_range_spatial_coordinates(coordinate, values) -> None:
    dataset = _nowcast_dataset().assign_coords({coordinate: values})
    dataset.coords[coordinate].attrs["units"] = (
        "degrees_north" if coordinate == "latitude" else "degrees_east"
    )

    with pytest.raises(DashboardDataError, match=f"{coordinate} coordinate is outside"):
        inspect_dataset(dataset)


def test_dashboard_rejects_radian_spatial_units() -> None:
    dataset = _nowcast_dataset()
    dataset.coords["latitude"].attrs["units"] = "radians"

    with pytest.raises(DashboardDataError, match="not recognized degree units"):
        inspect_dataset(dataset)


def test_dashboard_rejects_south_positive_spatial_units() -> None:
    dataset = _nowcast_dataset()
    dataset.coords["latitude"].attrs["units"] = "degree_s"

    with pytest.raises(DashboardDataError, match="south/west-positive"):
        inspect_dataset(dataset)


def test_dashboard_rejects_complex_spatial_coordinates() -> None:
    dataset = _nowcast_dataset().assign_coords(
        latitude=np.array([25 + 1j, 26 + 2j], dtype=np.complex128)
    )
    dataset.coords["latitude"].attrs["units"] = "degrees_north"

    with pytest.raises(DashboardDataError, match="real numeric"):
        inspect_dataset(dataset)


def test_dashboard_rejects_all_missing_selected_field() -> None:
    dataset = _nowcast_dataset()
    dataset["precipitation_rate"][:] = np.nan
    field = select_precipitation(dataset, 0)

    with pytest.raises(DashboardDataError, match="no valid cells"):
        summarize_precipitation(field)


def test_dashboard_rejects_shared_spatial_dimension() -> None:
    dataset = xr.Dataset(
        {
            "precipitation_rate": (
                ("valid_time", "point"),
                np.ones((2, 2)),
                {"units": "mm/h"},
            )
        },
        coords={
            "valid_time": np.array(
                ["2026-08-09T00:00", "2026-08-09T00:15"], dtype="datetime64[ns]"
            ),
            "latitude": ("point", [25.0, 26.0]),
            "longitude": ("point", [-81.0, -80.0]),
        },
    )
    dataset.coords["latitude"].attrs["units"] = "degrees_north"
    dataset.coords["longitude"].attrs["units"] = "degrees_east"

    with pytest.raises(DashboardDataError, match="distinct dimensions"):
        inspect_dataset(dataset)


def test_dashboard_rejects_irregular_raster_axis() -> None:
    dataset = _nowcast_dataset().assign_coords(longitude=[-81.0, -79.25])
    dataset = dataset.reindex(longitude=[-81.0, -80.5, -79.25], fill_value=0.0)
    dataset.coords["longitude"].attrs["units"] = "degrees_east"

    with pytest.raises(DashboardDataError, match="regularly spaced"):
        inspect_dataset(dataset)
