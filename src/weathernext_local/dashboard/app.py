"""Dash application for exploring locally generated precipitation forecasts."""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import dash_leaflet as dl
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from .catalog import RunCatalog, RunRecord, discover_runs
from .data import (
    DashboardDataError,
    DatasetLayout,
    format_valid_time,
    inspect_dataset,
    nearest_cell_series,
    open_forecast,
    sample_options,
    select_precipitation,
    summarize_precipitation,
)
from .raster import FixedColorScale, render_image_overlay, scale_for_precipitation

_OSM_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
_OSM_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)


def _family_value(record: RunRecord) -> str:
    return record.model_family or "__unspecified__"


def _family_options(catalog: RunCatalog) -> list[dict[str, str]]:
    options = [{"label": "All model families", "value": "__all__"}]
    options.extend({"label": family, "value": family} for family in catalog.families)
    if any(record.model_family is None for record in catalog.runs):
        options.append({"label": "Unspecified in manifest", "value": "__unspecified__"})
    return options


def _matching_runs(catalog: RunCatalog, family: str | None) -> tuple[RunRecord, ...]:
    if not family or family == "__all__":
        return catalog.runs
    return tuple(record for record in catalog.runs if _family_value(record) == family)


def _run_options(records: tuple[RunRecord, ...]) -> list[dict[str, str]]:
    return [{"label": record.display_label, "value": record.run_id} for record in records]


def _empty_figure(message: str = "Click the map to inspect the nearest grid cell") -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14, "color": "#60758a"},
    )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 46, "r": 20, "t": 18, "b": 42},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,251,254,0.72)",
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=305,
    )
    return figure


def _time_marks(layout: DatasetLayout) -> dict[int, dict[str, Any]]:
    size = layout.time_size
    indices = list(range(size)) if size <= 6 else sorted({0, size // 2, size - 1})
    return {
        index: {
            "label": format_valid_time(layout.valid_times[index]),
            "style": {"whiteSpace": "nowrap"},
        }
        for index in indices
    }


def _legend(scale: FixedColorScale) -> html.Div:
    rows: list[Any] = [
        html.Div(
            [html.Span(className="legend-swatch dry-swatch"), html.Span("0 · dry")],
            className="legend-item",
        )
    ]
    rows.extend(
        html.Div(
            [
                html.Span(
                    className="legend-swatch",
                    style={"backgroundColor": color},
                ),
                html.Span(label),
            ],
            className="legend-item",
        )
        for color, label in zip(scale.colors, scale.labels, strict=True)
    )
    rows.append(
        html.Div(
            [html.Span(className="legend-swatch missing-swatch"), html.Span("Missing")],
            className="legend-item",
        )
    )
    return html.Div(
        [
            html.Div(
                [
                    html.Strong("Fixed precipitation scale"),
                    html.Span(scale.units, className="unit-chip"),
                ],
                className="legend-title",
            ),
            html.Div(rows, className="legend-grid"),
        ],
        className="map-legend",
    )


def _manifest_value(manifest: Mapping[str, Any], *path: str) -> object:
    value: object = manifest
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _provenance_panel(
    record: RunRecord,
    layout: DatasetLayout | None,
    selection_note: str | None = None,
) -> html.Div:
    artifact_hash = _manifest_value(record.manifest, "artifact", "sha256")
    schema_id = record.manifest.get("schema_id")
    rows: list[tuple[str, str]] = [
        ("Run", record.run_id),
        ("Model family", record.model_family or "Not declared"),
        ("Model", record.model_id or "Not declared"),
        ("Manifest", str(record.manifest_path)),
        ("Forecast artifact", str(record.forecast_path)),
    ]
    if schema_id is not None:
        rows.append(("Manifest schema", str(schema_id)))
    if artifact_hash is not None:
        rows.append(("Manifest-declared SHA-256", str(artifact_hash)))
    if layout:
        spec = layout.precipitation
        rows.extend(
            [
                ("Displayed variable", spec.variable),
                ("Source units", spec.source_units),
                (
                    "Display conversion",
                    f"source x {spec.scale_factor:g} -> {spec.display_units}",
                ),
                ("Valid-time coordinate", layout.time_coordinate),
            ]
        )
    if selection_note:
        rows.append(("Current ensemble view", selection_note))
    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.Span(label), html.Strong(value)], className="source-row")
                    for label, value in rows
                ],
                className="source-grid",
            ),
            html.Details(
                [
                    html.Summary("Raw manifest.json"),
                    html.Pre(
                        json.dumps(record.manifest, indent=2, sort_keys=True),
                        className="manifest-json",
                    ),
                ],
                className="manifest-details",
            ),
        ]
    )


def _selection_note(layout: DatasetLayout, ensemble_mode: str, member_index: int) -> str:
    if not layout.sample_dimension:
        return "Deterministic (no sample dimension)"
    if ensemble_mode == "member":
        return f"Member index {member_index} of {layout.sample_size}"
    labels = {
        "mean": "Ensemble mean",
        "median": "Ensemble median",
        "maximum": "Ensemble maximum",
    }
    return f"{labels.get(ensemble_mode, ensemble_mode)} across {layout.sample_size} members"


def _click_coordinates(click_data: object) -> tuple[float, float] | None:
    if not isinstance(click_data, Mapping):
        return None
    lat_lng = click_data.get("latlng")
    if isinstance(lat_lng, Mapping):
        latitude, longitude = lat_lng.get("lat"), lat_lng.get("lng")
    else:
        latitude, longitude = click_data.get("lat"), click_data.get("lng")
    if isinstance(latitude, (int, float)) and isinstance(longitude, (int, float)):
        return float(latitude), float(longitude)
    return None


def _map_marker(latitude: float, longitude: float, label: str) -> dl.CircleMarker:
    map_longitude = ((longitude + 180.0) % 360.0) - 180.0
    return dl.CircleMarker(
        center=[latitude, map_longitude],
        radius=6,
        color="#071e36",
        weight=2,
        fill=True,
        fillColor="#f9d65c",
        fillOpacity=1,
        children=[dl.Tooltip(label)],
    )


def _page_layout(catalog: RunCatalog) -> html.Div:
    family_options = _family_options(catalog)
    runs = _matching_runs(catalog, "__all__")
    selected_run = runs[0].run_id if runs else None
    has_runs = bool(runs)
    issue_text = ""
    if catalog.issues:
        issue_text = f"{len(catalog.issues)} incomplete or invalid run director"
        issue_text += "y was skipped." if len(catalog.issues) == 1 else "ies were skipped."

    return html.Div(
        [
            html.Header(
                [
                    html.Div(
                        [
                            html.Div("WN", className="brand-mark", **{"aria-hidden": "true"}),
                            html.Div(
                                [
                                    html.P("EXPERIMENTAL FORECAST EXPLORER", className="eyebrow"),
                                    html.H1("WeatherNext Local"),
                                    html.P(
                                        "Inspect precipitation exactly as produced by authorized local model runs.",
                                        className="header-subtitle",
                                    ),
                                ]
                            ),
                        ],
                        className="brand-lockup",
                    ),
                    html.Div(
                        [html.Span(className="status-dot"), "LOCAL · SOURCE-BACKED"],
                        className="status-pill",
                    ),
                ],
                className="app-header",
            ),
            html.Div(
                [
                    html.Div("!", className="warning-icon", **{"aria-hidden": "true"}),
                    html.Div(
                        [
                            html.Strong("Experimental output — not an official weather forecast."),
                            html.Span(
                                " Do not use this dashboard for life-safety, emergency, aviation, marine, or other operational decisions."
                            ),
                        ]
                    ),
                ],
                className="warning-banner",
                role="alert",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.Label("Model family", htmlFor="model-family-selector"),
                            dcc.Dropdown(
                                id="model-family-selector",
                                options=family_options,
                                value="__all__",
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                        className="control-field",
                    ),
                    html.Div(
                        [
                            html.Label("Model run", htmlFor="run-selector"),
                            dcc.Dropdown(
                                id="run-selector",
                                options=_run_options(runs),
                                value=selected_run,
                                clearable=False,
                                searchable=True,
                                disabled=not has_runs,
                                placeholder="No complete runs found",
                            ),
                        ],
                        className="control-field run-field",
                    ),
                    html.Button(
                        [html.Span("↻", **{"aria-hidden": "true"}), " Refresh runs"],
                        id="refresh-runs",
                        n_clicks=0,
                        className="refresh-button",
                        title="Rescan the local run directory",
                    ),
                    html.P(issue_text, id="catalog-issues", className="catalog-issues"),
                ],
                className="control-bar",
                **{"aria-label": "Forecast selection"},
            ),
            html.Section(
                [
                    html.Div(className="empty-orbit orbit-one"),
                    html.Div(className="empty-orbit orbit-two"),
                    html.Div(
                        [
                            html.P("NO COMPLETE MODEL RUNS", className="eyebrow"),
                            html.H2("Generate or import an authorized forecast first"),
                            html.P(
                                "This dashboard never substitutes sample or synthetic weather data when local output is absent."
                            ),
                            html.Ol(
                                [
                                    html.Li("Run an authorized WeatherNext or nowcast workflow."),
                                    html.Li(
                                        [
                                            "Confirm the run directory contains ",
                                            html.Code("data/runs/<run-id>/manifest.json"),
                                            " and ",
                                            html.Code("forecast.nc"),
                                            ".",
                                        ]
                                    ),
                                    html.Li("Select Refresh runs above."),
                                ]
                            ),
                            html.P(
                                f"Scanning: {catalog.root}",
                                className="scan-path",
                            ),
                        ],
                        className="empty-content",
                    ),
                ],
                id="empty-state",
                className="empty-state",
                style={"display": "none" if has_runs else "block"},
            ),
            html.Main(
                [
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.P("VALID TIME", className="kpi-label"),
                                    html.Div(
                                        "—", id="kpi-valid-time", className="kpi-value time-value"
                                    ),
                                    html.P("Dataset coordinate", className="kpi-detail"),
                                ],
                                className="kpi-card accent-card",
                            ),
                            html.Div(
                                [
                                    html.P("MAXIMUM", className="kpi-label"),
                                    html.Div("—", id="kpi-maximum", className="kpi-value"),
                                    html.P("Highest valid grid cell", className="kpi-detail"),
                                ],
                                className="kpi-card",
                            ),
                            html.Div(
                                [
                                    html.P(
                                        "AREA-WEIGHTED MEAN",
                                        id="kpi-mean-label",
                                        className="kpi-label",
                                    ),
                                    html.Div("—", id="kpi-mean", className="kpi-value"),
                                    html.P("Cos(latitude), valid cells", className="kpi-detail"),
                                ],
                                className="kpi-card",
                            ),
                            html.Div(
                                [
                                    html.P("WET AREA", className="kpi-label"),
                                    html.Div("—", id="kpi-wet-area", className="kpi-value"),
                                    html.P("—", id="kpi-wet-detail", className="kpi-detail"),
                                ],
                                className="kpi-card",
                            ),
                        ],
                        className="kpi-strip",
                        **{"aria-label": "Selected forecast summary"},
                    ),
                    html.Div(id="dashboard-notice", className="dashboard-notice", role="status"),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.P("FORECAST FIELD", className="eyebrow"),
                                            html.H2("Precipitation map"),
                                        ]
                                    ),
                                    html.Div("—", id="variable-badge", className="variable-badge"),
                                ],
                                className="panel-header",
                            ),
                            html.Div(
                                [
                                    dl.MapContainer(
                                        [
                                            dl.TileLayer(
                                                url=_OSM_URL,
                                                attribution=_OSM_ATTRIBUTION,
                                                maxZoom=18,
                                            ),
                                            dl.LayerGroup(id="forecast-map-layers"),
                                            dl.ScaleControl(position="bottomleft", imperial=False),
                                        ],
                                        id="forecast-map",
                                        center=[20, 0],
                                        zoom=2,
                                        minZoom=1,
                                        worldCopyJump=True,
                                        className="forecast-map",
                                    ),
                                    html.Div(id="map-legend"),
                                    html.Div(
                                        "Click anywhere to load the nearest model-grid time series",
                                        className="map-hint",
                                    ),
                                ],
                                className="map-wrap",
                            ),
                        ],
                        className="panel map-panel",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Label("Valid time", htmlFor="valid-time-slider"),
                                    dcc.Slider(
                                        id="valid-time-slider",
                                        min=0,
                                        max=0,
                                        step=1,
                                        value=0,
                                        marks={},
                                        disabled=True,
                                        updatemode="mouseup",
                                        tooltip={"placement": "bottom", "always_visible": False},
                                    ),
                                ],
                                className="timeline-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Ensemble view"),
                                    dcc.RadioItems(
                                        id="ensemble-statistic",
                                        options=[
                                            {"label": "Mean", "value": "mean"},
                                            {"label": "Median", "value": "median"},
                                            {"label": "Maximum", "value": "maximum"},
                                            {"label": "Member", "value": "member"},
                                        ],
                                        value="mean",
                                        inline=True,
                                        className="segmented-control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Member", htmlFor="ensemble-member"),
                                            dcc.Dropdown(
                                                id="ensemble-member",
                                                options=[],
                                                value=0,
                                                clearable=False,
                                                searchable=False,
                                            ),
                                        ],
                                        id="member-control",
                                        className="member-control",
                                        style={"display": "none"},
                                    ),
                                ],
                                id="ensemble-panel",
                                className="ensemble-panel",
                                style={"display": "none"},
                            ),
                        ],
                        className="panel interaction-panel",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.P("NEAREST GRID CELL", className="eyebrow"),
                                            html.H2("Precipitation through time"),
                                        ]
                                    ),
                                    html.P(
                                        "No map point selected",
                                        id="point-label",
                                        className="point-label",
                                    ),
                                ],
                                className="panel-header",
                            ),
                            dcc.Graph(
                                id="point-timeseries",
                                figure=_empty_figure(),
                                config={"displayModeBar": False, "responsive": True},
                            ),
                        ],
                        className="panel timeseries-panel",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.P("SOURCE & LINEAGE", className="eyebrow"),
                                    html.H2("Provenance"),
                                ],
                                className="panel-header",
                            ),
                            html.Div(id="provenance-content"),
                        ],
                        className="panel provenance-panel",
                    ),
                    html.Div(id="dashboard-error", className="dashboard-error", role="alert"),
                ],
                id="dashboard-workspace",
                className="dashboard-workspace",
                style={"display": "grid" if has_runs else "none"},
            ),
            html.Footer(
                [
                    html.Span("WeatherNext Local"),
                    html.Span(
                        "Values are read from local run artifacts; the basemap is OpenStreetMap."
                    ),
                ],
                className="app-footer",
            ),
        ],
        className="app-shell",
    )


def _record(data_root: Path, run_id: str | None) -> tuple[RunCatalog, RunRecord | None]:
    catalog = discover_runs(data_root)
    return catalog, catalog.get(run_id) if run_id else None


def create_app(data_root: str | Path = "data/runs") -> Dash:
    """Create the local dashboard without eagerly opening forecast datasets."""

    resolved_root = Path(data_root).expanduser().resolve()
    assets = Path(__file__).parent / "assets"
    app = Dash(
        __name__,
        title="WeatherNext Local · Experimental Forecast Explorer",
        assets_folder=str(assets),
        suppress_callback_exceptions=True,
    )
    app.layout = lambda: _page_layout(discover_runs(resolved_root))

    @app.callback(
        Output("model-family-selector", "options"),
        Output("run-selector", "options"),
        Output("run-selector", "value"),
        Output("run-selector", "disabled"),
        Output("empty-state", "style"),
        Output("dashboard-workspace", "style"),
        Output("catalog-issues", "children"),
        Input("model-family-selector", "value"),
        Input("refresh-runs", "n_clicks"),
        State("run-selector", "value"),
    )
    def refresh_runs(
        family: str | None, _refresh_clicks: int, current_run: str | None
    ) -> tuple[Any, ...]:
        catalog = discover_runs(resolved_root)
        records = _matching_runs(catalog, family)
        run_ids = {record.run_id for record in records}
        selected = (
            current_run if current_run in run_ids else (records[0].run_id if records else None)
        )
        issue_count = len(catalog.issues)
        issue_text = ""
        if issue_count:
            issue_text = f"{issue_count} incomplete or invalid run director"
            issue_text += "y was skipped." if issue_count == 1 else "ies were skipped."
        return (
            _family_options(catalog),
            _run_options(records),
            selected,
            not bool(records),
            {"display": "none" if records else "block"},
            {"display": "grid" if records else "none"},
            issue_text,
        )

    @app.callback(
        Output("valid-time-slider", "max"),
        Output("valid-time-slider", "value"),
        Output("valid-time-slider", "marks"),
        Output("valid-time-slider", "disabled"),
        Output("ensemble-panel", "style"),
        Output("ensemble-member", "options"),
        Output("ensemble-member", "value"),
        Output("provenance-content", "children"),
        Output("dashboard-error", "children", allow_duplicate=True),
        Input("run-selector", "value"),
        prevent_initial_call=True,
    )
    def configure_run(run_id: str | None) -> tuple[Any, ...]:
        _, record = _record(resolved_root, run_id)
        if not record:
            return 0, 0, {}, True, {"display": "none"}, [], 0, "", ""
        try:
            with open_forecast(record.forecast_path) as dataset:
                layout = inspect_dataset(dataset, record.manifest)
                options = [
                    {"label": label, "value": value}
                    for label, value in sample_options(dataset, layout)
                ]
            return (
                layout.time_size - 1,
                0,
                _time_marks(layout),
                False,
                {"display": "flex"} if layout.sample_dimension else {"display": "none"},
                options,
                0,
                _provenance_panel(record, layout),
                "",
            )
        except (DashboardDataError, OSError, ValueError) as exc:
            return (
                0,
                0,
                {},
                True,
                {"display": "none"},
                [],
                0,
                _provenance_panel(record, None),
                f"Cannot display run {record.run_id}: {exc}",
            )

    @app.callback(
        Output("member-control", "style"),
        Input("ensemble-statistic", "value"),
    )
    def show_member_control(ensemble_mode: str) -> dict[str, str]:
        return {"display": "block" if ensemble_mode == "member" else "none"}

    @app.callback(
        Output("forecast-map-layers", "children"),
        Output("map-legend", "children"),
        Output("variable-badge", "children"),
        Output("kpi-valid-time", "children"),
        Output("kpi-maximum", "children"),
        Output("kpi-mean-label", "children"),
        Output("kpi-mean", "children"),
        Output("kpi-wet-area", "children"),
        Output("kpi-wet-detail", "children"),
        Output("point-timeseries", "figure"),
        Output("point-label", "children"),
        Output("dashboard-notice", "children"),
        Output("dashboard-error", "children", allow_duplicate=True),
        Input("run-selector", "value"),
        Input("valid-time-slider", "value"),
        Input("ensemble-statistic", "value"),
        Input("ensemble-member", "value"),
        Input("forecast-map", "clickData"),
        prevent_initial_call=True,
    )
    def render_selection(
        run_id: str | None,
        time_index: int | None,
        ensemble_mode: str | None,
        member_index: int | None,
        click_data: object,
    ) -> tuple[Any, ...]:
        _, record = _record(resolved_root, run_id)
        if not record:
            return (
                [],
                "",
                "—",
                "—",
                "—",
                "AREA-WEIGHTED MEAN",
                "—",
                "—",
                "—",
                _empty_figure("Select a complete run"),
                "No map point selected",
                "",
                "",
            )
        index = int(time_index or 0)
        mode = ensemble_mode or "mean"
        member = int(member_index or 0)
        try:
            with open_forecast(record.forecast_path) as dataset:
                layout = inspect_dataset(dataset, record.manifest)
                field = select_precipitation(
                    dataset,
                    index,
                    ensemble_mode=mode,
                    member_index=member,
                    manifest=record.manifest,
                )
                scale = scale_for_precipitation(layout.precipitation)
                overlay = render_image_overlay(field, scale)
                summary = summarize_precipitation(field, wet_threshold=scale.wet_threshold)
                layers: list[Any] = [
                    dl.ImageOverlay(
                        url=overlay.data_url,
                        bounds=[list(overlay.bounds[0]), list(overlay.bounds[1])],
                        opacity=1,
                        interactive=False,
                    )
                ]
                point = _click_coordinates(click_data)
                if point:
                    series = nearest_cell_series(
                        dataset,
                        point[0],
                        point[1],
                        ensemble_mode=mode,
                        member_index=member,
                        manifest=record.manifest,
                    )
                    point_text = (
                        f"Nearest cell | {series.latitude:.3f} deg, {series.longitude:.3f} deg"
                    )
                    layers.append(_map_marker(series.latitude, series.longitude, point_text))
                    figure = go.Figure(
                        go.Scatter(
                            x=[format_valid_time(value) for value in series.valid_times],
                            y=list(series.values),
                            mode="lines+markers",
                            line={"color": "#1261a6", "width": 2.5},
                            marker={"color": "#0a4c87", "size": 6},
                            connectgaps=False,
                            hovertemplate=(
                                "%{x}<br><b>%{y:.3f} "
                                + series.display_units
                                + "</b><extra></extra>"
                            ),
                        )
                    )
                    figure.add_vline(
                        x=format_valid_time(layout.valid_times[index]),
                        line_color="#f0a202",
                        line_width=1.5,
                        line_dash="dot",
                    )
                    figure.update_layout(
                        template="plotly_white",
                        height=305,
                        margin={"l": 58, "r": 18, "t": 18, "b": 48},
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(248,251,254,0.72)",
                        hovermode="x unified",
                        xaxis={
                            "title": "Valid time",
                            "showgrid": False,
                            "automargin": True,
                        },
                        yaxis={
                            "title": series.display_units,
                            "rangemode": "tozero",
                            "gridcolor": "#dfe9f2",
                            "automargin": True,
                        },
                    )
                else:
                    figure = _empty_figure()
                    point_text = "No map point selected"

            units = layout.precipitation.display_units
            validity = format_valid_time(layout.valid_times[index])
            mean_label = (
                "GLOBAL AREA-WEIGHTED MEAN" if summary.is_global else "DOMAIN AREA-WEIGHTED MEAN"
            )
            wet_detail = f">= {scale.wet_threshold:g} {units} | valid-cell area"
            if summary.valid_area_percent < 99.999:
                wet_detail += f" · {summary.valid_area_percent:.1f}% area valid"
            notice = ""
            floored_count = int(field.attrs.get("display_floor_count", 0))
            if floored_count:
                raw_minimum = float(field.attrs["display_floor_minimum"])
                notice = (
                    f"Display-only adjustment: floored {floored_count:,} negative model cells "
                    f"to zero for this map and its KPIs (raw minimum {raw_minimum:.4g} {units}). "
                    "The forecast.nc artifact is unchanged; point time series retain raw values."
                )
            return (
                layers,
                _legend(scale),
                f"{layout.precipitation.label} · {units}",
                validity,
                f"{summary.maximum:.2f} {units}",
                mean_label,
                f"{summary.area_weighted_mean:.2f} {units}",
                f"{summary.wet_area_percent:.1f}%",
                wet_detail,
                figure,
                point_text,
                notice,
                "",
            )
        except (DashboardDataError, OSError, ValueError, IndexError) as exc:
            return (
                [],
                "",
                "Unavailable",
                "—",
                "—",
                "AREA-WEIGHTED MEAN",
                "—",
                "—",
                "—",
                _empty_figure("The selected forecast field could not be rendered"),
                "No map point selected",
                "",
                f"Cannot display run {record.run_id}: {exc}",
            )

    return app


def run_dashboard(
    data_root: str | Path = "data/runs",
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
) -> None:
    """Start the dashboard, never exposing the development debugger remotely."""

    normalized_host = host.strip()
    try:
        loopback = ipaddress.ip_address(normalized_host).is_loopback
    except ValueError:
        loopback = normalized_host.casefold() == "localhost"
    if debug and not loopback:
        raise ValueError("Dash debug mode is allowed only on a loopback host.")
    app = create_app(data_root)
    app.run(host=host, port=port, debug=debug)
