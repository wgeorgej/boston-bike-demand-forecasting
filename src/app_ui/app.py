import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import math
from dash import Dash, Input, Output, State, dash_table, dcc, html

from .utils import load_dashboard_data, load_ui_config


COLORS = {
    "navy": "#102A43",
    "teal": "#0F9D91",
    "coral": "#E07A83",
    "muted": "#627D98",
}

UI_CONFIG = load_ui_config()

_DATA_CACHE = {
    "loaded_at": 0.0,
    "predictions": None,
    "observed": None,
}

APP_DIR = Path(__file__).resolve().parent

app = Dash(
    __name__,
    assets_folder=str(APP_DIR / "assets"),
    title="Blue Bike Demand Forecast",
)
server = app.server


def get_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache parquet data and reload it at the configured interval."""
    now = time.monotonic()
    refresh_seconds = UI_CONFIG["update_interval_ms"] / 1000

    if (
        _DATA_CACHE["predictions"] is None
        or _DATA_CACHE["observed"] is None
        or now - _DATA_CACHE["loaded_at"] >= refresh_seconds
    ):
        predictions, observed = load_dashboard_data()
        _DATA_CACHE["predictions"] = predictions
        _DATA_CACHE["observed"] = observed
        _DATA_CACHE["loaded_at"] = now

    return _DATA_CACHE["predictions"], _DATA_CACHE["observed"]


def filter_data_by_period(
    predictions: pd.DataFrame,
    observed: pd.DataFrame,
    year: int,
    month: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter predictions and observed demand by year and optional month."""
    selected_year = int(year)

    prediction_times = pd.to_datetime(predictions["forecast_timestamp"])
    observed_times = pd.to_datetime(observed["datetime"])

    prediction_mask = prediction_times.dt.year == selected_year
    observed_mask = observed_times.dt.year == selected_year

    # Month value 0 means all months in the selected year.
    if month:
        selected_month = int(month)
        prediction_mask &= prediction_times.dt.month == selected_month
        observed_mask &= observed_times.dt.month == selected_month

    filtered_predictions = predictions.loc[prediction_mask].copy()
    filtered_observed = observed.loc[observed_mask].copy()

    return (
        filtered_predictions.reset_index(drop=True),
        filtered_observed.reset_index(drop=True),
    )


def to_epoch_ms(value: pd.Timestamp) -> int:
    """Convert a timestamp to milliseconds without local timezone offsets."""
    return int(pd.Timestamp(value).value // 1_000_000)


def make_card(
    label: str,
    value: str,
    note: str,
    accent: str = "navy",
) -> html.Div:
    """Create a dashboard summary card."""
    return html.Div(
        [
            html.Div(label, className="metric-label"),
            html.Div(value, className=f"metric-value {accent}"),
            html.Div(note, className="metric-note"),
        ],
        className="metric-card",
    )


def build_cards(
    predictions: pd.DataFrame,
    observed: pd.DataFrame,
    row_index: int,
) -> list[html.Div]:
    """Create summary cards for the current replay position."""
    current = predictions.iloc[row_index]
    data_through = observed["datetime"].max()

    return [
        make_card(
            "Forecast at replay position",
            f"{current['prediction']:,.0f} riders",
            current["forecast_timestamp"].strftime("%d %b %Y · %H:%M"),
            "teal",
        ),
        make_card(
            "Input time",
            current["timestamp"].strftime("%d %b · %H:%M"),
            "Observed hour used by the model",
        ),
        make_card(
            "Inference data through",
            data_through.strftime("%d %b %Y"),
            data_through.strftime("%H:%M"),
        ),
        make_card(
            "Registered model",
            "Version 2",
            "Boston_Bike_Demand_Model",
        ),
    ]


def build_forecast_table(
    predictions: pd.DataFrame,
    row_index: int,
) -> list[dict]:
    """Build the recent-forecasts table at the current replay position."""
    start_index = max(0, row_index - 7)
    recent = predictions.iloc[start_index : row_index + 1].copy()
    recent = recent.sort_values("forecast_timestamp", ascending=False)

    recent["timestamp"] = recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    recent["forecast_timestamp"] = recent["forecast_timestamp"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    recent["prediction"] = recent["prediction"].round(1)

    return recent[
        ["timestamp", "forecast_timestamp", "prediction"]
    ].to_dict("records")


app.layout = html.Div(
    [
        dcc.Interval(
            id="playback-tick",
            interval=UI_CONFIG["playback_interval_ms"],
            n_intervals=0,
            disabled=True,
        ),
        dcc.Interval(
            id="animation-tick",
            interval=UI_CONFIG["animation_tick_ms"],
            n_intervals=0,
            disabled=True,
        ),
        dcc.Store(id="chart-data"),
        dcc.Store(id="playback-config"),
        html.Header(
            [
                html.Div(
                    [
                        html.Div(
                            "BOSTON · MOBILITY ANALYTICS",
                            className="eyebrow",
                        ),
                        html.H1("Blue Bike Demand Forecast"),
                        html.P(
                            "One-hour-ahead rider demand predictions "
                            "from historical data"
                        ),
                    ],
                    className="brand",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Year", htmlFor="year-select"),
                                dcc.Dropdown(
                                    id="year-select",
                                    options=[
                                        {"label": "2022", "value": 2022},
                                        {"label": "2023", "value": 2023},
                                        {"label": "2024", "value": 2024},
                                    ],
                                    value=2024,
                                    clearable=False,
                                    searchable=False,
                                ),
                            ],
                            className="range-control",
                        ),
                        html.Div(
                            [
                                html.Label("Month", htmlFor="month-select"),
                                dcc.Dropdown(
                                    id="month-select",
                                    options=[
                                        {"label": "All months", "value": 0},
                                        {"label": "January", "value": 1},
                                        {"label": "February", "value": 2},
                                        {"label": "March", "value": 3},
                                        {"label": "April", "value": 4},
                                        {"label": "May", "value": 5},
                                        {"label": "June", "value": 6},
                                        {"label": "July", "value": 7},
                                        {"label": "August", "value": 8},
                                        {"label": "September", "value": 9},
                                        {"label": "October", "value": 10},
                                        {"label": "November", "value": 11},
                                        {"label": "December", "value": 12},
                                    ],
                                    value=0,
                                    clearable=False,
                                    searchable=False,
                                ),
                            ],
                            className="range-control",
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Chart range",
                                    htmlFor="period-select",
                                ),
                                dcc.Dropdown(
                                    id="period-select",
                                    options=[
                                        {"label": "Last 24 hours", "value": 24},
                                        {"label": "Last 7 days", "value": 168},
                                        {"label": "Last 30 days", "value": 720},
                                    ],
                                    value=UI_CONFIG["default_lookback_hours"],
                                    clearable=False,
                                    searchable=False,
                                ),
                            ],
                            className="range-control",
                        ),
                        html.Button(
                            "Play",
                            id="playback-toggle",
                            n_clicks=0,
                            className="playback-button",
                        ),
                    ],
                    className="header-controls",
                ),
            ],
            className="page-header",
        ),
        html.Div(
            "HISTORICAL REPLAY · 1 SECOND = 1 DATA HOUR",
            className="replay-status",
        ),
        html.Section(id="summary-cards", className="metric-grid"),
        html.Div(
    [
        # Chart panel
        html.Section(
            [
                html.Div(
                    [
                        html.H2("Real-Time Total Riders Prediction"),
                        html.Div(
                            [
                                html.Span(
                                    [
                                        html.Span(className="legend-key actual-key"),
                                        "Actual",
                                    ],
                                    className="legend-item",
                                ),
                                html.Span(
                                    [
                                        html.Span(className="legend-key predicted-key"),
                                        "Predicted",
                                    ],
                                    className="legend-item",
                                ),
                            ],
                            className="static-chart-legend",
                        ),
                    ],
                    className="panel-heading chart-panel-heading",
                ),
                dcc.Graph(
                    id="demand-chart",
                    style={"height": "100%", "width": "100%"},
                    figure=go.Figure(),
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                    },
                ),
            ],
            className="panel chart-panel",
        ),

        # Recent forecasts panel
        html.Section(
            [
                html.Div(
                    [
                        html.H2("Recent forecasts"),
                        html.P(
                            "Predictions up to the current replay position."
                        ),
                    ],
                    className="panel-heading table-heading",
                ),
                dash_table.DataTable(
                    id="forecasts-table",
                    columns=[
                        {"name": "Input time", "id": "timestamp"},
                        {
                            "name": "Forecast time",
                            "id": "forecast_timestamp",
                        },
                        {
                            "name": "Predicted riders",
                            "id": "prediction",
                        },
                    ],
                    page_size=10,
                    sort_action="native",
                    style_table={
                        "overflowX": "auto",
                        "overflowY": "auto",
                        "maxHeight": "100%",
                    },
                    style_header={
                        "backgroundColor": "#F4F7FA",
                        "color": COLORS["muted"],
                        "fontWeight": "700",
                        "border": "none",
                    },
                    style_cell={
                        "padding": "9px 10px",
                        "textAlign": "left",
                        "border": "none",
                        "color": "#243B53",
                        "fontSize": "12px",
                    },
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "#F8FAFC",
                        }
                    ],
                ),
            ],
            className="panel forecast-panel",
        ),
    ],
    className="main-grid",
),
        
html.Footer(
    "The replay uses the saved inference batch. "
    "It is not a live data feed.",
    className="page-footer",
),
],
className="page-shell",
)


@app.callback(
    Output("playback-tick", "disabled"),
    Output("animation-tick", "disabled"),
    Output("playback-toggle", "children"),
    Input("playback-toggle", "n_clicks"),
    State("playback-tick", "disabled"),
    prevent_initial_call=True,
)
def toggle_playback(_clicks: int, currently_disabled: bool):
    """Pause or resume the replay timers."""
    new_disabled = not currently_disabled
    return new_disabled, new_disabled, ("Play" if new_disabled else "Pause")


@app.callback(
    Output("chart-data", "data"),
    Output("playback-config", "data"),
    Input("year-select", "value"),
    Input("month-select", "value"),
    Input("period-select", "value"),
    State("playback-tick", "n_intervals"),
    State("animation-tick", "n_intervals"),
)
def build_chart(
    year: int,
    month: int,
    hours: int,
    playback_tick_origin: int,
    animation_tick_origin: int,
):
    """Build the chart for the selected year and month."""
    predictions, observed = get_dashboard_data()
    predictions, observed = filter_data_by_period(
        predictions,
        observed,
        year,
        month,
    )

    predictions = predictions.sort_values(
        "forecast_timestamp"
    ).reset_index(drop=True)
    observed = observed.sort_values("datetime").reset_index(drop=True)

    if predictions.empty or observed.empty:
        return go.Figure().to_dict(), None

    window_size = min(max(int(hours), 1), len(predictions))
    initial_row_index = window_size - 1
    initial_playhead = predictions.iloc[initial_row_index]["forecast_timestamp"]
    initial_input_time = predictions.iloc[initial_row_index]["timestamp"]

    chart_start = initial_playhead - pd.Timedelta(hours=int(hours))
    data_start = min(
        observed["datetime"].min(),
        predictions["forecast_timestamp"].min(),
    )
    data_end = predictions["forecast_timestamp"].max()
    chart_signature = (
    f"{year}:{month}:{hours}:"
    f"{to_epoch_ms(initial_playhead)}:"
    f"{to_epoch_ms(data_end)}"
)
    lookback_ms = int(hours) * 60 * 60 * 1000

    time_differences = (
        predictions["forecast_timestamp"].diff().dropna().dt.total_seconds()
    )
    if time_differences.empty:
        row_duration_ms = 60 * 60 * 1000
    else:
        row_duration_ms = max(1, int(time_differences.median() * 1000))

    all_values = pd.concat(
        [
            observed["total_riders"],
            predictions["prediction"],
        ],
        ignore_index=True,
    ).dropna()

    # Use a tighter percentile so occasional peaks don't flatten the chart.
    max_value = max(float(all_values.max()), 0.0)
    y_axis_max = max(500, math.ceil(max_value / 500) * 500)

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=observed["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            y=observed["total_riders"].astype(float).tolist(),
            mode="lines+markers",
            name="Actual",
            line={"color": COLORS["coral"], "width": 2},
            marker={
                "symbol": "x",
                "size": 8,
                "color": COLORS["coral"],
            },
            hovertemplate=(
                "%{x|%d %b %H:%M}<br>"
                "Actual: %{y:,.0f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scattergl(
            x=predictions["forecast_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            y=predictions["prediction"].astype(float).tolist(),
            mode="lines+markers",
            name="Predicted",
            line={"color": COLORS["teal"], "width": 2},
            marker={
                "symbol": "circle",
                "size": 6,
                "color": COLORS["teal"],
            },
            hovertemplate=(
                "%{x|%d %b %H:%M}<br>"
                "Predicted: %{y:,.1f}<extra></extra>"
            ),
        )
    )
    # Pin the forecast cutoff to a fixed screen position: one data hour
    # before the right edge of the selected time window.
    marker_position = max(0.0, min(1.0, (int(hours) - 1) / int(hours)))
    figure.add_shape(
        type="line",
        xref="paper",
        x0=marker_position,
        x1=marker_position,
        y0=0,
        y1=1,
        yref="paper",
        line={"color": "#829AB1", "width": 2.5, "dash": "dot"},
    )
    figure.update_layout(
        template="plotly_white",
        autosize=True,
        margin={"l": 65, "r": 20, "t": 30, "b": 75},
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right", "yanchor": "bottom",},
        hovermode="x unified",
        yaxis_title="Riders per hour",
        showlegend=False,
        xaxis={
            "range": [chart_start, initial_playhead],
            "title": {"text": "Time", "standoff": 0},
            "showgrid": False,
        },
        yaxis={
            "range": [0, y_axis_max * 1.05],
            "dtick": 500,
            "gridcolor": "#EEF2F6",
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={
            "family": "Inter, Segoe UI, sans-serif",
            "color": COLORS["muted"],
        },
        meta={"chart_signature": chart_signature},
        uirevision=chart_signature,
    )

    playback_config = {
        "year": int(year),
        "month": int(month),
        "forecast_times_ms": [
            to_epoch_ms(value)
            for value in predictions["forecast_timestamp"]
        ],
        "input_times_ms": [
            to_epoch_ms(value)
            for value in predictions["timestamp"]
        ],
        "start_ms": to_epoch_ms(initial_playhead),
        "end_ms": to_epoch_ms(data_end),
        "data_start_ms": to_epoch_ms(data_start),
        "lookback_ms": lookback_ms,
        "row_duration_ms": row_duration_ms,
        "playback_interval_ms": int(UI_CONFIG["playback_interval_ms"]),
        "animation_tick_ms": int(UI_CONFIG["animation_tick_ms"]),
        "initial_row_index": initial_row_index,
        "playback_tick_origin": int(playback_tick_origin or 0),
        "animation_tick_origin": int(animation_tick_origin or 0),
        "signature": chart_signature,
    }

    return figure.to_dict(), playback_config


app.clientside_callback(
    """
    function(chartData, playbackTicks, config, currentFigure) {
        const noUpdate = window.dash_clientside.no_update;

        if (!chartData) {
            return noUpdate;
        }

        const triggeredId =
            window.dash_clientside.callback_context.triggered_id;

        const incomingRevision =
            chartData.layout && chartData.layout.uirevision
                ? chartData.layout.uirevision
                : null;

        const displayedRevision =
            currentFigure
            && currentFigure.layout
            && currentFigure.layout.uirevision
                ? currentFigure.layout.uirevision
                : null;

        const hasDisplayedFigure =
            currentFigure
            && currentFigure.data
            && currentFigure.data.length > 0;

        if (!hasDisplayedFigure) {
            return chartData;
        }

        // A changed selection gets a new base figure.
        if (triggeredId === "chart-data") {
            if (
                incomingRevision
                && incomingRevision === displayedRevision
            ) {
                return noUpdate;
            }
            return chartData;
        }

        if (triggeredId !== "playback-tick") {
            return noUpdate;
        }

        if (
            !config
            || playbackTicks === undefined
            || playbackTicks === null
            || !config.forecast_times_ms
            || !config.input_times_ms
        ) {
            return noUpdate;
        }

        const elapsedRows = Math.max(
            0,
            playbackTicks - config.playback_tick_origin
        );

        const rowIndex = Math.min(
            config.initial_row_index + elapsedRows,
            config.forecast_times_ms.length - 1
        );

        const playheadMs = config.forecast_times_ms[rowIndex];
        const inputTimeMs = config.input_times_ms[rowIndex];
        // Keep the full lookback width constant so the paper-anchored marker
        // remains aligned with the current input point as time scrolls.
        const chartStartMs = playheadMs - config.lookback_ms;

        const chartStart = new Date(chartStartMs)
            .toISOString()
            .slice(0, -1);
        const chartEnd = new Date(playheadMs)
            .toISOString()
            .slice(0, -1);
        const inputTime = new Date(inputTimeMs)
            .toISOString()
            .slice(0, -1);

        // Figure timestamps use whole seconds; normalize the keys so exact
        // boundary points are included in the visible window.
        const startKey = chartStart.slice(0, 19);
        const endKey = chartEnd.slice(0, 19);
        const inputKey = inputTime.slice(0, 19);

        function lowerBound(values, key) {
            let left = 0;
            let right = values.length;
            while (left < right) {
                const middle = Math.floor((left + right) / 2);
                if (values[middle] < key) {
                    left = middle + 1;
                } else {
                    right = middle;
                }
            }
            return left;
        }

        function upperBound(values, key) {
            let left = 0;
            let right = values.length;
            while (left < right) {
                const middle = Math.floor((left + right) / 2);
                if (values[middle] <= key) {
                    left = middle + 1;
                } else {
                    right = middle;
                }
            }
            return left;
        }

        const actualX = chartData.data[0].x || [];
        const actualY = chartData.data[0].y || [];
        const predictionX = chartData.data[1].x || [];
        const predictionY = chartData.data[1].y || [];

        const actualStart = lowerBound(actualX, startKey);
        const actualEnd = upperBound(actualX, inputKey);
        const predictionStart = lowerBound(predictionX, startKey);
        const predictionEnd = upperBound(predictionX, endKey);

        const visibleActualX = actualX.slice(actualStart, actualEnd);
        const visibleActualY = actualY.slice(actualStart, actualEnd);
        const visiblePredictionX = predictionX.slice(
            predictionStart,
            predictionEnd
        );
        const visiblePredictionY = predictionY.slice(
            predictionStart,
            predictionEnd
        );

        // Return a fresh figure for the current window. The dotted marker is
        // anchored in paper coordinates and intentionally is not moved here.
        const nextFigure = JSON.parse(JSON.stringify(chartData));
        nextFigure.data[0].x = visibleActualX;
        nextFigure.data[0].y = visibleActualY;
        nextFigure.data[1].x = visiblePredictionX;
        nextFigure.data[1].y = visiblePredictionY;
        nextFigure.layout.xaxis.range = [chartStart, chartEnd];

        return nextFigure;
    }
    """,
    Output("demand-chart", "figure"),
    Input("chart-data", "data"),
    Input("playback-tick", "n_intervals"),
    State("playback-config", "data"),
    State("demand-chart", "figure"),
)

@app.callback(
    Output("summary-cards", "children"),
    Output("forecasts-table", "data"),
    Input("year-select", "value"),
    Input("month-select", "value"),
    Input("period-select", "value"),
    Input("playback-tick", "n_intervals"),
    Input("playback-config", "data"),
)
def update_summary(
    year: int,
    month: int,
    hours: int,
    playback_ticks: int,
    playback_config: dict | None,
):
    """Update the summary cards and recent forecasts for replay position."""
    predictions, observed = get_dashboard_data()

    predictions, observed = filter_data_by_period(
        predictions,
        observed,
        year,
        month,
    )

    predictions = predictions.sort_values(
        "forecast_timestamp"
    ).reset_index(drop=True)

    if predictions.empty or observed.empty:
        return [], []

    if (
        playback_config
        and playback_config.get("year") == int(year)
        and playback_config.get("month") == int(month)
    ):
        initial_row_index = playback_config["initial_row_index"]
        tick_origin = playback_config["playback_tick_origin"]
        elapsed_rows = max(0, int(playback_ticks or 0) - tick_origin)
        row_index = min(
            initial_row_index + elapsed_rows,
            len(predictions) - 1,
        )
    else:
        window_size = min(max(int(hours), 1), len(predictions))
        row_index = window_size - 1

    return (
        build_cards(predictions, observed, row_index),
        build_forecast_table(predictions, row_index),
    )

if __name__ == "__main__":
    app.run(debug=False)