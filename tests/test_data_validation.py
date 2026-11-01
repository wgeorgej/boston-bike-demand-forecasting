import pandas as pd
import pytest

from seoul_bike.pipelines.nodes import validate_bike_data


def make_bike_data(rows: int = 3) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01 00:00", periods=rows, freq="h")
    return pd.DataFrame(
        {
            "datetime": timestamps,
            "hour": timestamps.hour,
            "day": timestamps.day,
            "weekday": timestamps.dayofweek,
            "weekend": (timestamps.dayofweek >= 5).astype(int),
            "holiday": 0,
            "month": timestamps.month,
            "year": timestamps.year,
            "temp_c": 5.0,
            "humidity": 50.0,
            "wind_kmh": 10.0,
            "pressure_hpa": 1010.0,
            "weather": "Clear",
            "total_duration": 100.0,
            "total_riders": [10, 11, 12][:rows],
        }
    )


def test_valid_data_is_passed_through_unchanged():
    data = make_bike_data()

    result = validate_bike_data(data)

    assert result is not data
    pd.testing.assert_frame_equal(result, data)


def test_missing_required_column_is_rejected():
    data = make_bike_data().drop(columns="weather")

    with pytest.raises(ValueError, match="missing required columns.*weather"):
        validate_bike_data(data)


def test_invalid_datetime_is_rejected():
    data = make_bike_data()
    data["datetime"] = data["datetime"].astype(object)
    data.loc[1, "datetime"] = "not-a-datetime"

    with pytest.raises(ValueError):
        validate_bike_data(data)


def test_duplicate_timestamps_are_rejected():
    data = make_bike_data()
    data.loc[1, "datetime"] = data.loc[0, "datetime"]

    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_bike_data(data)


def test_non_finite_numeric_values_are_rejected():
    data = make_bike_data()
    data.loc[1, "humidity"] = float("inf")

    with pytest.raises(ValueError, match="non-finite"):
        validate_bike_data(data)


def test_negative_rider_count_is_rejected():
    data = make_bike_data()
    data.loc[1, "total_riders"] = -1

    with pytest.raises(ValueError, match="cannot be negative"):
        validate_bike_data(data)


def test_timestamp_gap_is_reported_without_changing_rows(capsys):
    data = make_bike_data()
    data.loc[2, "datetime"] = data.loc[1, "datetime"] + pd.Timedelta(hours=2)

    result = validate_bike_data(data)

    assert "Found 1 timestamp gap" in capsys.readouterr().out
    pd.testing.assert_frame_equal(result, data)


