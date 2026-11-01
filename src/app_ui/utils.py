from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMETERS_PATH = PROJECT_ROOT / "conf" / "base" / "parameters.yml"


def load_ui_config() -> dict:
    """Load dashboard settings from Kedro's base parameters file."""
    with PARAMETERS_PATH.open("r", encoding="utf-8") as file:
        parameters = yaml.safe_load(file) or {}

    ui_config = parameters.get("ui")
    if not ui_config:
        raise ValueError(f"No 'ui' section found in {PARAMETERS_PATH}")

    return ui_config


def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load predictions and observed inference data for the dashboard."""
    ui_config = load_ui_config()

    predictions_path = (PROJECT_ROOT / ui_config["predictions_path"]).resolve()
    actual_data_path = (PROJECT_ROOT / ui_config["actual_data_path"]).resolve()

    missing_files = [
        path for path in (predictions_path, actual_data_path) if not path.exists()
    ]
    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Dashboard data file(s) not found:\n{missing}")

    predictions = pd.read_parquet(predictions_path)
    observed = pd.read_parquet(actual_data_path)

    required_prediction_columns = {
        "timestamp",
        "forecast_timestamp",
        "prediction",
    }
    required_observed_columns = {"datetime", "total_riders"}

    if not required_prediction_columns.issubset(predictions.columns):
        raise ValueError(
            "Predictions data must contain columns: "
            f"{sorted(required_prediction_columns)}"
        )
    if not required_observed_columns.issubset(observed.columns):
        raise ValueError(
            "Inference data must contain columns: "
            f"{sorted(required_observed_columns)}"
        )

    predictions = predictions.copy()
    observed = observed.copy()

    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    predictions["forecast_timestamp"] = pd.to_datetime(
        predictions["forecast_timestamp"]
    )
    observed["datetime"] = pd.to_datetime(observed["datetime"])

    predictions = predictions.sort_values("forecast_timestamp").reset_index(drop=True)
    observed = observed.sort_values("datetime").reset_index(drop=True)

    return predictions, observed