"""Data processing nodes for the Boston Blue Bike demand forecasting project."""

from typing import Dict,Any, Tuple, Union
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature
from pathlib import Path
import os


def validate_bike_data(data: pd.DataFrame) -> pd.DataFrame:
    """Check raw bike data and report timestamp gaps without changing rows."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Bike data input must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Bike data input is empty.")

    required_columns = {
        "datetime", "hour", "day", "weekday", "weekend", "holiday",
        "month", "year", "temp_c", "humidity", "wind_kmh",
        "pressure_hpa", "weather", "total_duration", "total_riders",
    }
    missing = sorted(required_columns.difference(data.columns))
    if missing:
        raise ValueError("Bike data is missing required columns: " + ", ".join(missing))

    parsed_times = pd.to_datetime(data["datetime"], errors="raise")
    if parsed_times.isna().any():
        raise ValueError("Bike data contains missing datetime values.")
    duplicate_times = parsed_times[parsed_times.duplicated()]
    if not duplicate_times.empty:
        raise ValueError(
            "Bike data contains duplicate timestamps: "
            f"{duplicate_times.astype(str).head(5).tolist()}"
        )

    numeric_columns = sorted(required_columns - {"datetime", "weather"})
    numeric_values: dict[str, pd.Series] = {}
    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values)
        if invalid.any():
            rows = data.index[invalid].tolist()[:5]
            raise ValueError(
                f"Bike data column '{column}' has missing, nonnumeric, or "
                f"non-finite values at row indices {rows}."
            )
        numeric_values[column] = values

    if data["weather"].isna().any():
        rows = data.index[data["weather"].isna()].tolist()[:5]
        raise ValueError(f"Bike data column 'weather' is missing at rows {rows}.")
    if (numeric_values["total_riders"] < 0).any():
        raise ValueError("Bike data column 'total_riders' cannot be negative.")

    ordered_times = parsed_times.sort_values(kind="stable").reset_index(drop=True)
    intervals = ordered_times.diff()
    gap_positions = np.flatnonzero((intervals > pd.Timedelta(hours=1)).to_numpy())
    if len(gap_positions):
        examples = [
            f"{ordered_times.iloc[position - 1]} -> {ordered_times.iloc[position]} "
            f"({intervals.iloc[position]})"
            for position in gap_positions[:5]
        ]
        print(
            f"WARNING: Found {len(gap_positions)} timestamp gap(s) longer than "
            f"one hour; largest interval is {intervals.max()}. "
            f"Examples: {examples}. Input rows are passed through unchanged."
        )

    return data.copy()

def prepare_inference_batch(
    inference_data: pd.DataFrame,
    year: int,
    lag_params: dict[str, list[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a year-only batch and a warm-up context for lag features.

    The saved batch contains only the requested year. The transient context
    also includes the preceding rows required to calculate lag features for
    the first observations in that year.
    """
    if "datetime" not in inference_data.columns:
        raise ValueError("Inference data must contain a 'datetime' column.")

    data = inference_data.copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    data = data.sort_values("datetime", kind="stable").reset_index(drop=True)

    year_mask = data["datetime"].dt.year.eq(int(year))
    year_positions = np.flatnonzero(year_mask.to_numpy())
    if len(year_positions) == 0:
        raise ValueError(f"No inference rows were found for year {year}.")

    max_lag = max(
        (max(lags, default=0) for lags in lag_params.values()),
        default=0,
    )
    first_year_position = int(year_positions[0])
    last_year_position = int(year_positions[-1]) + 1
    context_start = max(0, first_year_position - max_lag)

    inference_batch = data.iloc[
        first_year_position:last_year_position
    ].copy().reset_index(drop=True)
    inference_context = data.iloc[
        context_start:last_year_position
    ].copy().reset_index(drop=True)

    return inference_batch, inference_context


def filter_predictions_to_year(
    predictions: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Keep forecasts whose forecast timestamp belongs to the selected year."""
    if "forecast_timestamp" not in predictions.columns:
        raise ValueError(
            "Predictions must contain a 'forecast_timestamp' column."
        )

    forecast_times = pd.to_datetime(predictions["forecast_timestamp"])
    filtered = predictions.loc[forecast_times.dt.year.eq(int(year))]

    if filtered.empty:
        raise ValueError(f"No predictions were produced for year {year}.")

    return filtered.reset_index(drop=True)

def rename_columns(
    df: pd.DataFrame,
    renaming_dict: dict[str, str],
) -> pd.DataFrame:
    """Rename DataFrame columns using a supplied mapping.

    Args:
        df: Input DataFrame containing the original column names.
        renaming_dict: Dictionary mapping existing column names to new names.

    Returns:
        A DataFrame with the specified columns renamed.
    """
    return df.rename(columns=renaming_dict)


def drop_columns(
    df: pd.DataFrame,
    columns_to_drop: list[str],
) -> pd.DataFrame:
    """Remove specified columns from a DataFrame.

    Args:
        df: Input DataFrame containing the columns to remove.
        columns_to_drop: List of column names to drop.

    Returns:
        A DataFrame without the specified columns.
    """
    return df.drop(columns=columns_to_drop)


def create_features(
    df: pd.DataFrame,
    lag_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Create lag features from historical rider demand.

    For each configured feature, creates lagged columns using the specified
    lag periods. Missing lag values are filled using backward filling.

    The datetime column is converted to datetime format and used to produce
    a separate timestamp output. The datetime column is then removed from
    the feature DataFrame.

    Args:
        df: Input DataFrame containing rider demand and a datetime column.
        lag_params: Dictionary mapping feature names to lists of lag periods.
            For example, {"total_riders": [1, 2, 3, 22, 23, 24]}.

    Returns:
        A tuple containing:
            - A DataFrame containing the original features and generated
              demand lag features, excluding the datetime column.
            - A Series containing the converted timestamps.
    """

    # Sort observations chronologically.
    df = df.sort_values("datetime").reset_index(drop=True)

    # Convert the datetime column to datetime format.
    timestamps = pd.to_datetime(df["datetime"])

    # Create lag features for each configured feature.
    for feature, lags in lag_params.items():
        for lag in lags:
            df[f"{feature}_lag_{lag}"] = df[feature].shift(lag).bfill()

    # Calculate the change in rider demand from the previous observation.
    df["riders_change_1"] = df["total_riders"] - df["total_riders_lag_1"]

    # Remove the datetime column from the feature DataFrame.
    df.drop(columns=["datetime"], inplace=True)

    return df, timestamps

"""Training nodes for Boston Blue Bike demand forecasting."""

def create_target(
    df: pd.DataFrame,
    target_params: Dict[str, Any],
) -> pd.DataFrame:
    """Create a target column by shifting the source values by the horizon."""
    df[target_params["target_column"]] = (
        df[target_params["source_column"]]
        .shift(-target_params["forecast_horizon"])
        .ffill()
    )
    return df

def split_data(
    df: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a time-series dataset into chronological training and testing sets.

    The function preserves row order and does not shuffle observations.
    It validates the target column and training fraction before splitting.

    Args:
        df: Chronologically sorted dataset containing the target column.
        params: Configuration containing target_params.target_column
            and train_fraction.

    Returns:
        X_train, X_test, y_train, y_test.

    Raises:
        ValueError: If the dataset is empty, the target is missing or
            contains null values, or the training fraction is invalid.
    """

    # Validate the input dataset.
    if df.empty:
        raise ValueError("The input dataset must not be empty.")

    # Retrieve and validate the target column.
    target_name = params["target_params"]["target_column"]

    if target_name not in df.columns:
        raise ValueError(
            f"The dataset must contain the target column '{target_name}'."
        )

    if df[target_name].isna().any():
        raise ValueError(
            f"The target column '{target_name}' contains missing values."
        )

    # Retrieve and validate the training fraction.
    train_fraction = params["train_fraction"]

    if not 0 < train_fraction < 1:
        raise ValueError(
            "train_fraction must be strictly between 0 and 1."
        )

    # Separate predictor features from the target variable.
    X = df.drop(columns=[target_name])
    y = df[target_name]

    # Calculate the chronological split boundary.
    train_end = int(len(df) * train_fraction)

    # Ensure both datasets contain observations.
    if train_end == 0 or train_end == len(df):
        raise ValueError(
            "The training fraction must leave observations in both sets."
        )

    # Split without shuffling, preserving the original row order.
    X_train = X.iloc[:train_end].copy()
    X_test = X.iloc[train_end:].copy()

    y_train = y.iloc[:train_end].copy()
    y_test = y.iloc[train_end:].copy()

    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
) -> CatBoostRegressor:
    """Train a CatBoost regression model.

    Args:
        x_train: Training features, including categorical features.
        y_train: Target values for training.
        params: Dictionary containing:
            - model_type: Model type to train; currently supports CatBoost.
            - model_params: Dictionary of CatBoost hyperparameters.

    Returns:
        The fitted CatBoost regression model.

    Raises:
        ValueError: If the specified model type is unsupported.
    """
    # Normalise the model type.
    model_type = params["model_type"].lower().strip()

    # Retrieve the model-specific hyperparameters.
    if model_type not in ["catboost", "cb"]:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_params = params["model_params"]["catboost"]

    # Display the parameters used to initialise the model.
    print(f"CatBoost training parameters: {model_params}")

    # Identify categorical features present in the training data.
    categorical_features = [
        column
        for column in ["weather"]
        if column in X_train.columns
    ]

    # Initialise the CatBoost model.
    model = CatBoostRegressor(**model_params)

    # Train the model with native categorical feature handling.
    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
    )

    return model


def predict(
    model: CatBoostRegressor,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions using a fitted CatBoost model.

    Args:
        model: A fitted CatBoost regression model.
        X: Features for which predictions are required.

    Returns:
        A DataFrame containing predictions aligned with the input index.
    """
    y_pred = model.predict(X)

    predictions = pd.DataFrame(
        {"prediction": y_pred},
        index=X.index,
    )

    return predictions

def compute_metrics(
    y_true: Union[np.ndarray, list], 
    y_pred: Union[np.ndarray, list]
) -> Dict[str, float]:
    """
    Compute evaluation metrics between true and predicted values.

    Metrics returned:
    - MAPE: Mean Absolute Percentage Error (in %)
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error

    Parameters:
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns:
    -------
    dict
        Dictionary with keys 'MAPE', 'MAE', and 'RMSE' and their float values.
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    
    # Calculate the mean absolute prediction error.
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate the root mean squared prediction error.
    rmse = root_mean_squared_error(y_true, y_pred)
    
    # Calculate the coefficient of determination.
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R²": round(float(r2), 4),
    }
    print(f"Metrics {metrics}")
    return metrics


def log_to_mlflow(
    model: CatBoostRegressor,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    X_train: pd.DataFrame,
) -> None:
    """Log model parameters, evaluation metrics and model artefact to MLflow.

    Uses the existing Boston Bike MLflow experiment and SQLite database.

    Args:
        model: Trained CatBoost regression model.
        metrics: Evaluation metrics produced by compute_metrics().
        params: Training configuration containing model_type and model_params.
    """
    # Resolve the project root independently of the current working directory.
    project_root = Path(__file__).resolve().parents[3]

    # Configure the existing MLflow tracking database.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        database_path = project_root / "mlflow.db"
        tracking_uri = f"sqlite:///{database_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    # Reuse the existing experiment.
    experiment_name = "Boston_Bike_Demand_Forecasting"
    mlflow.set_experiment(experiment_name)

    # Retrieve the parameters used to train the model.
    model_type = params["model_type"]
    model_params = params["model_params"][model_type]

    # Start a new run for this Kedro training execution.
    with mlflow.start_run(run_name="Kedro_CatBoost_Training"):

        # Record the model type and training parameters.
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(model_params)

        # Record the evaluation metrics.
        # MLflow metric names cannot contain certain special characters,
        # so convert the R² label to a safe name.
        for metric_name, metric_value in metrics.items():
            safe_name = metric_name.replace("Â²", "2").replace("²", "2")
            mlflow.log_metric(safe_name, float(metric_value))

        # Select a small sample of the training features.
        input_example = X_train.head(5).copy()

        # Generate sample predictions using the trained model.
        sample_predictions = model.predict(input_example)

        # Infer the model's input and output schema.
        signature = infer_signature(
            input_example,
            sample_predictions,
        )

        # Log the model with its input example and signature.
        mlflow.catboost.log_model(
            cb_model=model,
            name="catboost_model",
            registered_model_name="Boston_Bike_Demand_Model",
            input_example=input_example,
            signature=signature,
        )

    print(
        f"MLflow logging completed. "
        f"Experiment: {experiment_name}"
    )


def load_mlflow_model(uri: str):
    """Load a registered MLflow model by version or alias."""
    if uri.startswith("models:/") and "@" in uri:
        model_name, alias = uri.removeprefix("models:/").rsplit("@", 1)

        if not model_name or not alias:
            raise ValueError(f"Invalid MLflow model alias URI: {uri}")

        client = mlflow.tracking.MlflowClient()
        version = client.get_model_version_by_alias(model_name, alias)
        uri = f"models:/{model_name}/{version.version}"

    return mlflow.pyfunc.load_model(uri)

def predict_with_mlflow(
    model,
    X: pd.DataFrame,
    timestamps: pd.Series,
) -> pd.DataFrame:
    """Predict demand and include the input and forecast timestamps."""
    predictions = model.predict(X)

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps).to_numpy(),
            "forecast_timestamp": (
                pd.to_datetime(timestamps) + pd.Timedelta(hours=1)
            ).to_numpy(),
            "prediction": predictions,
        },
        index=X.index,
    ) 






