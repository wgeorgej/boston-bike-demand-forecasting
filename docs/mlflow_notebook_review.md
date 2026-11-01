# MLflow Review: `03_Modelling.ipynb`

## Purpose

This document reviews the MLflow section in `notebooks/03_Modelling.ipynb` and compares it with the Kedro training and inference paths. It gives a corrected notebook pattern for tracking an evaluated CatBoost model in the project's MLflow server and registering it for inference.

The review is about MLflow tracking and model handoff. It does not rewrite the notebook or change the project's feature engineering.

## What the notebook currently does

The MLflow section is in cells 120-125:

1. Cell 120 builds a local SQLite URI from the project root, creates an `mlruns` directory, and selects `Boston_Bike_Demand_Forecasting`.
2. Cell 121 writes a separate native CatBoost `.cbm` file under `data/06_models`.
3. Cell 123 calculates MAE, RMSE, and R² on `X_test` and `y_test`.
4. Cell 125 starts an MLflow run and logs scalar CatBoost parameters, Optuna information, test metrics, row/feature counts, tags, the native `.cbm` artifact, and a CatBoost-flavor model with an input example and inferred signature.

This is a good foundation: it records the main model, metrics, parameters, and a usable example/signature. MLflow's CatBoost flavor can log and register a model, and the signature describes expected input and output shapes. [CatBoost flavor API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.catboost.html), [model signatures](https://mlflow.org/docs/latest/model/signatures/)

## Findings to fix or keep in mind

### 1. The notebook does not register its model

Cell 125 calls `mlflow.catboost.log_model(...)` without `registered_model_name`. That logs a model artifact under the run, but does not create a version in `Boston_Bike_Demand_Model`.

The Kedro training node in `src/seoul_bike/pipelines/nodes.py` does pass `registered_model_name="Boston_Bike_Demand_Model"`. That is the code path which creates model versions in the Registry. The notebook and Kedro pipeline therefore have different registration behavior. [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry)

The corrected notebook example below includes `registered_model_name`, but registration alone does not make that version the `champion` used by inference. Promotion should remain a separate, deliberate action after evaluation.

### 2. Tracking URI is hardcoded to the local store

Cell 120 always points the notebook at `<project root>/mlflow.db`. That works for local experimentation, but it ignores `MLFLOW_TRACKING_URI`. The Kedro logger already reads that environment variable and falls back to the local SQLite database.

This can create two separate MLflow locations:

- notebook run: local `mlflow.db` and its artifact location;
- Docker services: MLflow server at `http://mlflow:5000` from other containers, or `http://localhost:5000` from the host.

The notebook should follow the same environment variable convention as the Kedro logger. Use the local SQLite URI only as a development fallback when no tracking server is selected.

### 3. Creating `mlruns` does not configure MLflow's artifact root

The notebook creates `MLFLOW_ARTIFACT_DIR`, but never passes that directory to MLflow. Creating it alone does not set an experiment's artifact location. For a centralized setup, let the MLflow server manage artifact storage. Keep the server's database and artifact directory on its persistent Docker volume.

### 4. The model is saved twice

Cell 121 saves `final_model` as a native `.cbm`, and cell 125 logs both that file and an MLflow CatBoost model. The separate `.cbm` is optional. The MLflow model artifact already contains the model and its MLflow flavor/environment metadata, so use the MLflow model as the deployment source of truth. Keep the native export only if you have a separate CatBoost-only consumer that needs it.

### 5. Keep CatBoost's best-iteration index separate from its tree count

Cell 112 sets `iterations` to `best_trial.user_attrs["best_iteration"]`. CatBoost documents this as the identifier/index of the best iteration; to train that many trees, use `best_iteration + 1`. [CatBoost `get_best_iteration`](https://catboost.ai/docs/en/concepts/python-reference_catboost_get_best_iteration)

Log both values with clear names: `best_iteration_index` and `selected_tree_count`. Do not label a zero-based index as a tree count.

### 6. Add data and code lineage

The notebook logs row counts and feature count, but not the dataset identity/digest, the source code commit, or the actual feature column list. MLflow can store training and evaluation dataset metadata on a run with `mlflow.log_input()`. This records lineage metadata such as source, digest, schema, and profile; it does not require uploading the entire dataset as a run artifact. [MLflow dataset tracking](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.data.html)

The project also has timestamp gaps, and the user's chosen row-based target/lag feature transformations are intentionally preserved. Log the target/horizon definition accurately. A row shift can mean the next row rather than exactly the next clock hour when a gap exists; MLflow records that definition but cannot reconcile it.

## Correct the final model fit first

Use this pattern in place of the current best-parameter/final-fit cells (110-113). It uses the recorded best iteration index as a tree count and preserves the random seed used during tuning:

```python
best_trial = study.best_trial
best_iteration_index = best_trial.user_attrs.get("best_iteration")
if best_iteration_index is None:
    raise ValueError("The best Optuna trial has no recorded best iteration")

best_params = {
    **best_trial.params,
    "iterations": int(best_iteration_index) + 1,
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": 0,
    "allow_writing_files": False,
}

cat_features = ["weather"]
final_model = CatBoostRegressor(**best_params)
final_model.fit(X_train, y_train, cat_features=cat_features)
```

Then calculate final held-out test metrics from this `final_model` and the same `X_test` / `y_test` used for evaluation. Keep the training/validation split used for Optuna separate from the held-out test set.

## Recommended notebook MLflow logging cell

Run this after `final_model` has been fitted and final test metrics have been calculated. Set `MLFLOW_TRACKING_URI` before starting Jupyter if you want the notebook to use the containerized MLflow server. From the host, use `http://localhost:5000`; from a container on the Compose network, use `http://mlflow:5000`. When unset, this code uses the project's local SQLite database.

```python
import json
import os
import subprocess
from pathlib import Path

import mlflow
import mlflow.catboost
import mlflow.data
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

MODEL_NAME = "Boston_Bike_Demand_Model"
EXPERIMENT_NAME = "Boston_Bike_Demand_Forecasting"

# Locate the Kedro project even if Jupyter starts in notebooks/.
PROJECT_ROOT = next(
    (
        parent
        for parent in [Path.cwd(), *Path.cwd().parents]
        if (parent / "pyproject.toml").exists()
        and (parent / "conf" / "base" / "parameters.yml").exists()
    ),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Could not locate the Kedro project root")

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    tracking_uri = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(EXPERIMENT_NAME)

# Recalculate predictions/metrics from the final fitted model.
y_test_array = np.asarray(y_test).reshape(-1)
test_predictions = np.asarray(final_model.predict(X_test)).reshape(-1)
test_metrics = {
    "test_mae": float(mean_absolute_error(y_test_array, test_predictions)),
    "test_rmse": float(root_mean_squared_error(y_test_array, test_predictions)),
    "test_r2": float(r2_score(y_test_array, test_predictions)),
}

# Include the actual feature columns only; do not include the target or timestamps
# unless they are true model input features.
input_example = X_train.head(min(5, len(X_train))).copy()
example_predictions = final_model.predict(input_example)
signature = infer_signature(input_example, example_predictions)

# Identify code revision for reproducibility.
git_result = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=PROJECT_ROOT,
    capture_output=True,
    text=True,
    check=False,
)
git_commit = git_result.stdout.strip() if git_result.returncode == 0 else "unknown"

# Dataset lineage metadata: change this to the exact training file/source used
# to produce X_train and y_train in this notebook.
training_source = "data/03_primary/blue_bike_boston_model.parquet"
train_lineage = X_train.reset_index(drop=True).copy()
train_lineage["target"] = np.asarray(y_train).reshape(-1)
train_dataset = mlflow.data.from_pandas(
    train_lineage,
    source=training_source,
    name="blue-bike-training-features",
    targets="target",
)

eval_lineage = X_test.reset_index(drop=True).copy()
eval_lineage["target"] = y_test_array
eval_dataset = mlflow.data.from_pandas(
    eval_lineage,
    source=training_source,
    name="blue-bike-held-out-test-features",
    targets="target",
)

model_params = {
    key: value
    for key, value in final_model.get_params().items()
    if value is not None and isinstance(value, (str, int, float, bool))
}

with mlflow.start_run(run_name="CatBoost_Optuna_Tuned_Final") as run:
    # Model configuration and reproducibility details.
    mlflow.log_params(model_params)
    mlflow.log_params({
        "model_type": "CatBoostRegressor",
        "tuning_method": "Optuna_TPE",
        "optuna_trials": int(len(study.trials)),
        "validation_strategy": "chronological split within training data",
        "best_iteration_index": int(best_iteration_index),
        "selected_tree_count": int(best_iteration_index) + 1,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "feature_columns": json.dumps(X_train.columns.tolist()),
        "random_seed": 42,
        "git_commit": git_commit,
        "training_data_source": training_source,
        "target_definition": "project target column; document its exact row/time alignment",
    })

    # Tuning and final evaluation metrics.
    mlflow.log_metric("best_validation_rmse", float(study.best_value))
    mlflow.log_metrics(test_metrics)

    # Dataset source/digest/schema/profile lineage.
    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(eval_dataset, context="testing")

    mlflow.set_tags({
        "project": "Boston Blue Bike Demand Forecasting",
        "framework": "CatBoost",
        "selection_status": "candidate",
        "evaluation_strategy": "chronological holdout test set",
        "target_horizon": "one row ahead; verify timestamp spacing",
    })

    # Log the model with its input contract and register a new version.
    model_info = mlflow.catboost.log_model(
        cb_model=final_model,
        name="catboost_model",
        registered_model_name=MODEL_NAME,
        signature=signature,
        input_example=input_example,
    )

    print("Run ID:", run.info.run_id)
    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Model URI:", model_info.model_uri)
    print("Registered version:", model_info.registered_model_version)
    print("Final test metrics:", test_metrics)
```

The model logger writes the MLflow model artifact and its environment metadata. Review the logged `MLmodel`, `input_example.json`, `requirements.txt` / environment files in the run artifacts. Keep the runtime environment versions compatible with the project's locked dependencies. The project currently locks MLflow 3.17.2.

If `training_source` is not the precise file used in the notebook, correct it. Dataset metadata is useful only when its source accurately describes the data actually used.

## Smoke-test the logged registered version

A version in the Registry is not ready for inference until it loads from the intended tracking/artifact store and accepts the same feature frame used for training:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
version = model_info.registered_model_version
if version is None:
    raise RuntimeError("MLflow did not return a registered model version")

version_uri = f"models:/{MODEL_NAME}/{version}"
loaded_model = mlflow.pyfunc.load_model(version_uri)
loaded_predictions = np.asarray(loaded_model.predict(input_example)).reshape(-1)
np.testing.assert_allclose(
    loaded_predictions,
    np.asarray(example_predictions).reshape(-1),
    rtol=1e-6,
    atol=1e-6,
)
print("Loaded and verified:", version_uri)
print("Signature:", loaded_model.metadata.signature)
```

This verifies the registered model artifact and signature in the currently selected MLflow store. Test again from the Docker inference container when the model is meant to run there.

## Promote a version only after review

The current inference configuration uses:

```yaml
model_storage:
  uri: "models:/Boston_Bike_Demand_Model@champion"
```

That URI points to the version assigned the `champion` alias. Logging a new version does not automatically make it champion. Compare the candidate's held-out metrics and operational checks with the current champion, then run a separate promotion step only after it passes your acceptance criteria:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    "Boston_Bike_Demand_Model",
    "champion",
    str(model_info.registered_model_version),
)
```

This changes the inference target. Do not put it in a notebook cell that runs automatically after every fit. Record the decision and the metrics used to approve the promotion. To roll back, point `champion` to the previously approved version.

## How the notebook and Kedro should divide responsibility

- **Notebook:** explore data, compare candidates, tune, evaluate, record a reproducible candidate run, and optionally register that candidate.
- **Kedro training pipeline:** perform the repeatable training/evaluation run and register its model artifact. The current `log_to_mlflow` node already does this.
- **Promotion step:** select an approved registry version and update the `champion` alias deliberately.
- **Kedro inference pipeline:** read `params:model_storage.uri`, load the selected registered model, predict, and write timestamped predictions.

Avoid treating a notebook run as a production promotion just because it completed successfully. Keep the notebook experiment tracking URI, Kedro tracking URI, Docker MLflow server, and inference model URI connected to the intended same store.

## Improvements recommended for the current Kedro MLflow node

The current `log_to_mlflow()` in `src/seoul_bike/pipelines/nodes.py` correctly logs configured model parameters, evaluation metrics, an input example/signature, and registers `Boston_Bike_Demand_Model`. To improve lineage and make runs easier to compare, consider adding:

- training and test row counts;
- exact feature column names and categorical column names;
- dataset source and MLflow dataset digest via `mlflow.log_input()`;
- target definition and horizon semantics;
- feature-engineering configuration/version;
- Git commit hash;
- tags distinguishing candidate, approved, and promoted versions.

Keep registry alias promotion separate from the training logger. The existing `model_storage.uri` alias design makes this separation useful: inference can use champion while every training run creates an independently reviewable candidate version.

## MLflow checklist for this project

- [x] Tracking URI and experiment selected before starting the run.
- [x] Run context groups params, metrics, tags, and model.
- [x] CatBoost parameters and evaluation metrics are logged.
- [x] MLflow CatBoost model artifact is logged.
- [x] Model signature and input example are present.
- [x] Kedro training logger registers the model in the Registry.
- [x] Inference loads the configured champion alias through `mlflow.pyfunc`.
- [ ] Notebook logging uses `MLFLOW_TRACKING_URI` rather than hardcoding local SQLite.
- [ ] Notebook model logging registers its candidate version explicitly.
- [ ] Dataset lineage and Git revision are logged for each run.
- [ ] Candidate loading is smoke-tested from the target runtime before promotion.
- [ ] Champion alias is changed only after an explicit evaluation decision.

## Official references

- [MLflow Tracking](https://mlflow.org/docs/latest/tracking/)
- [MLflow Dataset Tracking](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.data.html)
- [MLflow Model Signatures and Input Examples](https://mlflow.org/docs/latest/model/signatures/)
- [MLflow CatBoost flavor API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.catboost.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry)
- [CatBoost `get_best_iteration`](https://catboost.ai/docs/en/concepts/python-reference_catboost_get_best_iteration)
