# Boston Blue Bike Demand Forecasting

## Project Charter: Problem Statement, Objectives, and Success Metrics

**Project status:** Planning and data preparation\
**Forecasting horizon:** One hour ahead\
**Primary target:** Total Blue Bike ridership\
**Primary evaluation metric:** Mean Absolute Error (MAE)

------------------------------------------------------------------------

## 1. Problem Statement

Blue Bike demand changes over time and may be influenced by temporal
patterns, weather, and other relevant factors. Accurate short-term
demand forecasts can support operational planning, including
anticipating periods of higher or lower bicycle demand.

This project will develop and evaluate machine-learning models that
forecast **total Blue Bike ridership one hour ahead** using only
information that would be available at the time the forecast is made.

The project will compare XGBoost, LightGBM, and CatBoost against a
simple baseline model. Evaluation will respect the chronological order
of the time series to better represent how the models would perform on
future observations.

## 2. Project Aim

To build a reproducible, leakage-aware forecasting workflow that
predicts hourly Blue Bike ridership one hour ahead and compares the
performance of XGBoost, LightGBM, and CatBoost.

## 3. Specific Objectives

1.  Document the dataset, its variables, time coverage, data quality,
    and limitations.
2.  Define the forecasting target so that each row predicts ridership
    exactly one hour after its forecast timestamp.
3.  Prepare time-based and lag-based features without introducing target
    leakage.
4.  Create chronological training and validation splits.
5.  Establish a simple baseline and compare it with XGBoost, LightGBM,
    and CatBoost.
6.  Evaluate models using appropriate regression metrics and residual
    diagnostics.
7.  Preserve 2024 observations for a later out-of-time evaluation,
    provided those observations have not been used for model selection
    or tuning.
8.  Develop a reproducible inference workflow with input-schema checks
    and clear output definitions.
9.  Document assumptions, decisions, experiments, and final results.

## 4. Data and Evaluation Plan

### Dataset roles

-   **Model dataset:** `data/03_primary/blue_bike_boston_model.parquet`
    --- observations from 2022--2023 for model development.
-   **Inference dataset:**
    `data/03_primary/blue_bike_boston_inference.parquet` --- the wider
    historical dataset intended for developing and testing the inference
    workflow.

The inference dataset contains historical observations, so it is useful
for pipeline development and simulation. It should not be described as
entirely unseen production data.

### Recommended evaluation strategy

1.  Use chronological training and validation periods within 2022--2023.
2.  Fit preprocessing steps, imputers, and model parameters using
    training data only.
3.  Use validation data for model comparison and tuning.
4.  Keep 2024 observations separate for a final out-of-time evaluation.
5.  Do not use the 2024 evaluation results to repeatedly tune the model;
    doing so would compromise the holdout.

### One-hour-ahead target alignment

For a forecast made at time `t`, the target should represent ridership
at `t + 1 hour`.

Do not assume that the next row always represents the next hour. Before
constructing the target, verify that the next timestamp is exactly one
hour later. Rows where this condition fails should not be treated as
valid one-hour-ahead target pairs.

### Leakage precautions

-   Use only features available at the forecast origin.
-   Do not use weather measured at the target hour unless a suitable
    forecast for that hour would have been available when the prediction
    was made.
-   Do not include `casual_riders` or `member_riders` as predictors if
    they directly sum to the target `total_riders` for the same hour.
-   Construct lag and rolling features so that they use past information
    only.
-   Do not fill missing ridership observations with zero unless there is
    evidence that zero genuinely represents no rides.

## 5. Success Metrics

### Primary metric: Mean Absolute Error (MAE)

MAE is the average absolute difference between predicted and observed
ridership. It is straightforward to interpret in the target's units and
will be the primary metric for model selection.

### Secondary metric: Root Mean Squared Error (RMSE)

RMSE gives greater weight to large prediction errors and helps reveal
whether a model makes occasional substantial mistakes.

### Supporting metric: R-squared (R²)

R² provides a measure of explained variation relative to a mean-based
reference. It should be interpreted alongside MAE and RMSE, not used
alone.

### Proposed performance goal

A proposed initial goal is to reduce validation MAE by at least **10%
compared with the selected baseline**, if feasible. This is a target to
assess, not a result already achieved. The baseline must be defined and
evaluated using the same chronological validation period as the
candidate models.

### Engineering and process criteria

-   Target timestamps are aligned exactly one hour ahead.
-   No future information leaks into model features.
-   Training, validation, and final evaluation respect time order.
-   Preprocessing is fitted on training data only.
-   The workflow is reproducible and documented.
-   The inference pipeline checks required columns and data types.
-   Prediction outputs identify the forecast timestamp and predicted
    target hour clearly.
-   Inference latency is measured; a numerical service-level target can
    be set after expected workload and deployment requirements are
    known.

## 6. Planned Deliverables

1.  Documented problem statement, objectives, assumptions, and success
    metrics.
2.  Clean and validated analysis dataset.
3.  Exploratory data analysis report.
4.  Leakage-safe target and feature-engineering pipeline.
5.  Chronological training and validation workflow.
6.  Baseline and trained XGBoost, LightGBM, and CatBoost models.
7.  Model comparison report with MAE, RMSE, and R².
8.  Final out-of-time evaluation using the reserved 2024 period.
9.  Reusable inference pipeline and input validation.
10. Project README and supporting technical documentation.

## 7. Definition of Done

The project will be considered complete when:

-   The forecasting target is correctly aligned one hour ahead.
-   Data-quality decisions and feature definitions are documented.
-   Training and validation are chronological and leakage-aware.
-   The three candidate models are compared against a baseline using the
    agreed metrics.
-   Final evaluation is reported on the reserved out-of-time period.
-   The chosen model and preprocessing workflow can be reproduced.
-   The inference pipeline accepts appropriately prepared inputs and
    produces clearly timestamped predictions.
-   Limitations, assumptions, and results are documented.

## 8. Key Assumptions and Limitations

-   Historical relationships may change over time, so performance on
    past data may not perfectly represent future performance.
-   Missing hourly timestamps exist in the model dataset. They should be
    handled through explicit timestamp alignment rather than
    automatically interpreted as zero demand.
-   Model performance depends on the quality and availability of the
    features at forecast time.
-   The proposed 10% MAE improvement is an initial goal and may need to
    be revised after baseline evaluation.
-   The project is a forecasting and pipeline-development exercise;
    operational deployment requirements will need to be defined
    separately.

------------------------------------------------------------------------

**Recommended project location:**
`docs/Blue_Bike_Boston_Project_Charter.md`
