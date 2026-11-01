from kedro.pipeline import Pipeline, node
from .nodes import (
    create_features,
    drop_columns,
    prepare_inference_batch,
    validate_bike_data,
    rename_columns,
)


def create_feature_eng_pipeline(input_dataset: str = "train_data") -> Pipeline:
    """Build the shared feature engineering pipeline."""
    return Pipeline(
        [
            node(
                func=rename_columns,
                inputs=[
                    input_dataset,
                    "params:feature_engineering.rename_columns",
                ],
                outputs="renamed_data",
                name="rename_columns_node",
            ),
            node(
                func=drop_columns,
                inputs=[
                    "renamed_data",
                    "params:feature_engineering.drop_columns",
                ],
                outputs="features",
                name="drop_columns_node",
            ),
            node(
                func=create_features,
                inputs=["features", "params:feature_engineering.lag_params"],
                outputs=["featured_data", "timestamps"],
                name="created_features_node",
            ),
        ]
    )


def create_training_feature_eng_pipeline() -> Pipeline:
    """Validate training inputs, then run the existing feature engineering."""
    validation = Pipeline(
        [
            node(
                func=validate_bike_data,
                inputs="train_data",
                outputs="validated_train_data",
                name="validate_training_data_node",
            ),
        ]
    )
    return validation + create_feature_eng_pipeline(
        input_dataset="validated_train_data"
    )


def create_inference_feature_eng_pipeline() -> Pipeline:
    """Save the selected-year batch and build features with warm-up context."""
    prepare_batch = Pipeline(
        [
            node(
                func=validate_bike_data,
                inputs="inference_data",
                outputs="validated_inference_data",
                name="validate_inference_data_node",
            ),
            node(
                func=prepare_inference_batch,
                inputs=[
                    "validated_inference_data",
                    "params:inference.year",
                    "params:feature_engineering.lag_params",
                ],
                outputs=["inference_batch", "inference_context"],
                name="prepare_inference_batch_node",
            ),
        ]
    )
    return prepare_batch + create_feature_eng_pipeline(
        input_dataset="inference_context"
    )


