from kedro.pipeline import Pipeline, node
from .nodes import (
    filter_predictions_to_year,
    load_mlflow_model,
    predict_with_mlflow,
)


def create_inference_pipeline() -> Pipeline:
    """Load the registered model, predict, and keep the selected forecast year."""
    return Pipeline(
        [
            node(
                func=load_mlflow_model,
                inputs="params:model_storage.uri",
                outputs="mlflow_model",
                name="load_mlflow_model_node",
            ),
            node(
                func=predict_with_mlflow,
                inputs=["mlflow_model", "featured_data", "timestamps"],
                outputs="all_inference_predictions",
                name="predict_with_mlflow_node",
            ),
            node(
                func=filter_predictions_to_year,
                inputs=["all_inference_predictions", "params:inference.year"],
                outputs="predictions_with_timestamps",
                name="filter_predictions_to_year_node",
            ),
        ]
    )
