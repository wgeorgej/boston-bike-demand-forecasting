from kedro.pipeline import Pipeline, node
from .nodes import create_target, split_data, train_model, predict, compute_metrics, log_to_mlflow

def create_training_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_target,
                inputs=["featured_data", "params:training.target_params"],
                outputs= "data_with_target",
                name="created_target_node"
            ),
            node(
                func=split_data,
                inputs= ["data_with_target", "params:training"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node"
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:training"],
                outputs="trained_model",
                name= "trained_model_node"

            ),
            node(
                func=predict,
                inputs=["trained_model", "X_test"],
                outputs="predictions",
                name="predict_node"
            ),
            node(
                func=compute_metrics,
                inputs=["y_test", "predictions"],
                outputs="metrics",
                name="compute_metrics_node",
            ),
            node(
                func=log_to_mlflow,
                inputs=["trained_model", "metrics", "params:training", "X_train"],
                outputs=None,
                name="log_to_mlflow_node",
            )
        ]
    )

