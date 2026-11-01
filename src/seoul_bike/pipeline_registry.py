from kedro.pipeline import Pipeline

from .pipelines.feature_eng import (
    create_training_feature_eng_pipeline,
    create_inference_feature_eng_pipeline,
)
from .pipelines.inference import create_inference_pipeline
from .pipelines.training import create_training_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    training_feature_eng = create_training_feature_eng_pipeline()
    training_pipeline = (
        training_feature_eng + create_training_pipeline()
    )

    inference_feature_eng = create_inference_feature_eng_pipeline()
    inference_pipeline = (
        inference_feature_eng + create_inference_pipeline()
    )

    return {
        "__default__": training_pipeline,
        "feature_eng": training_feature_eng,
        "training": training_pipeline,
        "inference": inference_pipeline,
    }