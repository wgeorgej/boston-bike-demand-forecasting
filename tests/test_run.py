from seoul_bike.pipeline_registry import register_pipelines


def test_project_registers_expected_pipelines():
    pipelines = register_pipelines()

    assert set(pipelines) == {"__default__", "feature_eng", "training", "inference"}
    assert pipelines["__default__"].nodes
