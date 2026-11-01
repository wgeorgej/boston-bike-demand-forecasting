from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


def run_inference() -> None:
    project_path = Path(__file__).resolve().parent.parent
    bootstrap_project(project_path)
    configure_project("seoul_bike")

    with KedroSession.create(project_path=project_path) as session:
        session.run(pipeline_names=["inference"])

    print("Inference pipeline completed successfully!")


if __name__ == "__main__":
    run_inference()
