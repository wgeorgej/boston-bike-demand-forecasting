# seoul_bike

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project, which was generated using `kedro 1.7.0`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.


## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/deploy/package_a_project/#package-an-entire-kedro-project)

## Docker Compose workflow

The local Docker Compose setup runs the dashboard and MLflow tracking server. Training and inference are available as on-demand jobs.

### Start the services

Start Docker Desktop, open PowerShell in the project root, then run:

```powershell
docker compose up -d --build
```

Open the dashboard at [http://localhost:8050](http://localhost:8050) and the MLflow UI at [http://localhost:5000](http://localhost:5000).

Check service status and logs with:

```powershell
docker compose ps
docker compose logs --tail 50 app-ui
docker compose logs --tail 50 mlflow
```

The MLflow server stores its database and model artifacts in the persistent `mlflow-store` Docker volume. This is separate from the original root-level `mlflow.db` and `mlruns` files.

### Run training

Training is an on-demand job. It logs the run to the Docker MLflow server and registers a new version of `Boston_Bike_Demand_Model`:

```powershell
docker compose --profile jobs run --rm ml-train
```

Training does not automatically move the `champion` alias. Review the new version and its metrics in the MLflow UI before promoting it.

### Promote a model to `champion`

Choose a model version to promote, then run this from the project root. Replace `3` with the version you reviewed:

```powershell
uv run python -c "from mlflow import MlflowClient; c=MlflowClient(tracking_uri='http://localhost:5000'); name='Boston_Bike_Demand_Model'; version='3'; candidate=c.get_model_version(name, version); assert candidate.status == 'READY'; c.set_registered_model_alias(name, 'champion', version); print(f'champion now points to version {version}')"
```

Inference reads `model_storage.uri` from `conf/base/parameters.yml`. Keep it set to the alias URI:

```yaml
model_storage:
  uri: "models:/Boston_Bike_Demand_Model@champion"
```

### Run inference and view predictions

Run the inference job to load the model assigned to `champion` and write predictions to the configured Parquet output:

```powershell
docker compose --profile jobs run --rm ml-inference
```

The dashboard reads the Parquet files under the mounted `data` directory and displays the saved historical replay. It is not a live data feed.

### Stop the services

Stop the containers and Compose network with:

```powershell
docker compose down
```

This keeps the `mlflow-store` volume. Avoid `docker compose down -v` unless you intend to delete the Docker MLflow database and model artifacts.
