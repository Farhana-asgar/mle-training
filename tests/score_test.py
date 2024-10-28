
import subprocess
import time

import mlflow
import pytest

from house_value_prediction.ingest_data import IngestData
from house_value_prediction.score import Score
from house_value_prediction.train import Train


@pytest.fixture
def sample_data():

    temp_dir_dataset = './test_data'
    temp_dir_model = './model_data'

    experiment_name = "House Value Prediction Test"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # If it doesn't exist, create it
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Experiment '{experiment_name}' already exists with ID: {experiment_id}")

    # Set the experiment to be used
    mlflow.set_experiment(experiment_name)
    try:
        remote_server_uri = "http://localhost:5000"
        mlflow.set_tracking_uri(remote_server_uri)

        # Run the server as a subprocess
        subprocess.Popen(
            ["python", "-m", "mlflow", "server", "--host", "localhost", "--port", str(5000)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Allow time for the server to start
        time.sleep(5)
    except Exception as e:
        print(f"Failed to start MLflow server: {e}")

    # Initialize IngestData with the temp directory
    ingest = IngestData(dataset_location=temp_dir_dataset, no_console_log=True)
    train = Train(dataset_location=temp_dir_dataset,
                  model_location=temp_dir_model, no_console_log=True)
    score = Score(dataset_location=temp_dir_dataset,
                  model_location=temp_dir_model, no_console_log=True)

    score.global_variable_initialization(temp_dir_dataset)

    return temp_dir_dataset, temp_dir_model, score


def test_lin_reg_scoring(sample_data):
    dataset_location, model_location, score = sample_data
    e1, e2 = score.lin_reg_scoring(model_location)
    assert isinstance(e1, float)
    assert isinstance(e2, float)


def test_dec_tree_scoring(sample_data):
    dataset_location, model_location, score = sample_data
    e1, e2 = score.dec_tree_scoring(model_location)
    assert isinstance(e1, float)
    assert isinstance(e2, float)


def test_random_forest_scoring(sample_data):
    dataset_location, model_location, score = sample_data
    e1, e2 = score.random_forest_scoring(model_location, dataset_location)
    assert isinstance(e1, float)
    assert isinstance(e2, float)
