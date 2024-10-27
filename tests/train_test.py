import subprocess
import time

import mlflow
import numpy as np
import pandas as pd
import pytest
import sklearn

from house_value_prediction.ingest_data import IngestData
from house_value_prediction.train import Train


@pytest.fixture
def sample_data():
    housing_prepared = pd.DataFrame({
        'longitude': np.random.rand(25),
        'latitude': np.random.rand(25),
        'total_rooms': np.random.rand(25),
        'total_bedrooms': np.random.rand(25),
        'population': np.random.rand(25),
        'households': np.random.rand(25),
        'rooms_per_household': np.random.rand(25),
        'bedrooms_per_room': np.random.rand(25),
        'population_per_household': np.random.rand(25)
    })
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

    # Clean up after tests
    housing_labels = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    return (housing_prepared, housing_labels, train)


def test_lin_reg(sample_data):
    housing_prepared, housing_labels, train = sample_data
    print(housing_prepared.shape)
    lin_reg_model = train.lin_reg(housing_prepared, housing_labels)
    print(type(lin_reg_model))
    assert isinstance(lin_reg_model, sklearn.linear_model._base.
                      LinearRegression)


def test_dec_tree(sample_data):
    housing_prepared, housing_labels, train = sample_data
    print(housing_prepared.shape)
    lin_reg_model = train.dec_tree(housing_prepared, housing_labels)
    print(type(lin_reg_model))
    assert isinstance(lin_reg_model, sklearn.tree._classes.
                      DecisionTreeRegressor)
