
import argparse
import subprocess
import time

import mlflow

from house_value_prediction.ingest_data import IngestData
from house_value_prediction.score import Score
from house_value_prediction.train import Train


def get_or_create_experiment(experiment_name):
    # Check if the experiment exists
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
    return experiment_id


def start_mlflow_server(host="0.0.0.0", port=5000):
    try:
        remote_server_uri = "http://localhost:5000"
        mlflow.set_tracking_uri(remote_server_uri)
        exp_name = "House Value Prediction"

        # Run the server as a subprocess
        server = subprocess.Popen(
            ["python", "-m", "mlflow", "server", "--host", host, "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Allow time for the server to start
        time.sleep(5)
        get_or_create_experiment(experiment_name=exp_name)
        return server
    except Exception as e:
        print(f"Failed to start MLflow server: {e}")
        return None


def main(no_console_log=False):
    with mlflow.start_run(run_name="Main Script"):
        dataset_location = './scripts/scripts_output/ingest_data'
        model_location = './scripts/scripts_output/train'
        IngestData(dataset_location, no_console_log)
        Train(dataset_location, model_location, no_console_log)
        Score(dataset_location, model_location, no_console_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Option to get whether to display console logs or not")

    # Add an optional argument
    parser.add_argument('--no-console-logs', action='store_true', help='Option to get whether to display console logs or not')

    # Parse the command-line arguments
    args = parser.parse_args()
    start_mlflow_server()
    main(no_console_log=args.no_console_logs)
