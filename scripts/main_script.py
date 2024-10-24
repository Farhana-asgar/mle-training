
import mlflow

from house_value_prediction.ingest_data import IngestData
from house_value_prediction.score import Score
from house_value_prediction.train import Train

remote_server_uri = "http://localhost:5000"
mlflow.set_tracking_uri(remote_server_uri)

exp_name = "ElasticNet_wine"
mlflow.set_experiment(exp_name)


def main():
    with mlflow.start_run():
        dataset_location = './scripts_output/ingest_data'
        model_location = './scripts_output/train'
        IngestData(dataset_location)
        Train(dataset_location, model_location)
        Score(dataset_location, model_location)


if __name__ == "__main__":
    main()
