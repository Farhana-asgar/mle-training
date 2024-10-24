
import os

import pytest

from house_value_prediction.ingest_data import IngestData
from house_value_prediction.score import Score
from house_value_prediction.train import Train


@pytest.fixture
def sample_data():

    temp_dir_dataset = './test_data'
    temp_dir_model = './model_data'

    # Initialize IngestData with the temp directory
    ingest = IngestData(dataset_location=temp_dir_dataset)
    train = Train(dataset_location=temp_dir_dataset,
                  model_location=temp_dir_model)
    score = Score(dataset_location=temp_dir_dataset,
                  model_location=temp_dir_model)

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
