
import pytest

from house_value_prediction.scripts import score


@pytest.fixture
def sample_data():
    dataset_location = "F://Training//GIT//mle-training//src//house_value_prediction//scripts_output//ingest_data"
    score.global_variable_initialization(dataset_location)
    model_location = "F://Training//GIT//mle-training//src//house_value_prediction//scripts_output//train"
    return dataset_location, model_location


def test_lin_reg_scoring(sample_data):
    dataset_location, model_location = sample_data
    e1, e2 = score.lin_reg_scoring(model_location)
    assert isinstance(e1, float)
    assert isinstance(e2, float)


def test_dec_tree_scoring(sample_data):
    dataset_location, model_location = sample_data
    e1, e2 = score.dec_tree_scoring(model_location)
    assert isinstance(e1, float)
    assert isinstance(e2, float)


def test_random_forest_scoring(sample_data):
    dataset_location, model_location = sample_data
    e1, e2 = score.random_forest_scoring(model_location, dataset_location)
    assert isinstance(e1, float)
    assert isinstance(e2, float)
