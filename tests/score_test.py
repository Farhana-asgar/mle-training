
import pytest

from house_value_prediction import score


@pytest.fixture
def sample_data():
    dataset_location = "/home/runner/work/mle-training/mle-training/scripts_output/ingest_data"
    # dataset_location = "F:/Training/GIT/mle-training/scripts_output/ingest_data"
    model_location = "/home/runner/work/mle-training/mle-training/scripts_output/train"
    # model_location = "F:/Training/GIT/mle-training/scripts_output/train"

    score.global_variable_initialization(dataset_location)

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
