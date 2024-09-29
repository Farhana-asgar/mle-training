import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from house_value_prediction import scoring


@pytest.fixture
def sample_data():
    housing_predictions = np.random.randint(1, 5, size=5)
    housing_labels = np.random.randint(1, 5, size=5)
    return housing_predictions, housing_labels


def test_lin_reg_scoring(sample_data):
    housing_predictions, housing_labels = sample_data
    actual_lin_rmse, actual_lin_mae = scoring.line_reg_scoring(
        housing_predictions, housing_labels)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    assert actual_lin_rmse == lin_rmse
    assert actual_lin_mae == lin_mae





def test_dec_tree_scoring(sample_data):
    housing_predictions, housing_labels = sample_data
    actual_tree_mse, actual_tree_rmse = scoring.dec_tree_scoring(
        housing_predictions, housing_labels)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    assert tree_mse == actual_tree_mse
    assert tree_rmse == actual_tree_rmse


def test_random_forest_scoring(sample_data):
    y_test, final_predictions = sample_data
    actual_final_mse, actual_final_rmse = scoring.random_forest_scoring(
        y_test, final_predictions)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    assert final_mse == actual_final_mse
    assert final_rmse == actual_final_rmse
