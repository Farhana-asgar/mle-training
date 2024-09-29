import numpy as np
import pandas as pd
import pytest

from house_value_prediction import models


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
    housing_labels = pd.Series( [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    return housing_prepared, housing_labels


def test_lin_reg(sample_data):
    housing_prepared, housing_labels = sample_data
    print(housing_prepared.shape)
    predictions = models.lin_reg(housing_prepared, housing_labels)

    assert predictions.shape == housing_labels.shape
    assert not np.any(np.isnan(predictions))


def test_dec_tree(sample_data):
    housing_prepared, housing_labels = sample_data
    predictions = models.dec_tree(housing_prepared, housing_labels)

    assert predictions.shape == housing_labels.shape
    assert not np.any(np.isnan(predictions))
