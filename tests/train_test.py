import numpy as np
import pandas as pd
import pytest
import sklearn

from house_value_prediction import train


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
    housing_labels = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    return housing_prepared, housing_labels


def test_lin_reg(sample_data):
    housing_prepared, housing_labels = sample_data
    print(housing_prepared.shape)
    lin_reg_model = train.lin_reg(housing_prepared, housing_labels)
    print(type(lin_reg_model))
    assert isinstance(lin_reg_model, sklearn.linear_model._base.
                      LinearRegression)


def test_dec_tree(sample_data):
    housing_prepared, housing_labels = sample_data
    print(housing_prepared.shape)
    lin_reg_model = train.dec_tree(housing_prepared, housing_labels)
    print(type(lin_reg_model))
    assert isinstance(lin_reg_model, sklearn.tree._classes.
                      DecisionTreeRegressor)
