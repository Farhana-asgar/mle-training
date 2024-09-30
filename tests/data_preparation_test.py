import numpy as np
import pandas as pd

from house_value_prediction import data_preparation


def test_prepare_dataset():
    housing = pd.DataFrame({
        'median_income': np.random.rand(25),
        'longitude': np.random.rand(25),
        'latitude': np.random.rand(25)
    })
    train_set, strat_train_set, strat_test_set = \
        data_preparation.prepare_dataset(housing)
    print("start")
    print(strat_test_set)

    # Assert the shape of the returned datasets
    assert train_set.shape[0] == 20  # 80% of 6
    assert strat_test_set.shape[0] == 5  # 20% of 6

    # Check the income_cat column
    assert 'income_cat' not in train_set.columns
    assert 'income_cat' not in strat_test_set.columns
    print("strat")

    print(strat_train_set)

    # Check if the function returns a DataFrame
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(strat_train_set, pd.DataFrame)
    assert isinstance(strat_test_set, pd.DataFrame)


def test_feature_engineering():
    housing = pd.DataFrame({
        'median_income': np.random.rand(25),
        'longitude': np.random.rand(25),
        'latitude': np.random.rand(25),
        'income_cat': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3,
                       4, 5, 1, 2, 3, 4, 5],
        'total_rooms': np.random.rand(25),
        'total_bedrooms': np.random.rand(25),
        'population': np.random.rand(25),
        'households': np.random.rand(25)
    })
    strat_train_set = pd.DataFrame({
        'longitude': np.random.rand(25),
        'latitude': np.random.rand(25),
        'median_house_value': np.random.rand(25)
    })
    housing, housing_labels = data_preparation.feature_engineering(
        housing, strat_train_set)
    assert "median_house_value" not in housing.columns
    assert isinstance(housing_labels, pd.Series)


def test_fill_missing_values():
    random_values = np.random.rand(25 - 1)  # Generate random values, leaving space for NaN
    with_nan = np.insert(random_values, np.random.randint(0, 25), np.nan)
    housing = pd.DataFrame({
        'longitude': with_nan,
        'latitude': np.random.rand(25),
        'ocean_proximity': np.random.rand(25),
        'total_rooms': np.random.rand(25),
        'total_bedrooms': with_nan,
        'population': np.random.rand(25),
        'households': np.random.rand(25)
    })
    housing_prepared, imputer = data_preparation.fill_missing_values(housing)
    assert not housing_prepared['longitude'].isna().any()
    assert "rooms_per_household" in housing_prepared.columns
    assert "bedrooms_per_room" in housing_prepared.columns
    assert "population_per_household" in housing_prepared.columns
    assert "ocean_proximity" in housing_prepared.columns
