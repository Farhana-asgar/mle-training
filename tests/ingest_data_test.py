import os

import numpy as np
import pandas as pd
import pytest

from house_value_prediction.scripts import ingest_data

# Constants for testing
MOCK_HOUSING_URL = "http://mock.url/housing.tgz"
MOCK_HOUSING_PATH = "mock/path/housing"


@pytest.fixture
def mock_urlretrieve(mocker):
    return mocker.patch("urllib.request.urlretrieve", return_value=None)


@pytest.fixture
def mock_tarfile_open(mocker):
    mock_tar = mocker.Mock()
    # Mocking the tarfile.open method correctly
    return mocker.patch("tarfile.open", return_value=mock_tar)


@pytest.fixture
def mock_makedirs(mocker):
    return mocker.patch("os.makedirs", return_value=None)


def test_fetch_housing_data(mock_urlretrieve,
                            mock_tarfile_open, mock_makedirs):

    ingest_data.fetch_housing_data(housing_url=MOCK_HOUSING_URL,
                                   housing_path=MOCK_HOUSING_PATH)

    # Assertions
    mock_makedirs.assert_called_once_with(MOCK_HOUSING_PATH, exist_ok=True)

    mock_urlretrieve.assert_called_once_with(MOCK_HOUSING_URL, os.path.join(
        MOCK_HOUSING_PATH, "housing.tgz"))

    # Check that tarfile.open was called with the correct path
    mock_tarfile_open.assert_called_once_with(os.path.join(
        MOCK_HOUSING_PATH, "housing.tgz"))

    # Check that extractall was called
    mock_tarfile_open().extractall.assert_called_once_with(
        path=MOCK_HOUSING_PATH)

    # Check that close was called
    mock_tarfile_open().close.assert_called_once()


def test_load_housing_data(tmp_path):
    # Create a temporary CSV file
    data = {
        "income_cat": ["high", "medium", "medium", "low", "high", "low",
                       "medium"]
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "housing.csv"
    df.to_csv(csv_path, index=False)

    # Load the housing data from the temporary file
    loaded_data = ingest_data.load_housing_data(housing_path=tmp_path)

    # Check that the DataFrame is not empty
    assert not loaded_data.empty, "DataFrame should not be empty"

    # Check that the 'income_cat' column exists
    assert "income_cat" in loaded_data.columns, "'income_cat' column should \
        be in the DataFrame"


def test_income_cat_proportions(tmp_path):
    # Create a temporary CSV file
    data = {
        "income_cat": ["high", "medium", "medium", "low", "high",
                       "low", "medium"]
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "housing.csv"
    df.to_csv(csv_path, index=False)

    # Load the housing data from the temporary file
    loaded_data = ingest_data.load_housing_data(housing_path=tmp_path)

    # Calculate the proportions
    proportions = ingest_data.income_cat_proportions(loaded_data)

    # Expected proportions based on the example data
    expected_proportions = {
        "high": 2 / 7,
        "medium": 3 / 7,
        "low": 2 / 7
    }

    # Check if the calculated proportions match the expected values
    for category, expected in expected_proportions.items():
        assert proportions[category] == expected, f"Expected {expected} for \
            category '{category}' but got {proportions[category]}"


def test_prepare_dataset():
    housing = pd.DataFrame({
        'median_income': np.random.rand(25),
        'longitude': np.random.rand(25),
        'latitude': np.random.rand(25)
    })
    train_set, strat_train_set, strat_test_set = \
        ingest_data.prepare_dataset(housing)

    # Assert the shape of the returned datasets
    assert train_set.shape[0] == 20  # 80% of 6
    assert strat_test_set.shape[0] == 5  # 20% of 6

    # Check the income_cat column
    assert 'income_cat' not in train_set.columns
    assert 'income_cat' not in strat_test_set.columns


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
    housing, housing_labels = ingest_data.feature_engineering(
        housing, strat_train_set)
    assert "median_house_value" not in housing.columns
    assert isinstance(housing_labels, pd.Series)


def test_fill_missing_values():
    random_values = np.random.rand(25 - 1)
    # Generate random values, leaving space for NaN
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
    housing_prepared, imputer = ingest_data.fill_missing_values(housing)
    assert not housing_prepared['longitude'].isna().any()
    assert "rooms_per_household" in housing_prepared.columns
    assert "bedrooms_per_room" in housing_prepared.columns
    assert "population_per_household" in housing_prepared.columns
    assert "ocean_proximity" in housing_prepared.columns
    assert "ocean_proximity" in housing_prepared.columns
