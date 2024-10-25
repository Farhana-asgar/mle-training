import os

import pandas as pd
import pytest

from house_value_prediction.ingest_data import (  # Importing
    CombinedAttributesAdder,
    IngestData,
)


@pytest.fixture
def ingest_data():
    """Fixture to set up the IngestData object."""
    # Set a temporary directory for testing
    temp_dir = './test_data'
    os.makedirs(temp_dir, exist_ok=True)

    # Initialize IngestData with the temp directory
    ingest = IngestData(dataset_location=temp_dir)
    yield ingest


def test_fetch_housing_data(ingest_data):
    """Test if the housing data is fetched and extracted correctly."""
    ingest_data.fetch_housing_data()
    assert os.path.exists(ingest_data.housing_path)
    assert os.path.exists(os.path.join(
        ingest_data.housing_path, "housing.csv"))


def test_load_housing_data(ingest_data):
    """Test if the housing data is loaded correctly."""
    ingest_data.fetch_housing_data()
    df = ingest_data.load_housing_data()
    assert isinstance(df, pd.DataFrame), "Loaded data should be a DataFrame."
    assert not df.empty, "DataFrame should not be empty."


def test_income_cat_proportions(ingest_data):
    """Test the income category proportions."""
    print("---housing---")
    data = pd.DataFrame({
        'median_income': [1.0, 2.0, 3.0, 4.0, 5.0],
        'income_cat': [1, 2, 3, 4, 5]
    })
    proportions = ingest_data.income_cat_proportions(data)
    assert isinstance(proportions, pd.Series)
    assert len(proportions) > 0


def test_prepare_dataset(ingest_data):
    """Test the dataset preparation process."""
    ingest_data.fetch_housing_data()
    housing = ingest_data.load_housing_data()
    housing, strat_train_set, strat_test_set = ingest_data.prepare_dataset(
        housing)

    assert not housing.empty
    assert 'income_cat' not in housing.columns


def test_feature_engineering(ingest_data):
    """Test the feature engineering process."""
    ingest_data.fetch_housing_data()
    housing = ingest_data.load_housing_data()
    housing, housing_labels = ingest_data.feature_engineering(housing, housing)

    assert isinstance(housing, pd.DataFrame)
    assert 'median_house_value' not in housing.columns


def test_fill_missing_values(ingest_data):
    """Test the missing values filling process."""
    ingest_data.fetch_housing_data()
    housing = ingest_data.load_housing_data()
    housing_prepared, imputer = ingest_data.fill_missing_values(housing)

    assert isinstance(housing_prepared, pd.DataFrame)
    assert housing_prepared.isnull().sum().sum() == 0


def test_combined_attributes_adder():
    """Test the CombinedAttributesAdder transformer."""
    adder = CombinedAttributesAdder()
    data = pd.DataFrame({
        'ocean_proximity': ['NEAR BAY', 'NEAR OCEAN'],
        'longitude': [-122.23, -122.22],
        'latitude': [37.88, 37.89],
        'total_rooms': [1000.9, 2000.9],
        'total_bedrooms': [200.9, 400.9],
        'population': [500.9, 1000.9],
        'households': [300.9, 600.9]
    })

    transformed_df = adder.transform(data)

    assert 'rooms_per_household' in transformed_df.columns
    assert 'population_per_household' in transformed_df.columns
    assert 'bedrooms_per_room' in transformed_df.columns


if __name__ == "__main__":
    pytest.main()
