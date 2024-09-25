import pandas as pd

from house_value_prediction import data_loading


def test_load_housing_data(tmp_path):
    # Create a temporary CSV file
    data = {
        "income_cat": ["high", "medium", "medium", "low", "high", "low", "medium"]
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "housing.csv"
    df.to_csv(csv_path, index=False)

    # Load the housing data from the temporary file
    loaded_data = data_loading.load_housing_data(housing_path=tmp_path)

    # Check that the DataFrame is not empty
    assert not loaded_data.empty, "DataFrame should not be empty"

    # Check that the 'income_cat' column exists
    assert "income_cat" in loaded_data.columns, "'income_cat' column should be in the DataFrame"


def test_income_cat_proportions(tmp_path):
    # Create a temporary CSV file
    data = {
        "income_cat": ["high", "medium", "medium", "low", "high", "low", "medium"]
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "housing.csv"
    df.to_csv(csv_path, index=False)

    # Load the housing data from the temporary file
    loaded_data = data_loading.load_housing_data(housing_path=tmp_path)

    # Calculate the proportions
    proportions = data_loading.income_cat_proportions(loaded_data)

    # Expected proportions based on the example data
    expected_proportions = {
        "high": 2 / 7,
        "medium": 3 / 7,
        "low": 2 / 7
    }

    # Check if the calculated proportions match the expected values
    for category, expected in expected_proportions.items():
        assert proportions[category] == expected, f"Expected {expected} for category '{category}' but got {proportions[category]}"

# To run the tests, execute `pytest` in the terminal within your project directory.
