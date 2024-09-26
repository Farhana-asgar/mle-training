import os

import pandas as pd

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

HOUSING_PATH = os.path.join(BASE_PATH, "datasets", "housing")


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    return (data["income_cat"].value_counts() / len(data))
