import os
import tarfile

import pandas as pd
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

HOUSING_PATH = os.path.join(BASE_PATH, "datasets", "housing")


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    # This function is to extract and fetch the dataset housing.tgz
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    # This function is to load the dataset housing.tgz
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    # Returns proportion of income
    return (data["income_cat"].value_counts() / len(data))
