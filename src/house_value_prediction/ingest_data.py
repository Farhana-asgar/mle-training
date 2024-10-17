
import logging
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):

        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            column_names = ["rooms_per_household",
                            "population_per_household",
                            "bedrooms_per_room"]
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room], column_names

        else:
            column_names = ["rooms_per_household",
                            "population_per_household"]
            return np.c_[X, rooms_per_household, population_per_household], \
                column_names


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
BASE_PATH = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
HOUSING_PATH = os.path.join(BASE_PATH, "datasets", "housing")
logger = logging.getLogger(__name__)


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Downloads and extracts the housing dataset.

    Args:
        housing_url (str): URL of the housing dataset to download.
        housing_path (str): \
            Local directory path where the dataset should be stored.

    Returns:
        None: The function doesn't return a value, \
            but it downloads and extracts the dataset.

    """

    # This function is to extract and fetch the dataset housing.tgz
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    logger.info(f"Read and downloaded the data from {housing_url}")
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """
    Reads the housing dataset

    Args:
        housing_path (str): Local directory path where the dataset is stored

    Returns:
        Dataframe: The function returns the contents of the csv file

    """

    # This function is to load the dataset housing.tgz
    csv_path = os.path.join(housing_path, "housing.csv")
    logger.info("Housing data loaded")
    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    """
    Downloads and extracts the housing dataset.

    Args:
        data (Dataframe): The data that is to be used in model creation.

    Returns:
        Series: Proportion of income
    """
    # Returns proportion of income
    return (data["income_cat"].value_counts() / len(data))


def prepare_dataset(housing):
    """
    Downloads and extracts the housing dataset.

    Args:
        housing (Dataframe): The data that is to be used in model creation.

    Returns:
        housing (Dataframe): Modified housing dataframe
        strat_train_set (Dataframe): Split from housing dataset for training
        strat_test_set (Dataframe): Split from housing dataset for testing
    """
    # Prepare and split the dataset for training and testing purpose
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    train_set, test_set = train_test_split(housing, test_size=0.2,
                                           random_state=42)

    compare_props = pd.DataFrame({
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }).sort_index()
    compare_props["Rand. %error"] = 100 * compare_props["Random"]   \
        / compare_props["Overall"] - 100
    compare_props["Strat. %error"] = 100 * compare_props["Stratified"] \
        / compare_props["Overall"] - 100

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()

    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    # corr_matrix = housing.corr()
    # corr_matrix["median_house_value"].sort_values(ascending=False)
    logging.info("Preparation of dataset Done")

    return housing, strat_train_set, strat_test_set


def feature_engineering(housing, strat_train_set):
    """
    Does feature engineering on the dataset

    Args:
        housing (Dataframe): The data that is to be used in model creation.
        strat_train_set (Dataframe): The train dataset

    Returns:
        housing (Dataframe): The modified housing dataframe for model training
        housing_labels (Series): The target labels data
    """

    # Get features and labels from dataset
    housing = strat_train_set.drop("median_house_value", axis=1)
    # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
    logging.info("Feature Engineering performed")
    return housing, housing_labels


def fill_missing_values(housing):
    """
    Does feature engineering on the dataset

    Args:
        housing (Dataframe): The data that is to be used in model creation.

    Returns:
        housing_prepared (Dataframe): The modified housing_prepared\
              dataframe for model training
        imputer (SimpleImputer): The imputer function to deal with missing data
    """
    # Fill out missing values and add new columns

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                              index=housing.index)
    col_names = list(housing_tr.columns)
    # housing_tr["rooms_per_household"] = housing_tr["total_rooms"] \
    #     / housing_tr["households"]
    # housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] \
    #     / housing_tr["total_rooms"]
    # housing_tr["population_per_household"] = housing_tr["population"] \
    #     / housing_tr["households"]

    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
    housing_extra_attribs, new_cols = attr_adder.transform(
        housing_tr.values)

    col_names += new_cols
    housing_extra_attribs_df = pd.DataFrame(
        housing_extra_attribs,
        columns=col_names)

    housing_cat = housing[['ocean_proximity']]
    housing_prepared = housing_extra_attribs_df.join(pd.get_dummies(
        housing_cat, drop_first=True))

    logging.info("Missing values are filled and new attributes are derived")
    return housing_prepared, imputer
