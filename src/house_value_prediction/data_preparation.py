import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from .data_ingestion import income_cat_proportions

# Local dev
# from data_ingestion import income_cat_proportions


def prepare_dataset(housing):
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

    return housing, strat_train_set, strat_test_set


def feature_engineering(housing, strat_train_set):
    # Get features and labels from dataset
    housing = strat_train_set.drop("median_house_value", axis=1)
    # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
    return housing, housing_labels


def fill_missing_values(housing):
    # Fill out missing values and add new columns

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                              index=housing.index)
    housing_tr["rooms_per_household"] = housing_tr["total_rooms"] \
        / housing_tr["households"]
    housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] \
        / housing_tr["total_rooms"]
    housing_tr["population_per_household"] = housing_tr["population"] \
        / housing_tr["households"]

    housing_cat = housing[['ocean_proximity']]
    housing_prepared = housing_tr.join(pd.get_dummies(
        housing_cat, drop_first=True))
    return housing_prepared, imputer
