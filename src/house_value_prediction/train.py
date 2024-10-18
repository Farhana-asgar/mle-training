import logging

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


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


def lin_reg(housing_prepared, housing_labels):
    """
    Develops Linear Regression Model

    Args:
        housing_prepared (Dataframe): The training data
        housing_labels (Series): The label data

    Returns:
        lin_reg (sklearn.linear_model._base.LinearRegression): \
            The Linear Regression Model
    """
    # Linear Regression model implementation

    lin_reg = LinearRegression()
    print(housing_prepared.columns)
    lin_reg.fit(housing_prepared, housing_labels)

    # housing_predictions = lin_reg.predict(housing_prepared)
    # return housing_predictions
    logger.info("Linear Regression Model Ready")
    return lin_reg


def dec_tree(housing_prepared, housing_labels):
    """
    Develops Decision Tree  Model

    Args:
        housing_prepared (Dataframe): The training data
        housing_labels (Series): The label data

    Returns:
        tree_reg (sklearn.tree._classes.DecisionTreeRegressor): \
            The Decision Tree Model
    """
    # Decision Tree model implementation
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    # housing_predictions = tree_reg.predict(housing_prepared)
    # return housing_predictions
    logger.info("Decision Tree Model Ready")
    return tree_reg


def random_forest(housing_prepared, housing_labels, strat_test_set, imputer,
                  dataset_location):
    """
    Develops Random Forest  Model

    Args:
        housing_prepared (Dataframe): The training data
        housing_labels (Series): The label data
        strat_test_set (Dataframe): Split from housing dataset for testing
        imputer (SimpleImputer): The imputer function to deal with missing data
        dataset_location (str): The location where the dataset is stored

    Returns:
        final_model (sklearn.tree._classes.DecisionTreeRegressor): \
            The Decision Tree Model
        y_test (Series): The label column value
    """
    # Random Forest model implementation
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error", random_state=42,)
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    print("Random Forest Hyperparameter tuning results:")
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print("Root Mean Squared Error - {}".format((np.sqrt(-mean_score),
                                                    params)))

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10],
         "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)
    grid_search.best_params_
    cvres = grid_search.cv_results_
    print("\n Random Forest Grid Search Hyper parameter tuning")
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print("Root Mean Squared Error - {}".format((np.sqrt(-mean_score),
                                                     params)))

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    final_model = grid_search.best_estimator_

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(X_test_prepared, columns=X_test_num.columns,
                                   index=X_test.index)
    col_names = list(X_test_prepared.columns)
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
    X_test_prepared_attribs, new_cols = attr_adder.transform(
        X_test_prepared.values)

    col_names += new_cols
    X_test_prepared_df = pd.DataFrame(
        X_test_prepared_attribs,
        columns=col_names)

    # X_test_prepared["rooms_per_household"] = X_test_prepared["total_rooms"] \
    #     / X_test_prepared["households"]
    # X_test_prepared["bedrooms_per_room"] = X_test_prepared["total_bedrooms"]\
    #     / X_test_prepared["total_rooms"]
    # X_test_prepared["population_per_household"] = \
    #     X_test_prepared["population"] / X_test_prepared["households"]
    X_test_cat = X_test[['ocean_proximity']]
    X_test_prepared = X_test_prepared_df.join(pd.get_dummies
                                              (X_test_cat, drop_first=True))

    X_test_prepared.to_csv(dataset_location+'/X_test_prepared.csv',
                           index=False)
    # final_predictions = final_model.predict(X_test_prepared)

    # return y_test, final_predictions
    logger.info("Random Forest Model Ready")

    return y_test, final_model
