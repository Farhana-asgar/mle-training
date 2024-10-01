import argparse
import pickle

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


def lin_reg(housing_prepared, housing_labels):
    # Linear Regression model implementation

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    # housing_predictions = lin_reg.predict(housing_prepared)
    # return housing_predictions
    return lin_reg


def dec_tree(housing_prepared, housing_labels):
    # Decision Tree model implementation
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    # housing_predictions = tree_reg.predict(housing_prepared)
    # return housing_predictions
    return tree_reg


def random_forest(housing_prepared, housing_labels, strat_test_set, imputer, dataset_location):
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
    X_test_prepared["rooms_per_household"] = X_test_prepared["total_rooms"] \
        / X_test_prepared["households"]
    X_test_prepared["bedrooms_per_room"] = X_test_prepared["total_bedrooms"]\
        / X_test_prepared["total_rooms"]
    X_test_prepared["population_per_household"] = \
        X_test_prepared["population"] / X_test_prepared["households"]
    X_test_cat = X_test[['ocean_proximity']]
    X_test_prepared = X_test_prepared.join(pd.get_dummies
                                           (X_test_cat, drop_first=True))

    X_test_prepared.to_csv(dataset_location+'/X_test_prepared.csv',index=False)
    # final_predictions = final_model.predict(X_test_prepared)

    # return y_test, final_predictions

    return y_test, final_model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_location', type=str,
                        help='Dataset location')

    parser.add_argument('--model_location', type=str, help='Model location')

    args = parser.parse_args()

    housing_prepared = pd.read_csv(args.dataset_location +
                                   '/housing_prepared.csv')

    housing_labels_df = pd.read_csv(args.dataset_location +
                                    '/housing_labels.csv')

    housing_labels = housing_labels_df['median_house_value']

    with open(args.dataset_location+'/imputer.pkl', 'rb') as file:
        imputer = pickle.load(file)

    strat_test_set = pd.read_csv(args.dataset_location+'/strat_test_set.csv')

    lin_reg_model = lin_reg(housing_prepared, housing_labels)

    print("Obtained Linear Regression Model")

    dec_tree_model = dec_tree(housing_prepared, housing_labels)

    print("Obtianed Decision Tree Model")

    y_test, random_forest_model = random_forest(housing_prepared,
                                                housing_labels, strat_test_set,
                                                imputer, args.dataset_location)

    print("Obtained Random Forest Model")

    with open(args.model_location + '/lin_reg.pkl', 'wb') as file:
        pickle.dump(lin_reg_model, file)
    with open(args.model_location + '/dec_tree.pkl', 'wb') as file:
        pickle.dump(dec_tree_model, file)
    with open(args.model_location + '/random_forest.pkl', 'wb') as file:
        pickle.dump(random_forest_model, file)

    y_test.to_csv(args.model_location+'/y_test.csv', index=False)

    print("All models saved")


if __name__ == "__main__":
    main()
