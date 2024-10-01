import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def lin_reg_scoring(model_location):
    # Scoring for Linear Regression implementation

    with open(model_location+'/lin_reg.pkl', 'rb') as file:
        lin_reg_model = pickle.load(file)

    housing_predictions = lin_reg_model.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print("Root Mean Squared Error for Linear Regression - {}".
          format(lin_rmse))

    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    print("Mean Absolute Error for Linear Regression - {} \n".
          format(lin_mae))
    return lin_rmse, lin_mae


def dec_tree_scoring(model_location):
    # Scoring for Decesion Tree implementation
    with open(model_location+'/dec_tree.pkl', 'rb') as file:
        dec_tree_model = pickle.load(file)

    housing_predictions = dec_tree_model.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    print("Mean Squared Error for Decision Tree - {}".format(tree_mse))

    tree_rmse = np.sqrt(tree_mse)
    print("Root Mean Squared Error for Decision Tree - {}".format(tree_rmse))
    return tree_mse, tree_rmse


def random_forest_scoring(model_location, dataset_location):
    # Scoring for Random Forest implementation
    with open(model_location+'/random_forest.pkl', 'rb') as file:
        random_forest_model = pickle.load(file)
    y_test_df = pd.read_csv(model_location+'/y_test.csv')
    y_test = y_test_df["median_house_value"]
    X_test_prepared = pd.read_csv(dataset_location+'/X_test_prepared.csv')

    housing_predictions = random_forest_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, housing_predictions)
    final_rmse = np.sqrt(final_mse)
    print("\n Root Mean Squared Error of Random Forest - {}".
          format(final_mse))
    print("Root Mean Squared Error of Random Forest - {}".
          format(final_rmse))
    return final_mse, final_rmse


def main(model_location, dataset_location):
    print("latest changes")

    lin_reg_scoring(model_location)

    dec_tree_scoring(model_location)

    random_forest_scoring(model_location, dataset_location)


def global_variable_initialization(dataset_location):

    global housing_prepared
    global housing_labels

    housing_prepared = pd.read_csv(dataset_location +
                                   '/housing_prepared.csv')
    housing_labels_df = pd.read_csv(dataset_location +
                                    '/housing_labels.csv')
    housing_labels = housing_labels_df["median_house_value"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_location', type=str,
                        help='Dataset location')

    parser.add_argument('--model_location', type=str,
                        help='Model location')

    args = parser.parse_args()

    global_variable_initialization(args.dataset_location)

    main(args.model_location, args.dataset_location)
