import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def lin_reg_scoring(model_location):
    """
    Scoring for Linear Regression implementation

    Args:
        model_location (str): The location where the model is stored

    Returns:
        lin_rmse (float): Root Mean Squared Error for Linear Regression
        lin_mae (float): Mean Absolute Error for Linear Regression
    """
    # Scoring for Linear Regression implementation

    with open(model_location+'/lin_reg.pkl', 'rb') as file:
        lin_reg_model = pickle.load(file)
    logging.info("Linear Regression Model Saved")
    print(housing_prepared.columns)

    housing_predictions = lin_reg_model.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print("Root Mean Squared Error for Linear Regression - {}".
          format(lin_rmse))
    logging.info(f"Root Mean Squared Error for Linear Regression - {lin_rmse}")

    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    print("Mean Absolute Error for Linear Regression - {} \n".
          format(lin_mae))
    logging.info(f"Mean Absolute Error for Linear Regression - {lin_mae}")
    return lin_rmse, lin_mae


def dec_tree_scoring(model_location):
    """
    Scoring for Decision Tree implementation

    Args:
        model_location (str): The location where the model is stored

    Returns:
        tree_mse (float): Mean Squared Error for Decision Tree
        tree_rmse (float): Root Mean Squared Error for Decision Tree

    """
    # Scoring for Decision Tree implementation
    with open(model_location+'/dec_tree.pkl', 'rb') as file:
        dec_tree_model = pickle.load(file)
    logging.info("Decision Tree Model Saved")
    print(housing_prepared.columns)

    housing_predictions = dec_tree_model.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    print("Mean Squared Error for Decision Tree - {}".format(tree_mse))
    logging.info(f"Mean Squared Error for Decision Tree - {tree_mse}")

    tree_rmse = np.sqrt(tree_mse)
    print("Root Mean Squared Error for Decision Tree - {}".format(tree_rmse))
    logging.info(f"Root Mean Squared Error for Decision Tree - {tree_rmse}")
    return tree_mse, tree_rmse


def random_forest_scoring(model_location, dataset_location):
    """
    Scoring for Random Forest implementation

    Args:
        model_location (str): The location where the model is stored
        dataset_location (str): The location where the dataset is stored

    Returns:
        tree_mse (float): Mean Squared Error for Random Forest
        tree_rmse (float): Root Mean Squared Error for Random Forest
    """

    # Scoring for Random Forest implementation
    with open(model_location+'/random_forest.pkl', 'rb') as file:
        random_forest_model = pickle.load(file)
    logging.info("Random Forest Model Saved")
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

    logging.info(f"\n Root Mean Squared Error of Random Forest - {final_mse}")
    logging.info(f"Root Mean Squared Error of Random Forest - {final_rmse}")
    return final_mse, final_rmse


def global_variable_initialization(dataset_location):
    """
    Assigning global variables

    Args:
        dataset_location (str): The location where the dataset is stored

    Returns:
        None: The function does not return anything
    """

    global housing_prepared
    global housing_labels

    housing_prepared = pd.read_csv(dataset_location +
                                   '/housing_prepared.csv')
    housing_labels_df = pd.read_csv(dataset_location +
                                    '/housing_labels.csv')
    housing_labels = housing_labels_df["median_house_value"]
    housing_labels = housing_labels_df["median_house_value"]
    logging.info(f"Global variables {housing_prepared} and {housing_labels} \
                 are initialized")
