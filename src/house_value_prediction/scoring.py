import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def line_reg_scoring(housing_predictions, housing_labels):
    # Scoring for Linear Regression implementation
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print("Root Mean Squared Error for Linear Regression - {}".
          format(lin_rmse))

    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    print("Mean Absolute Error for Linear Regression - {} \n".
          format(lin_mae))
    return lin_rmse, lin_mae


def dec_tree_scoring(housing_predictions, housing_labels):
    # Scoring for Decesion Tree implementation
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    print("Mean Squared Error for Decision Tree - {}".format(tree_mse))

    tree_rmse = np.sqrt(tree_mse)
    print("Root Mean Squared Error for Decision Tree - {}".format(tree_rmse))
    return tree_mse, tree_rmse


def random_forest_scoring(y_test, final_predictions):
    # Scoring for Random Forest implementation
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("\n Root Mean Squared Error of Random Forest - {}".
          format(final_mse))
    print("Root Mean Squared Error of Random Forest - {}".
          format(final_rmse))
    return final_mse, final_rmse

