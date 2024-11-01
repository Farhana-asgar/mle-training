import logging
import pickle

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Score:
    def __init__(self, dataset_location,
                 model_location, no_console_log,
                 log_path='./scripts/script_logs/score_logs.txt', log_level='INFO',
                 ):

        with mlflow.start_run(nested=True, run_name="Score"):
            print(f"Running experiment: \
                {mlflow.active_run().info.experiment_id}")
            print(mlflow.active_run().info.run_id)

            self.logger = configure_logger(log_path=log_path,
                                           log_level=log_level,
                                           console=not no_console_log)
            self.global_variable_initialization(dataset_location)

            lin_rmse, lin_mae = self.lin_reg_scoring(model_location)
            mlflow.log_metric("lin_reg_rmse", lin_rmse)
            mlflow.log_metric("lin_reg_mae", lin_mae)

            tree_mse, tree_rmse = self.dec_tree_scoring(model_location)
            mlflow.log_metric("dec_tree_mse", tree_mse)
            mlflow.log_metric("dec_tree_rmse", tree_rmse)

            random_forest_mse, random_forest_rmse = self.random_forest_scoring(
                model_location, dataset_location)
            mlflow.log_metric("random_forest_mse", random_forest_mse)
            mlflow.log_metric("random_forest_rmse", random_forest_rmse)

    def lin_reg_scoring(self, model_location):
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

        housing_predictions = lin_reg_model.predict(housing_prepared)
        lin_mse = mean_squared_error(housing_labels, housing_predictions)
        lin_rmse = np.sqrt(lin_mse)
        print("Root Mean Squared Error for Linear Regression - {}".
              format(lin_rmse))
        logging.info(f"Root Mean Squared Error for Linear Regression -\
                     {lin_rmse}")

        lin_mae = mean_absolute_error(housing_labels, housing_predictions)
        print("Mean Absolute Error for Linear Regression - {} \n".
              format(lin_mae))
        logging.info(f"Mean Absolute Error for Linear Regression - {lin_mae}")
        return lin_rmse, lin_mae

    def dec_tree_scoring(self, model_location):
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

        housing_predictions = dec_tree_model.predict(housing_prepared)
        tree_mse = mean_squared_error(housing_labels, housing_predictions)
        print("Mean Squared Error for Decision Tree - {}".format(tree_mse))
        logging.info(f"Mean Squared Error for Decision Tree - {tree_mse}")

        tree_rmse = np.sqrt(tree_mse)
        print("Root Mean Squared Error for Decision Tree - {}".format
              (tree_rmse))
        logging.info(f"Root Mean Squared Error for Decision Tree - \
                     {tree_rmse}")
        return tree_mse, tree_rmse

    def random_forest_scoring(self, model_location, dataset_location):
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

        logging.info(f"\n Root Mean Squared Error of Random Forest - \
                     {final_mse}")
        logging.info(f"Root Mean Squared Error of Random Forest - \
                     {final_rmse}")
        return final_mse, final_rmse

    def global_variable_initialization(self, dataset_location):
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
        logging.info(f"Global variables {housing_prepared} and \
                     {housing_labels} are initialized")


LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - \
                %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(message)s"},
    },
    "root": {"level": "INFO"},
}


def configure_logger(
        logger=None, cfg=None, log_path=None, console=True, log_level="INFO"):

    if not cfg:
        logging.config.dictConfig(LOGGING_DEFAULT_CONFIG)
    else:
        logging.config.dictConfig(cfg)

    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    logger = logger or logging.getLogger()

    if log_path or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_path:
            fh = logging.FileHandler(log_path, 'w')
            fh.setLevel(getattr(logging, log_level))
            # Set the formatter for the file handler
            formatter = logging.Formatter(LOGGING_DEFAULT_CONFIG['formatters']
                                          ['default']['format'],
                                          datefmt=LOGGING_DEFAULT_CONFIG
                                          ['formatters']['default']['datefmt'])
            fh.setFormatter(formatter)  # Set the formatter
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()

            sh.setLevel(getattr(logging, log_level))
            formatter = logging.Formatter(LOGGING_DEFAULT_CONFIG['formatters']
                                          ['default']['format'],
                                          datefmt=LOGGING_DEFAULT_CONFIG
                                          ['formatters']['default']['datefmt'])
            # Set the formatter for the console handler
            sh.setFormatter(formatter)
            logger.addHandler(sh)

    return logger
