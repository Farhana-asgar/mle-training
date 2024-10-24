import logging
import logging.config
import pickle

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

remote_server_uri = "http://localhost:5000"
mlflow.set_tracking_uri(remote_server_uri)

exp_name = "ElasticNet_wine"
mlflow.set_experiment(exp_name)


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):  # no *self or **kself
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


class Train:
    def __init__(self, dataset_location,
                 model_location,
                 log_path='script_logs/train_logs.txt',
                 log_level='INFO',
                 no_console_log=False):

        with mlflow.start_run(nested=True, run_name="Train"):
            print(f"Running experiment: \
                  {mlflow.active_run().info.experiment_id}")
            print(mlflow.active_run().info.run_id)

            self.logger = configure_logger(log_path=log_path,
                                           log_level=log_level,
                                           console=not no_console_log)

            self.logger.info("In train_script.py")
            self.logger.info("---Training Starts---")

            self.logger.info(f"Read the inputs - Dataset location  \
                            {dataset_location} and Model location\
                                {model_location}")

            housing_prepared = pd.read_csv(dataset_location +
                                           '/housing_prepared.csv')

            housing_labels_df = pd.read_csv(dataset_location +
                                            '/housing_labels.csv')

            housing_labels = housing_labels_df['median_house_value']

            with open(dataset_location+'/imputer.pkl', 'rb') as file:
                imputer = pickle.load(file)

            strat_test_set = pd.read_csv(dataset_location +
                                         '/strat_test_set.csv')
            self.logger.info("All the required training data are loaded")

            self.lin_reg_model = self.lin_reg(housing_prepared, housing_labels)

            self.dec_tree_model = self.dec_tree(housing_prepared,
                                                housing_labels)

            self.y_test, self.random_forest_model = self.random_forest(
                housing_prepared,
                housing_labels,
                strat_test_set,
                imputer,
                dataset_location)

            with open(model_location + '/lin_reg.pkl', 'wb') as file:
                pickle.dump(self.lin_reg_model, file)
            with open(model_location + '/dec_tree.pkl', 'wb') as file:
                pickle.dump(self.dec_tree_model, file)
            with open(model_location + '/random_forest.pkl',
                      'wb')as file:
                pickle.dump(self.random_forest_model, file)

            self.y_test.to_csv(model_location+'/y_test.csv', index=False)
            self.logger.info("All 3 models are saved")

            print("All models saved")

    def lin_reg(self, housing_prepared, housing_labels):
        """
        Develops Linear Regression Model

        self:
            housing_prepared (Dataframe): The training data
            housing_labels (Series): The label data

        Returns:
            lin_reg (sklearn.linear_model._base.LinearRegression): \
                The Linear Regression Model
        """
        # Linear Regression model implementation

        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)

        # housing_predictions = lin_reg.predict(housing_prepared)
        # return housing_predictions
        self.logger.info("Linear Regression Model Ready")
        return lin_reg

    def dec_tree(self, housing_prepared, housing_labels):
        """
        Develops Decision Tree  Model

        self:
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
        self.logger.info("Decision Tree Model Ready")
        return tree_reg

    def random_forest(self, housing_prepared, housing_labels, strat_test_set,
                      imputer, dataset_location):
        """
        Develops Random Forest  Model

        self:
            housing_prepared (Dataframe): The training data
            housing_labels (Series): The label data
            strat_test_set (Dataframe): Split from housing dataset for testing
            imputer (SimpleImputer): The imputer function to deal with \
                missing data
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
        for mean_score, params in zip(cvres["mean_test_score"],
                                      cvres["params"]):
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
        # train across 5 folds, that's a total of (12+6)*5=90 rounds \
        # of training
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                                   scoring='neg_mean_squared_error',
                                   return_train_score=True)
        grid_search.fit(housing_prepared, housing_labels)
        grid_search.best_params_
        cvres = grid_search.cv_results_
        print("\n Random Forest Grid Search Hyper parameter tuning")
        for mean_score, params in zip(cvres["mean_test_score"],
                                      cvres["params"]):
            print("Root Mean Squared Error - {}".format((np.sqrt(-mean_score),
                                                        params)))

        feature_importances = grid_search.best_estimator_.feature_importances_
        sorted(zip(feature_importances, housing_prepared.columns),
               reverse=True)

        final_model = grid_search.best_estimator_

        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()

        X_test_num = X_test.drop("ocean_proximity", axis=1)
        X_test_prepared = imputer.transform(X_test_num)
        X_test_prepared = pd.DataFrame(X_test_prepared,
                                       columns=X_test_num.columns,
                                       index=X_test.index)
        col_names = list(X_test_prepared.columns)
        attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
        X_test_prepared_attribs, new_cols = attr_adder.transform(
            X_test_prepared.values)

        col_names += new_cols
        X_test_prepared_df = pd.DataFrame(
            X_test_prepared_attribs,
            columns=col_names)
        X_test_cat = X_test[['ocean_proximity']]
        X_test_prepared = X_test_prepared_df.join(pd.get_dummies
                                                  (X_test_cat,
                                                   drop_first=False))

        X_test_prepared.to_csv(dataset_location+'/X_test_prepared.csv',
                               index=False)
        # final_predictions = final_model.predict(X_test_prepared)

        # return y_test, final_predictions
        self.logger.info("Random Forest Model Ready")

        return y_test, final_model


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
