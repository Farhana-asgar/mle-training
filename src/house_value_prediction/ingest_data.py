
import logging
import logging.config
import os
import pickle
import tarfile

import mlflow
import numpy as np
import pandas as pd
from logging_tree import printout
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

remote_server_uri = "http://localhost:5000"
mlflow.set_tracking_uri(remote_server_uri)

exp_name = "ElasticNet_wine"
mlflow.set_experiment(exp_name)


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, df):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

        # Avoid division by zero by adding a small epsilon
        households = df.iloc[:, households_ix]
        epsilon = 1e-10  # Small constant to avoid division by zero

        # Calculate features
        df['rooms_per_household'] = df.iloc[:, rooms_ix] / (
            households + epsilon)
        df['population_per_household'] = df.iloc[:, population_ix] / (
            households + epsilon)

        df['bedrooms_per_room'] = df.iloc[:, bedrooms_ix] / (
                df.iloc[:, rooms_ix] + epsilon)

        # Specify the names of the new columns
        return df


class IngestData:

    def __init__(self, dataset_location,
                 log_path='script_logs/ingest_data_logs.txt',
                 log_level='INFO', no_console_log=False, housing_url=None,
                 housing_path=None):
        with mlflow.start_run(nested=True, run_name="Ingest Data"):
            print(f"Running experiment: \
                  {mlflow.active_run().info.experiment_id}")
            print(mlflow.active_run().info.run_id)

            self.logger = configure_logger(log_path=log_path,
                                           log_level=log_level,
                                           console=not no_console_log)
            if not housing_url:
                DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
                self.housing_url = DOWNLOAD_ROOT +\
                    "datasets/housing/housing.tgz"

            if not housing_path:
                BASE_PATH = os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))
                self.housing_path = os.path.join(BASE_PATH, "datasets",
                                                 "housing")

            self.logger.info("In ingest_data_script.py")
            self.logger.info("---Data Ingestion Starts---")
            self.logger.info(f"Read the inputs - Dataset location  \
                             {dataset_location}")
            self.fetch_housing_data()

            housing = self.load_housing_data()

            housing, strat_train_set, strat_test_set =\
                self.prepare_dataset(housing)

            housing, housing_labels = self.feature_engineering(
                housing, strat_train_set)

            housing_prepared, imputer = self.fill_missing_values(
                housing)

            housing_prepared.to_csv(dataset_location+'/housing_prepared.csv',
                                    index=False)

            strat_test_set.to_csv(dataset_location+'/strat_test_set.csv',
                                  index=False)

            housing_labels.to_csv(dataset_location+'/housing_labels.csv',
                                  index=False)

            with open(dataset_location+'/imputer.pkl', 'wb') as file:
                pickle.dump(imputer, file)

            mlflow.log_metric("Dataset size", housing_prepared.shape[0])
            mlflow.log_artifact(dataset_location + '/housing_prepared.csv')
            mlflow.log_artifact(dataset_location + '/strat_test_set.csv')
            mlflow.log_artifact(dataset_location + '/housing_labels.csv')
            mlflow.log_artifact(dataset_location + '/imputer.pkl')

            self.logger.info("Training Data saved")

            printout()

    def fetch_housing_data(self):
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
        os.makedirs(self.housing_path, exist_ok=True)
        tgz_path = os.path.join(self.housing_path, "housing.tgz")
        urllib.request.urlretrieve(self.housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=self.housing_path)
        self.logger.info(f"Read and downloaded the data from \
                         {self.housing_url}")
        housing_tgz.close()

    def load_housing_data(self):
        """
        Reads the housing dataset

        Args:
            housing_path (str): Local directory path where the dataset \
                is stored

        Returns:
            Dataframe: The function returns the contents of the csv file

        """

        # This function is to load the dataset housing.tgz
        csv_path = os.path.join(self.housing_path, "housing.csv")
        self.logger.info("Housing data loaded")
        return pd.read_csv(csv_path)

    def income_cat_proportions(self, data):
        """
        Downloads and extracts the housing dataset.

        Args:
            data (Dataframe): The data that is to be used in model creation.

        Returns:
            Series: Proportion of income
        """
        # Returns proportion of income
        return (data["income_cat"].value_counts() / len(data))

    def prepare_dataset(self, housing):
        """
        Downloads and extracts the housing dataset.

        Args:
            housing (Dataframe): The data that is to be used in model creation.

        Returns:
            housing (Dataframe): Modified housing dataframe
            strat_train_set (Dataframe): Split from housing dataset for \
                training
            strat_test_set (Dataframe): Split from housing dataset for testing
        """
        # Prepare and split the dataset for training and testing purpose
        housing["income_cat"] = pd.cut(housing["median_income"],
                                       bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                       labels=[1, 2, 3, 4, 5])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                       random_state=42)
        for train_index, test_index in split.split(housing,
                                                   housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]
        train_set, test_set = train_test_split(housing, test_size=0.2,
                                               random_state=42)

        compare_props = pd.DataFrame({
            "Overall": self.income_cat_proportions(housing),
            "Stratified": self.income_cat_proportions(strat_test_set),
            "Random": self.income_cat_proportions(test_set),
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

    def feature_engineering(self, housing, strat_train_set):
        """
        Does feature engineering on the dataset

        Args:
            housing (Dataframe): The data that is to be used in model creation.
            strat_train_set (Dataframe): The train dataset

        Returns:
            housing (Dataframe): The modified housing dataframe for model \
                training
            housing_labels (Series): The target labels data
        """

        # Get features and labels from dataset
        housing = strat_train_set.drop("median_house_value", axis=1)
        # drop labels for training set
        housing_labels = strat_train_set["median_house_value"].copy()
        logging.info("Feature Engineering performed")
        return housing, housing_labels

    def fill_missing_values(self, housing):
        """
        Does feature engineering on the dataset

        Args:
            housing (Dataframe): The data that is to be used in model creation.

        Returns:
            housing_prepared (Dataframe): The modified housing_prepared\
                dataframe for model training
            imputer (SimpleImputer): The imputer function to deal with missing\
                data
        """
        # Fill out missing values and add new columns

        imputer = SimpleImputer(strategy="median")

        housing_num = housing.drop("ocean_proximity", axis=1)

        imputer.fit(housing_num)
        X = imputer.transform(housing_num)

        housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                                  index=housing.index)
        # housing_tr["rooms_per_household"] = housing_tr["total_rooms"] \
        #     / housing_tr["households"]
        # housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] \
        #     / housing_tr["total_rooms"]
        # housing_tr["population_per_household"] = housing_tr["population"] \
        #     / housing_tr["households"]

        attr_adder = CombinedAttributesAdder()
        housing_extra_attribs = attr_adder.transform(
            housing_tr)

        housing_cat = housing[['ocean_proximity']]

        housing_prepared = housing_extra_attribs.join(pd.get_dummies(
            housing_cat, drop_first=False))

        logging.info("Missing values are filled and new attributes are derived"
                     )
        return housing_prepared, imputer


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
            print(logging)
            print(log_level)

            sh.setLevel(getattr(logging, log_level))
            formatter = logging.Formatter(LOGGING_DEFAULT_CONFIG['formatters']
                                          ['default']['format'],
                                          datefmt=LOGGING_DEFAULT_CONFIG
                                          ['formatters']['default']['datefmt'])
            # Set the formatter for the console handler
            sh.setFormatter(formatter)
            logger.addHandler(sh)

    return logger
