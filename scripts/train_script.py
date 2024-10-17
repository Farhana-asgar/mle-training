import argparse
import logging
import logging.config
import pickle

import pandas as pd

from house_value_prediction.train import dec_tree, lin_reg, random_forest

LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:\
                %(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(message)s"},
    },
    "root": {"level": "DEBUG"},
}


def configure_logger(
        logger=None, cfg=None, log_path=None, console=True, log_level="DEBUG"):

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_location', type=str,
                        help='Dataset location')
    parser.add_argument('--model_location', type=str, help='Model location')
    parser.add_argument('--log-level', type=str, help='Describe the log level')
    parser.add_argument('--log-path', type=str,
                        help='Describe whether or not to write logs to a file')
    parser.add_argument('--no-console-log',  action='store_true',
                        help='Describe whether or not to write logs to a file')
    args = parser.parse_args()

    if not args.log_level:
        args.log_level = 'DEBUG'
    logger = configure_logger(log_path=args.log_path, log_level=args.log_level,
                              console=not args.no_console_log)

    logger.info("In train_script.py")
    logger.info("---Training Starts---")

    logger.info(f"Read the inputs - Dataset location  {args.dataset_location} \
                and Model location {args.model_location}")

    housing_prepared = pd.read_csv(args.dataset_location +
                                   '/housing_prepared.csv')

    housing_labels_df = pd.read_csv(args.dataset_location +
                                    '/housing_labels.csv')

    housing_labels = housing_labels_df['median_house_value']

    with open(args.dataset_location+'/imputer.pkl', 'rb') as file:
        imputer = pickle.load(file)

    strat_test_set = pd.read_csv(args.dataset_location+'/strat_test_set.csv')
    logger.info("All the required training data are loaded")

    nan_rows = housing_prepared[housing_prepared.isna().any(axis=1)]
    print(nan_rows)

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
    logger.info("All 3 models are saved")

    print("All models saved")


if __name__ == "__main__":
    main()
