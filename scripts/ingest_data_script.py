import argparse
import logging
import logging.config
import pickle

from logging_tree import printout

from house_value_prediction.ingest_data import (
    feature_engineering,
    fetch_housing_data,
    fill_missing_values,
    load_housing_data,
    prepare_dataset,
)

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


def read_input():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_location', type=str,
                        help='Dataset location')
    parser.add_argument('--log-level', type=str, help='Describe the log level')
    parser.add_argument('--log-path', type=str,
                        help='Describe whether or not to write logs to a file')
    parser.add_argument('--no-console-log',  action='store_true',
                        help='Describe whether or not to write logs to a file')

    args = parser.parse_args()

    return args.dataset_location, args.log_level, args.log_path, \
        args.no_console_log


def main():

    dataset_location, log_level, log_path, no_console_log = read_input()
    if not log_level:
        log_level = 'DEBUG'

    logger = configure_logger(log_path=log_path, log_level=log_level,
                              console=not no_console_log)

    logger.info("In ingest_data_script.py")
    logger.info("---Data Ingestion Starts---")

    logger.info(f"Read the inputs - Dataset location  {dataset_location}")

    fetch_housing_data()

    housing = load_housing_data()

    prepare_dataset(housing)

    housing, strat_train_set, strat_test_set = prepare_dataset(housing)

    housing, housing_labels = feature_engineering(housing, strat_train_set)

    housing_prepared, imputer = fill_missing_values(housing)

    housing_prepared.to_csv(dataset_location+'/housing_prepared.csv',
                            index=False)

    strat_test_set.to_csv(dataset_location+'/strat_test_set.csv',
                          index=False)

    housing_labels.to_csv(dataset_location+'/housing_labels.csv',
                          index=False)

    with open(dataset_location+'/imputer.pkl', 'wb') as file:
        pickle.dump(imputer, file)

    logger.info("Training Data saved")

    printout()


if __name__ == "__main__":
    main()
