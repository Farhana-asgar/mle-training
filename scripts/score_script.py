import argparse
import logging
import logging.config

from house_value_prediction.score import (
    dec_tree_scoring,
    global_variable_initialization,
    lin_reg_scoring,
    random_forest_scoring,
)

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


def main(model_location, dataset_location):

    global_variable_initialization(dataset_location)

    lin_reg_scoring(model_location)

    dec_tree_scoring(model_location)

    random_forest_scoring(model_location, dataset_location)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_location', type=str,
                        help='Dataset location')

    parser.add_argument('--model_location', type=str,
                        help='Model location')
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
    logger.info("In score_script.py")
    logger.info("---Scoring starts---")

    args = parser.parse_args()

    logger.info(f"Read the inputs - Dataset location  {args.dataset_location} \
                and Model location {args.model_location}")

    main(args.model_location, args.dataset_location)
