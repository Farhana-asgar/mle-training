import argparse

from house_value_prediction.score import (
    dec_tree_scoring,
    lin_reg_scoring,
    random_forest_scoring,
)


def main(model_location, dataset_location):

    lin_reg_scoring(model_location)

    dec_tree_scoring(model_location)

    random_forest_scoring(model_location, dataset_location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_location', type=str,
                        help='Dataset location')

    parser.add_argument('--model_location', type=str,
                        help='Model location')

    args = parser.parse_args()

    main(args.model_location, args.dataset_location)

    args = parser.parse_args()