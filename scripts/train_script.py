import argparse
import pickle

import pandas as pd

from house_value_prediction.train import dec_tree, lin_reg, random_forest


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_location', type=str,
                        help='Dataset location')

    parser.add_argument('--model_location', type=str, help='Model location')

    args = parser.parse_args()

    housing_prepared = pd.read_csv(args.dataset_location +
                                   '/housing_prepared.csv')

    housing_labels_df = pd.read_csv(args.dataset_location +
                                    '/housing_labels.csv')

    housing_labels = housing_labels_df['median_house_value']

    with open(args.dataset_location+'/imputer.pkl', 'rb') as file:
        imputer = pickle.load(file)

    strat_test_set = pd.read_csv(args.dataset_location+'/strat_test_set.csv')

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

    print("All models saved")


if __name__ == "__main__":
    main()
