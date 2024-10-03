import argparse
import pickle

from house_value_prediction.ingest_data import (
    feature_engineering,
    fetch_housing_data,
    fill_missing_values,
    load_housing_data,
    prepare_dataset,
)


def read_input():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_location', type=str,
                        help='Dataset location')

    args = parser.parse_args()

    return args.dataset_location


def main():

    dataset_location = read_input()

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

    print("Training Data saved")


if __name__ == "__main__":
    main()
