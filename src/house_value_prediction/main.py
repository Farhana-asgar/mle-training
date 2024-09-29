# Local dev
# from data_ingestion import fetch_housing_data, load_housing_data
# from data_preparation import (  # noqa: E401, E501
#     feature_engineering,
#     fill_missing_values,
#     prepare_dataset,
# )
# from models import dec_tree, lin_reg, random_forest
# from scoring import dec_tree_scoring, line_reg_scoring, random_forest_scoring

from .data_ingestion import fetch_housing_data, load_housing_data
from .data_preparation import (  # Imports for data preparation
    feature_engineering,
    fill_missing_values,
    prepare_dataset,
)
from .models import dec_tree, lin_reg, random_forest
from .scoring import dec_tree_scoring, line_reg_scoring, random_forest_scoring


def main():

    # Call all the functions to execute house value prediction algorithm

    fetch_housing_data()

    housing = load_housing_data()

    prepare_dataset(housing)

    housing, strat_train_set, strat_test_set = prepare_dataset(housing)

    housing, housing_labels = feature_engineering(housing, strat_train_set)

    housing_prepared, imputer = fill_missing_values(housing)

    lin_reg_housing_predictions = lin_reg(housing_prepared, housing_labels)

    line_reg_scoring(lin_reg_housing_predictions, housing_labels)

    dec_tree_housing_predictions = dec_tree(housing_prepared, housing_labels)

    dec_tree_scoring(dec_tree_housing_predictions, housing_labels)

    y_test, final_predictions = random_forest(housing_prepared, housing_labels,
                                              strat_test_set, imputer)

    random_forest_scoring(y_test, final_predictions)


if __name__ == "__main__":
    main()
