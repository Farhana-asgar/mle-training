
# House Value Prediction

This repository contains the **House Value Prediction** project, which includes a machine learning model to predict housing values based on various features. The code leverages a custom package, `house_value_prediction`, with scripts for data ingestion, model training, and scoring.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/house-value-prediction.git
   cd house-value-prediction
   ```

2. **Install the package**:
   Navigate to the `src/` directory, build the package, and install it.

   ```bash
   python -m build
   ```

## Running the Project

The main script to execute the end-to-end flow is located in the `scripts/` directory.

1. **Navigate to the `scripts` directory**:
   ```bash
   cd scripts/
   ```

2. **Run the main script**:
   The `main_script.py` calls all necessary functions from the `house_value_prediction` package for data ingestion, training, and prediction. Run it with:

   ```bash
   python main_script.py
   ```

## Key Components

### `house_value_prediction` Package (`src/house_value_prediction`)

- **`ingest`**: Module for loading and preprocessing housing data.
- **`train`**: Module to train a machine learning model based on the processed data.
- **`score`**: Module for scoring and evaluating model predictions.

### Scripts (`scripts/`)

- **`main_script.py`**: The main execution script, calling functions from the `house_value_prediction` package to ingest data, train the model, and score the predictions.
- **Outputs**: Results and logs are saved in the `scripts/scripts_output` folder.

### Tests (`tests/`)

- **Unit and integration tests** for each module in the `house_value_prediction` package.
- Test data and logs are stored in `tests/test_data` and `tests/script_logs`.

## Testing the Installation

After installing the package, you can verify the installation by importing the package in Python:

```python
import house_value_prediction
print("House Value Prediction package installed successfully!")
```

Alternatively, run the main script to verify everything is set up correctly:

```bash
python scripts/main_script.py
```

## Configuration

Logs of the script will be available in script_log folder.  You can opt whether or not to view it on the screen while running the script by specifying the option when calling the script


Certain configurations, such as dataset paths or model parameters, can be adjusted within the code files under `src/house_value_prediction/`. You can customize the following:

- **Disable Log Display in screen**:
```bash
cd scripts
python main_script.py --no-console-log

```

## GitHub Actions

The `.github/workflows` folder contains configuration for continuous integration (CI) with GitHub Actions. Automated workflows are set up to run tests and ensure code quality with every push.


## License

This project is licensed under the MIT License.
