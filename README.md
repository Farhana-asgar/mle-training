
# House Value Prediction

This repository contains the **House Value Prediction** project, which includes a machine learning model to predict housing values based on various features. The code leverages a custom package, `house_value_prediction`, with scripts for data ingestion, model training, and scoring.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/house-value-prediction.git
   cd house-value-prediction
   ```

2. **Create Conda Environment**:
   Run the below command to create a conda environment

   ```bash
   conda env create -f env.yaml
   conda activate house_value_prediction_env
   ```

3. **Install the package**:
   Run the below command to create a conda environment

   ```bash
   pip install house_value_prediction-0.0.1-py3-none-any.whl
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

3. **Configuration(Optional)**:

Logs of the script will be available in scripts/script_logs folder.  You can opt whether or not to view it on the screen while running.


- **Disable Log Display in screen**:

```bash
cd scripts
python main_script.py --no-console-log

```

## Key Components

### `house_value_prediction` Package (`src/house_value_prediction`)

- **`ingest`**: Module for loading and preprocessing housing data.
- **`train`**: Module to train a machine learning model based on the processed data.
- **`score`**: Module for scoring and evaluating model predictions.

### Scripts (`scripts/`)

- **`main_script.py`**: The main execution script, calling functions from the `house_value_prediction` package to ingest data, train the model, and score the predictions.
- **Outputs**: Results and logs are saved in the `scripts/scripts_output` folder.


## Testing

### **Unit Test**
- For each module in the `house_value_prediction` package.
- Test data and logs are stored in `tests/test_data` and `tests/script_logs`.

### **Installation Test**
After installing the package, you can verify the installation by importing the package in Python:

```bash
cd tests
python installation_test.py
```

## GitHub Actions

The `.github/workflows` folder contains configuration for continuous integration (CI) with GitHub Actions. Automated workflows are set up to run tests and ensure code quality with every push.


## License

This project is licensed under the MIT License.
