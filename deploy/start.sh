#!/bin/sh

# Start the application in the background
mlflow ui --host 0.0.0.0 --port 5000 &

python scripts/main_script.py

# Start MLflow UI
