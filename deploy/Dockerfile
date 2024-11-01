# Dockerfile
# Use a base image with Python
FROM python:3.11.3-slim

# Set the working directory in the container
WORKDIR /app


# Copy the application code to the container
COPY . .
RUN pip install -r requirements.txt
# Install dependencies
RUN pip install --upgrade pip && pip install house_value_prediction-0.0.1-py3-none-any.whl

# Expose the ports for both the application and MLflow (assuming 5000 for MLflow and 8000 for the application)
EXPOSE 5000

# Start both services: the application and MLflow UI
CMD ["./start.sh"]
