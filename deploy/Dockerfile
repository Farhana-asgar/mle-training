# Stage 1: Build and Install Dependencies
FROM python:3.11.3-slim AS builder

# Set the working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt ./
COPY house_value_prediction-0.0.1-py3-none-any.whl ./

# Install dependencies, including MLflow and project package
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir mlflow house_value_prediction-0.0.1-py3-none-any.whl

# Copy the rest of the application files
COPY . .

# Stage 2: Runtime Environment
FROM python:3.11.3-slim

# Create a non-root user
RUN useradd -m myuser

# Set the working directory
WORKDIR /app

# Copy only necessary artifacts from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/mlflow /usr/local/bin/mlflow
COPY --from=builder /app /app

# Check mlflow installation and setup path
RUN mlflow --version || echo "MLflow not installed or accessible"
RUN which mlflow || echo "MLflow binary not found"
RUN echo "PATH: $PATH"

# Ensure permissions for the working directory and start script
RUN chown -R myuser:myuser /app && chmod +x ./start.sh

# Switch to non-root user
USER myuser

# Expose the port for MLflow UI
EXPOSE 5000

# Run the start script

CMD ["./start.sh"]