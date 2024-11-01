#Stage 1: Build
FROM python:3.11.3-slim AS builder

RUN useradd -m myuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --upgrade pip && pip install house_value_prediction-0.0.1-py3-none-any.whl

# Stage 2: Run the application
FROM python:3.11.3-slim

RUN useradd -m myuser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

RUN chown -R myuser:myuser /app
USER myuser

EXPOSE 5000

RUN chmod +x /app/start.sh
CMD ["./start.sh"]
