FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/final_model_xgboost.pkl \
    METRICS_PATH=/app/models/metrics.json \
    TRAIN_SCRIPT=/app/training/train_v2.py \
    COLLECT_SCRIPT=/app/src/scripts/acquire_data.py

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --default-timeout=100 --retries 10 -r requirements.txt

# API
COPY src/api/ ./src/api/

# scripts (collect)
COPY src/scripts/ ./src/scripts/

# training
COPY src/training/ ./src/training/
# models (pkl + metrics)
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]