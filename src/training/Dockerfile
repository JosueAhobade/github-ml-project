FROM python:3.10-slim

WORKDIR /app

# Dépendances système (XGBoost)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# COPY ../models/ ../models/

CMD ["python", "src/training/train_xgboost.py"]