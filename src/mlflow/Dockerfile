FROM python:3.10-slim

WORKDIR /mlflow

# deps utiles (mysqlclient/psycopg2 peuvent nécessiter build tools — ici on reste minimal)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir mlflow==2.14.3

# dossier par défaut pour artifacts
RUN mkdir -p /mlflow/artifacts

EXPOSE 5030

# Par défaut:
# - backend-store: sqlite (fichier) => /mlflow/mlflow.db
# - artifacts: /mlflow/artifacts
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5030", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--default-artifact-root", "/mlflow/artifacts"]