#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_model.py
- Charge dataset CSV (github_issues_full_500k.csv)
- Feature engineering (full_text, text_len, is_crash, is_feature)
- Split 60/20/20
- Pipeline: TF-IDF + OneHotEncoder + Numeric (imputer+scaler) + XGBoost
- Logs MLflow: params, metrics, model
- Sauvegarde:
    - models/final_model_xgboost.pkl
    - models/metrics.json
- Print MLFLOW_RUN_ID=... (pour que l'API le capture)
"""

import argparse
import json
import os
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


# =========================
# TARGET
# =========================
def categorize_time_balanced(hours: float) -> int:
    # mêmes seuils que ton script
    if hours < 0.56:
        return 0      # Flash
    elif hours < 23.88:
        return 1      # Day
    else:
        return 2      # Slow


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data-path", type=str, default="data/raw/github_issues_full_500k.csv")
    p.add_argument("--experiment", type=str, default="GitHub_Issues_XGBoost")

    p.add_argument("--n-samples", type=int, default=500000, help="Nombre de lignes max à utiliser")
    p.add_argument("--random-state", type=int, default=42)

    # Split
    p.add_argument("--test-size-temp", type=float, default=0.4, help="Part temp (val+test) ex: 0.4 => 60/40")
    p.add_argument("--val-share-of-temp", type=float, default=0.5, help="Part du temp pour val (0.5 => val=test)")

    # TFIDF
    p.add_argument("--tfidf-max-features", type=int, default=1500)
    p.add_argument("--tfidf-ngram-max", type=int, default=2)
    p.add_argument("--tfidf-stop-words", type=str, default="english")  # "english" ou "none"

    # XGB
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=0.08)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9)
    p.add_argument("--tree-method", type=str, default="hist")

    # Outputs
    p.add_argument("--model-path", type=str, default="models/final_model_xgboost.pkl")
    p.add_argument("--metrics-path", type=str, default="models/metrics.json")

    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset introuvable: {args.data_path}")

    df = pd.read_csv(args.data_path)
    if args.n_samples and len(df) > args.n_samples:
        df = df.head(args.n_samples)

    # Sécurité: s'assurer des colonnes
    required_cols = [
        "title_text", "body_text", "language",
        "stars", "forks", "num_comments", "num_labels",
        "contains_bug", "repo_age_days", "created_hour",
        "time_to_close"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

    # Target
    df["target"] = df["time_to_close"].apply(categorize_time_balanced)

    # Features
    df["full_text"] = df["title_text"].fillna("") + " " + df["body_text"].fillna("")
    df["text_len"] = df["full_text"].str.len()

    df["is_crash"] = df["full_text"].str.contains(
        "crash|exception|error|fail|panic", case=False, regex=True
    ).astype(int)

    df["is_feature"] = df["full_text"].str.contains(
        "feature|add|request|support|implement", case=False, regex=True
    ).astype(int)

    X = df[[
        "full_text", "language",
        "stars", "forks", "num_comments", "num_labels",
        "contains_bug", "repo_age_days", "created_hour",
        "text_len", "is_crash", "is_feature"
    ]]

    y = df["target"]

    # Split 60/20/20 par défaut (via temp=0.4, val_share=0.5)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=args.test_size_temp,
        stratify=y,
        random_state=args.random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=args.val_share_of_temp,
        stratify=y_temp,
        random_state=args.random_state
    )

    stop_words = None if args.tfidf_stop_words.lower() == "none" else args.tfidf_stop_words
    ngram_range = (1, int(args.tfidf_ngram_max))

    preprocessor = ColumnTransformer(
        transformers=[
            ("txt", TfidfVectorizer(
                stop_words=stop_words,
                max_features=int(args.tfidf_max_features),
                ngram_range=ngram_range
            ), "full_text"),

            ("cat", OneHotEncoder(handle_unknown="ignore"), ["language"]),

            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), [
                "stars", "forks", "num_comments", "num_labels",
                "contains_bug", "repo_age_days", "created_hour",
                "text_len", "is_crash", "is_feature"
            ])
        ]
    )

    clf = XGBClassifier(
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        objective="multi:softmax",
        num_class=3,
        n_jobs=-1,
        random_state=int(args.random_state),
        eval_metric="mlogloss",
        tree_method=args.tree_method
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLFLOW_RUN_ID={run_id}")  # important pour l’API

        # Log params
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("tfidf_max_features", args.tfidf_max_features)
        mlflow.log_param("tfidf_ngram_range", str(ngram_range))
        mlflow.log_param("tfidf_stop_words", str(stop_words))
        mlflow.log_param("xgb_n_estimators", args.n_estimators)
        mlflow.log_param("xgb_learning_rate", args.learning_rate)
        mlflow.log_param("xgb_max_depth", args.max_depth)
        mlflow.log_param("xgb_subsample", args.subsample)
        mlflow.log_param("xgb_colsample_bytree", args.colsample_bytree)
        mlflow.log_param("xgb_tree_method", args.tree_method)

        # Train
        model.fit(X_train, y_train)

        # Metrics
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        mlflow.log_metric("train_accuracy", float(train_acc))
        mlflow.log_metric("val_accuracy", float(val_acc))
        mlflow.log_metric("test_accuracy", float(test_acc))

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Save model for API
        joblib.dump(model, args.model_path)

        # Save metrics.json for UI
        cm = confusion_matrix(y_test, y_test_pred)
        report = classification_report(
            y_test, y_test_pred,
            target_names=["Flash (<35m)", "Day (<24h)", "Slow (>24h)"],
            output_dict=True
        )

        metrics_payload = {
            "run_id": run_id,
            "timestamp": int(time.time()),
            "data_path": args.data_path,
            "n_used": int(len(df)),
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "test_accuracy": float(test_acc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "model_path": args.model_path
        }

        with open(args.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved model => {args.model_path}")
    print(f"[OK] Saved metrics => {args.metrics_path}")


if __name__ == "__main__":
    main()