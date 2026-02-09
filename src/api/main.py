# main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import re
import uuid
import time
import os
import json
import subprocess
from typing import Dict, Any, Optional

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent      # src/api
PROJECT_ROOT = BASE_DIR.parents[1]              # projet/

COLLECT_SCRIPT = os.getenv(
    "COLLECT_SCRIPT",
    str(PROJECT_ROOT / "src" / "scripts" / "acquire_data.py")
)
MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "final_model_xgboost.pkl"))
METRICS_PATH = os.getenv("METRICS_PATH", str(PROJECT_ROOT / "models" / "metrics.json"))
TRAIN_SCRIPT = os.getenv("TRAIN_SCRIPT", str(PROJECT_ROOT / "src" / "training" / "train_v2.py"))
PYTHON_BIN = os.getenv("PYTHON_BIN", "python3")

# MODEL_PATH = os.getenv("MODEL_PATH", "../../models/final_model_xgboost.pkl")
# METRICS_PATH = os.getenv("METRICS_PATH", "../../models/metrics.json")
# TRAIN_SCRIPT = os.getenv("TRAIN_SCRIPT", "training/train_model.py")
# PYTHON_BIN = os.getenv("PYTHON_BIN", "python")
# COLLECT_SCRIPT = os.getenv("COLLECT_SCRIPT", "collect_data.py")


app = FastAPI(title="GitHub Issue Time-to-Close Predictor")

# =========================
# LOAD MODEL
# =========================
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

LABELS = {0: "Flash (<35m)", 1: "Day (<24h)", 2: "Slow (>24h)"}


# In-memory job store
JOBS: Dict[str, Dict[str, Any]] = {}


# =========================
# INPUT SCHEMAS
# =========================
class IssueInput(BaseModel):
    title: str
    body: str = ""
    language: str
    stars: int
    forks: int
    num_comments: int
    num_labels: int
    contains_bug: int
    repo_age_days: int
    created_hour: int


class TrainRequest(BaseModel):
    data_path: str = Field("data/raw/github_issues_full_500k.csv")
    experiment: str = Field("GitHub_Issues_XGBoost")

    n_samples: int = Field(500000, ge=1000)
    random_state: int = Field(42, ge=0, le=9999)

    tfidf_max_features: int = Field(1500, ge=500, le=20000)
    tfidf_ngram_max: int = Field(2, ge=1, le=3)
    tfidf_stop_words: str = Field("english")  # "english" ou "none"

    n_estimators: int = Field(200, ge=50, le=3000)
    learning_rate: float = Field(0.08, ge=0.01, le=1.0)
    max_depth: int = Field(6, ge=2, le=20)
    subsample: float = Field(0.9, ge=0.1, le=1.0)
    colsample_bytree: float = Field(0.9, ge=0.1, le=1.0)
    tree_method: str = Field("hist")

    save_model: bool = True  # ici toujours True en pratique
    # si tu veux: log_to_mlflow bool etc (déjà géré par le script via mlflow.start_run)

class CollectRequest(BaseModel):
    n_rows: int = Field(50000, ge=1000, le=5_000_000)
    out_path: str = Field("data/raw/collect_latest.csv")
    mark_processed: bool = True


# =========================
# FEATURE ENGINEERING
# =========================
def build_features(issue: IssueInput) -> pd.DataFrame:
    full_text = f"{issue.title} {issue.body or ''}"
    text_len = len(full_text)

    is_crash = int(bool(re.search(r"crash|exception|error|fail|panic", full_text, re.IGNORECASE)))
    is_feature = int(bool(re.search(r"feature|add|request|support|implement", full_text, re.IGNORECASE)))

    return pd.DataFrame([{
        "full_text": full_text,
        "language": issue.language,
        "stars": issue.stars,
        "forks": issue.forks,
        "num_comments": issue.num_comments,
        "num_labels": issue.num_labels,
        "contains_bug": issue.contains_bug,
        "repo_age_days": issue.repo_age_days,
        "created_hour": issue.created_hour,
        "text_len": text_len,
        "is_crash": is_crash,
        "is_feature": is_feature
    }])


# =========================
# ROUTES
# =========================
@app.get("/")
def health_check():
    return {"status": "API is running 🚀"}


@app.get("/model/info")
def model_info():
    mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    metrics_exists = os.path.exists(METRICS_PATH)
    return {
        "model_path": MODEL_PATH,
        "model_last_modified_epoch": mtime,
        "metrics_path": METRICS_PATH,
        "metrics_file_exists": metrics_exists
    }


@app.post("/reload-model")
def reload_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model file not found")
    model = joblib.load(MODEL_PATH)
    return {"status": "reloaded", "model_path": MODEL_PATH}


@app.post("/predict")
def predict(issue: IssueInput):
    X = build_features(issue)
    pred_class = int(model.predict(X)[0])
    return {"prediction_class": pred_class, "prediction_label": LABELS[pred_class]}


def _run_training_job(job_id: str, req: TrainRequest):
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["started_at"] = time.time()

    try:
        if not os.path.exists(TRAIN_SCRIPT):
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = f"TRAIN_SCRIPT not found: {TRAIN_SCRIPT}"
            JOBS[job_id]["finished_at"] = time.time()
            return

        cmd = [
            PYTHON_BIN, TRAIN_SCRIPT,
            "--data-path", req.data_path,
            "--experiment", req.experiment,
            "--n-samples", str(req.n_samples),
            "--random-state", str(req.random_state),
            "--tfidf-max-features", str(req.tfidf_max_features),
            "--tfidf-ngram-max", str(req.tfidf_ngram_max),
            "--tfidf-stop-words", str(req.tfidf_stop_words),
            "--n-estimators", str(req.n_estimators),
            "--learning-rate", str(req.learning_rate),
            "--max-depth", str(req.max_depth),
            "--subsample", str(req.subsample),
            "--colsample-bytree", str(req.colsample_bytree),
            "--tree-method", str(req.tree_method),
            "--model-path", MODEL_PATH,
            "--metrics-path", METRICS_PATH
        ]

        p = subprocess.run(cmd, capture_output=True, text=True)

        JOBS[job_id]["stdout"] = (p.stdout or "")[-20000:]
        JOBS[job_id]["stderr"] = (p.stderr or "")[-20000:]

        # Capture run_id from stdout
        run_id = None
        for line in (p.stdout or "").splitlines():
            if line.startswith("MLFLOW_RUN_ID="):
                run_id = line.split("=", 1)[1].strip()
                break
        if run_id:
            JOBS[job_id]["mlflow_run_id"] = run_id

        if p.returncode != 0:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["finished_at"] = time.time()
            return

        # Load metrics.json
        if os.path.exists(METRICS_PATH):
            try:
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    JOBS[job_id]["metrics"] = json.load(f)
            except Exception as e:
                JOBS[job_id]["metrics"] = {"error": f"Could not read metrics.json: {e}"}

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["finished_at"] = time.time()

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["finished_at"] = time.time()


@app.post("/train")
def train(req: TrainRequest, background: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "queued",
        "created_at": time.time(),
        "request": req.model_dump(),
        "stdout": "",
        "stderr": "",
        "mlflow_run_id": None,
        "metrics": None,
    }
    background.add_task(_run_training_job, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@app.get("/train/status/{job_id}")
def train_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job_id not found")
    return JOBS[job_id]


@app.post("/collect")
def collect(req: CollectRequest):
    if not os.path.exists(COLLECT_SCRIPT):
        raise HTTPException(status_code=404, detail=f"COLLECT_SCRIPT not found: {COLLECT_SCRIPT}")

    # out_path relatif -> on l’ancre dans BASE_DIR
    out_path = (BASE_DIR / req.out_path).resolve()

    # Autoriser seulement data/raw
    allowed_dir = (BASE_DIR / "data" / "raw").resolve()
    if allowed_dir not in out_path.parents:
        raise HTTPException(status_code=400, detail="out_path must be inside data/raw/")

    cmd = [
        PYTHON_BIN, COLLECT_SCRIPT,
        "--n-rows", str(req.n_rows),
        "--out-path", str(out_path),
        "--mark-processed" if req.mark_processed else "--no-mark-processed",
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        raise HTTPException(status_code=500, detail={
            "error": "collect failed",
            "stdout": (p.stdout or "")[-5000:],
            "stderr": (p.stderr or "")[-5000:],
        })

    return {"status": "done", "out_path": str(out_path), "stdout": (p.stdout or "")[-5000:]}