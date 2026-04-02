"""
AutoML FastAPI Backend
Production-quality API for automated machine learning.
"""

from __future__ import annotations

import os
import uuid
import logging
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from core.validator import detect_problem_type
from core.preprocessing import build_preprocessing_pipeline
from core.trainer import train_and_evaluate

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AutoML Web App",
    description="Automated Machine Learning API — upload a CSV, get trained models.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Storage directories
# ---------------------------------------------------------------------------
DATASET_DIR = "storage/datasets"
MODEL_DIR = "storage/models"
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================= HELPERS =====================================

def _read_csv(file: Any) -> pd.DataFrame:
    """Read and validate an uploaded CSV file."""
    try:
        return pd.read_csv(file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {exc}")


def _validate_target(df: pd.DataFrame, target: str) -> None:
    """Ensure the target column exists in the DataFrame."""
    if target not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target}' not found. Available columns: {df.columns.tolist()}",
        )


def _build_dataset_stats(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Return descriptive statistics for the dataset."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(v) for col, v in df.isnull().sum().items()},
        "total_missing": int(df.isnull().sum().sum()),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "target_column": target,
        "describe": df.describe(include="all").fillna("").to_dict(),
    }


# ============================= ENDPOINTS ===================================

@app.get("/")
def root() -> Dict[str, str]:
    """Health/root endpoint."""
    return {"status": "ok", "message": "AutoML API v2.0 is running"}


@app.get("/health")
def health() -> Dict[str, str]:
    """Lightweight health check."""
    return {"status": "healthy"}


# ----------------------------- Preview ------------------------------------

@app.post("/preview")
async def preview_dataset(
    file: UploadFile = File(...),
    target: str = Form(...),
) -> Dict[str, Any]:
    """Upload a CSV and return a preview + statistics (no training)."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    df = _read_csv(file.file)
    _validate_target(df, target)

    problem_type = detect_problem_type(df, target)
    stats = _build_dataset_stats(df, target)

    return {
        "message": "Preview generated successfully.",
        "problem_type": problem_type,
        "stats": stats,
        "head": df.head(10).to_dict(orient="records"),
    }


# ----------------------------- Upload & Train -----------------------------

@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target: str = Form(...),
) -> Dict[str, Any]:
    """
    Upload a CSV, run the full AutoML pipeline, and return results.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    df = _read_csv(file.file)
    _validate_target(df, target)

    dataset_id = str(uuid.uuid4())

    # Persist raw dataset
    dataset_path = os.path.join(DATASET_DIR, f"{dataset_id}.csv")
    df.to_csv(dataset_path, index=False)

    # Detect problem type
    problem_type = detect_problem_type(df, target)
    logger.info("Dataset %s | %s | rows=%d cols=%d", dataset_id, problem_type, *df.shape)

    # Preprocess
    X_train, X_test, y_train, y_test, preprocessor, feature_names = (
        build_preprocessing_pipeline(df, target)
    )

    # Train & evaluate all models
    results, best_model_name, best_pipeline, feature_importance = train_and_evaluate(
        problem_type, X_train, X_test, y_train, y_test, preprocessor,
    )

    # Save best pipeline
    model_path = os.path.join(MODEL_DIR, f"{dataset_id}_best_model.pkl")
    if best_pipeline is not None:
        joblib.dump(best_pipeline, model_path)

    # Build response
    stats = _build_dataset_stats(df, target)

    return {
        "message": "Training completed successfully.",
        "dataset_id": dataset_id,
        "problem_type": problem_type,
        "stats": stats,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "model_results": results,
        "best_model": best_model_name,
        "feature_names": feature_names,
        "feature_importance": feature_importance,
    }


# ----------------------------- Download -----------------------------------

@app.get("/download/{dataset_id}")
def download_model(dataset_id: str) -> FileResponse:
    """Download the best trained model as a .pkl file."""
    model_path = os.path.join(MODEL_DIR, f"{dataset_id}_best_model.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found for this dataset.")

    return FileResponse(
        path=model_path,
        filename=f"{dataset_id}_best_model.pkl",
        media_type="application/octet-stream",
    )