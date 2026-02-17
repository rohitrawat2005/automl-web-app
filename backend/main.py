from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import uuid
import os
import joblib

from core.validator import detect_problem_type
from core.preprocessing import build_preprocessing_pipeline
from core.trainer import train_and_evaluate

app = FastAPI(title="AutoML Web App")

DATASET_DIR = "storage/datasets"
MODEL_DIR = "storage/models"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "AutoML API is running"}


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target: str = Form(...)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    if target not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target}' not found in dataset"
        )

    # Save dataset
    dataset_id = str(uuid.uuid4())
    file_path = os.path.join(DATASET_DIR, f"{dataset_id}.csv")
    df.to_csv(file_path, index=False)

    # Detect problem type
    problem_type = detect_problem_type(df, target)

    # Preprocess
    X_train, X_test, y_train, y_test, preprocessor = build_preprocessing_pipeline(df, target)

    # Train models
    results, best_model_name, best_pipeline = train_and_evaluate(
        problem_type,
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    # Save best model
    model_path = os.path.join(MODEL_DIR, f"{dataset_id}_best_model.pkl")
    joblib.dump(best_pipeline, model_path)

    # ✅ RETURN MUST BE INSIDE UPLOAD FUNCTION
    return {
        "message": "Dataset uploaded successfully",
        "dataset_id": dataset_id,
        "columns": df.columns.tolist(),
        "target": target,
        "rows": df.shape[0],
        "problem_type": problem_type,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "model_results": results,
        "best_model": best_model_name,
        "model_path": model_path
    }


@app.get("/download/{dataset_id}")
def download_model(dataset_id: str):
    model_path = os.path.join(MODEL_DIR, f"{dataset_id}_best_model.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    return FileResponse(
        path=model_path,
        filename=f"{dataset_id}_best_model.pkl",
        media_type="application/octet-stream"
    )
