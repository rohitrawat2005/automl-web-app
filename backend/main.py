from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pandas as pd
import uuid
import os

app = FastAPI(title="AutoML Web App")

DATASET_DIR = "storage/datasets"
os.makedirs(DATASET_DIR, exist_ok=True)


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

    dataset_id = str(uuid.uuid4())
    file_path = os.path.join(DATASET_DIR, f"{dataset_id}.csv")
    df.to_csv(file_path, index=False)

    return {
        "message": "Dataset uploaded successfully",
        "dataset_id": dataset_id,
        "columns": df.columns.tolist(),
        "target": target,
        "rows": df.shape[0]
    }
