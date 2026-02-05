from fastapi import FastAPI

app = FastAPI(title="AutoML Web App")

@app.get("/")
def root():
    return {"message": "AutoML API is running"}
