# app/main.py
from fastapi import FastAPI
from app.schemas import HeartInput
from app.utils import predict_heart, FEATURES

app = FastAPI(title="Heart Disease Prediction API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model": "RandomForestClassifier",
        "features": FEATURES
    }

@app.post("/predict")
def predict(input: HeartInput):
    result = predict_heart(input.dict())
    return {"heart_disease": result}
