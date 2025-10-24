# app/utils.py
import joblib
import numpy as np
import os

model_path = os.path.join("model", "heart_model.joblib")
model = joblib.load(model_path)

FEATURES = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

def predict_heart(data: dict) -> bool:
    X = np.array([data[feat] for feat in FEATURES]).reshape(1, -1)
    pred = model.predict(X)
    return bool(pred[0])
