# breast-cancer-mlops/app/main.py
# FastAPI inference service

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

MODEL_PATH = (
    "models/breast_cancer_pipeline_v1.joblib"
)

# Load model once when API starts
model = joblib.load(
    MODEL_PATH
)
app = FastAPI(
    title="Breast Cancer Prediction API",
    version="1.0.0"
)

# ==========================
# Input schema
# ==========================
class BreastCancerInput(BaseModel):

    features: list[float]

# ==========================
# Health check
# ==========================
@app.get("/health")
def health():

    return {
        "status": "ok"
    }

# ==========================
# Model information
# ==========================
@app.get("/model-info")
def model_info():
    return {
        "model": "breast_cancer_pipeline_v1",

        "algorithm": 
        "StandardScaler + LogisticRegression",

        "version": "1.0"
    }

# ==========================
# Prediction endpoint
# ==========================
@app.post("/predict")
def predict(
    data: BreastCancerInput
):
    # Convert input
    X = np.array(
        data.features
    ).reshape(
        1,
        -1
    )

    # Prediction
    prediction = model.predict(
        X
    )[0]

    probability = model.predict_proba(
        X
    )[0]

    confidence = float(
        max(probability)
    )

    label = (
        "malignant"
        if prediction == 0
        else
        "benign"
    )

    return {
        "prediction": int(prediction),
        "label": label,
        "confidence": confidence
    }