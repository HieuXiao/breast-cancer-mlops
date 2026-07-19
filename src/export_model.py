# src/export_model.py
# Serialize complete ML pipeline using joblib

import joblib
import os

from data import load_data
from pipeline import build_pipeline

MODEL_PATH = "models/breast_cancer_pipeline_v1.joblib"

def export_model():
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    # Build pipeline
    pipeline = build_pipeline()
    # Train pipeline
    pipeline.fit(
        X_train,
        y_train
    )
    # Create models directory
    os.makedirs(
        "models",
        exist_ok=True
    )
    # Save entire pipeline
    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print(
        f"Model saved: {MODEL_PATH}"
    )

if __name__ == "__main__":
    export_model()