import joblib
import numpy as np

from data import load_data
from pipeline import build_pipeline

MODEL_PATH = (
    "models/breast_cancer_pipeline_v1.joblib"
)

def test_serialization():
    X_train, X_test, y_train, y_test = load_data()

    # Original pipeline
    model_original = build_pipeline()
    model_original.fit(
        X_train,
        y_train
    )
    pred_original = (
        model_original.predict(X_test)
    )

    # Load saved pipeline
    model_loaded = joblib.load(
        MODEL_PATH
    )
    pred_loaded = (
        model_loaded.predict(X_test)
    )

    # Compare prediction
    assert np.array_equal(
        pred_original,
        pred_loaded
    )
    print(
        "Serialization test passed"
    )

if __name__ == "__main__":
    test_serialization()