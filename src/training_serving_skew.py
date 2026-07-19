# Demonstrate training-serving skew

import joblib
from data import load_data
from pipeline import build_pipeline

BAD_MODEL_PATH = (
    "models/logistic_regression_only_v1.joblib"
)

def create_bad_model():
    X_train, X_test, y_train, y_test = load_data()

    # Train correct pipeline
    pipeline = build_pipeline()
    pipeline.fit(
        X_train,
        y_train
    )

    # Remove scaler intentionally
    estimator = pipeline.named_steps[
        "classifier"
    ]

    joblib.dump(
        estimator,
        BAD_MODEL_PATH
    )

    print(
        "Bad model saved"
    )

def compare_prediction():
    X_train, X_test, y_train, y_test = load_data()

    # Correct model
    correct_model = joblib.load(
        "models/breast_cancer_pipeline_v1.joblib"
    )
    correct_prediction = (
        correct_model.predict(X_test)
    )

    # Incorrect model
    bad_model = joblib.load(
        BAD_MODEL_PATH
    )
    bad_prediction = (
        bad_model.predict(X_test)
    )

    difference = (
        correct_prediction != bad_prediction
    ).sum()

    print(
        "Prediction difference:",
        difference
    )

if __name__ == "__main__":
    create_bad_model()
    compare_prediction()