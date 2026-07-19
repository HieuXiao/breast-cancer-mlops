# breast-cancer-mlops/src/mlflow_train.py
# Train Logistic Regression with MLflow Tracking.

import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from data import load_data
from pipeline import build_pipeline


def train_with_mlflow(c):
    # Connect to MLflow Tracking Server
    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )
    # Create / select experiment
    mlflow.set_experiment(
        "Breast Cancer Classification"
    )
    # Enable sklearn autologging
    mlflow.autolog(
        log_models=False
    )
    # Create MLflow Run
    with mlflow.start_run(
        run_name=f"LogisticRegression_C_{c}"
    ):

        # =========================
        # 1. Load data
        # =========================
        X_train, X_test, y_train, y_test = load_data()

        # =========================
        # 2. Log parameters
        # =========================
        mlflow.log_param(
            "model",
            "LogisticRegression"
        )
        mlflow.log_param(
            "C",
            c
        )

        # =========================
        # 3. Build Pipeline
        # =========================
        pipeline = build_pipeline(
            c=c
        )

        # =========================
        # 4. Train
        # =========================
        pipeline.fit(
            X_train,
            y_train
        )

        # =========================
        # 5. Prediction
        # =========================
        y_pred = pipeline.predict(
            X_test
        )
        y_prob = pipeline.predict_proba(
            X_test
        )[:, 1]

        # =========================
        # 6. Metrics
        # =========================
        accuracy = accuracy_score(
            y_test,
            y_pred
        )
        f1 = f1_score(
            y_test,
            y_pred
        )
        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )
        # Log metrics to MLflow
        mlflow.log_metric(
            "accuracy",
            accuracy
        )
        mlflow.log_metric(
            "f1_score",
            f1
        )
        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        # =========================
        # 7. Log model artifact
        # =========================
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model"
        )

        # Print result
        print(
            f"C={c}"
        )
        print(
            f"Accuracy : {accuracy:.4f}"
        )
        print(
            f"F1-score : {f1:.4f}"
        )
        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

if __name__ == "__main__":

    configs = [
        0.1,
        1.0,
        10.0
    ]

    for c in configs:
        print("=" * 50)
        print(
            f"Training LogisticRegression(C={c})"
        )
        print("=" * 50)
        train_with_mlflow(c)