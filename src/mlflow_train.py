# breast-cancer-mlops\src\mlflow_train.py
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


def train_with_mlflow():

    # Enable automatic logging
    mlflow.autolog()

    # Start an MLflow run
    with mlflow.start_run():

        # Load dataset
        X_train, X_test, y_train, y_test = load_data()

        # Build pipeline
        pipeline = build_pipeline()

        # Train model
        pipeline.fit(X_train, y_train)

        # Prediction
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")


if __name__ == "__main__":
    train_with_mlflow()