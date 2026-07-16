# breast-cancer-mlops\src\train.py
# -- Train The ML Pipeline & Evaluate Its Performance --

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data import load_data
from pipeline import build_pipeline

def train():
    # Train the pipeline and evaluate the model

    # Load dataset
    x_train, x_test, y_train, y_test = load_data()
    # Build pipeline
    pipeline = build_pipeline()

    # Train model
    pipeline.fit(x_train, y_train)

    # Predict class labels
    y_pred = pipeline.predict(x_test)
    # Predict class probabilities
    y_prob = pipeline.predict_proba(x_test)[:, 1]

    # Evaluate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

if __name__ == "__main__":
    train()