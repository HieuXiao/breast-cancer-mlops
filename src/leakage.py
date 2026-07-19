from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline import build_pipeline


def correct_workflow():
    # Evaluate the correct ML workflow using Pipeline.
    # The scaler is fitted only on training data.
    dataset = load_breast_cancer()
    x = dataset.data
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_prob = pipeline.predict_proba(x_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    return roc_auc


def leakage_workflow():
    # Evaluate incorrect workflow containing data leakage.
    # The scaler uses information from the entire datase before train-test splitting.
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target

    # WRONG: Scaler learns statistics from both train and test data.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    return roc_auc


def leakage_experiment():
    # Compare correct workflow and leakage workflow.

    correct_auc = correct_workflow()
    leakage_auc = leakage_workflow()

    difference = leakage_auc - correct_auc

    print(f"Correct ROC-AUC : {correct_auc:.4f}")
    print(f"Leakage ROC-AUC : {leakage_auc:.4f}")
    print(f"Difference      : {difference:.4f}")

if __name__ == "__main__":
    leakage_experiment()