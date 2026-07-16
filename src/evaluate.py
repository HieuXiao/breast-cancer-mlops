# breast-cancer-mlops\src\evaluate.py
# -- Evaluate The ML Pipeline Using Cross Validation --

from sklearn.model_selection import cross_val_score

from data import load_data
from pipeline import build_pipeline

def evaluate():
    # Evaluate the model using 5-fold Cross Validation
    x_train, x_test, y_train, y_test = load_data()
    
    pipeline = build_pipeline()

    accuracy_scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    roc_auc_scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=5,
        scoring="roc_auc"
    )

    print("Accuracy")
    print(accuracy_scores)
    print(f"Mean: {accuracy_scores.mean():.4f}")
    print(f"Std : {accuracy_scores.std():.4f}")

    print()

    print("ROC-AUC")
    print(roc_auc_scores)
    print(f"Mean: {roc_auc_scores.mean():.4f}")
    print(f"Std : {roc_auc_scores.std():.4f}")


if __name__ == "__main__":
    evaluate()