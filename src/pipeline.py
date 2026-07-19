# breast-cancer-mlops/src/pipeline.py

# -- Machine Learning Pipeline Definition --
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression

def build_pipeline(c=1.0):
    """
    Build a Machine Learning Pipeline.

    Parameters
    ----------
    c : float
        Inverse regularization strength of Logistic Regression.

    Returns
    -------
    Pipeline
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    random_state=42
                )
            ),
        ]
    )