# breast-cancer-mlops/src/pipeline.py

# -- Machine Learning Pipeline Definition --
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression

def build_pipeline():
    # Create the ML pipeline
    pl = Pipeline(
        steps = [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )

    return pl