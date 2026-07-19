# Module 2 — MLflow Experiment Tracking


## Experiment

Name:
Breast Cancer Classification


## Experiment Runs

Models tested:

- LogisticRegression C=0.1
- LogisticRegression C=1.0
- LogisticRegression C=10.0


## Best Run

Model:

LogisticRegression_C_10.0


Run ID:

9f78bb5533694dc8a8043e8ed0c3cb51


Metrics:

ROC-AUC:
0.9954


Accuracy:
0.9825


F1-score:
0.9861


## Model Registry


Registered Model:

BreastCancerClassifier


Version:

1


Alias:

champion


Model URI:

models:/BreastCancerClassifier@champion


## Reproducibility

MLflow stores:

- Experiment
- Run ID
- Parameters
- Metrics
- Model artifact
- Model version

This allows tracking, rollback and reproducible deployment.