import mlflow


mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)


MODEL_URI = (
    "models:/BreastCancerClassifier@champion"
)


model = mlflow.sklearn.load_model(
    MODEL_URI
)


print(
    "Model loaded successfully"
)

print(model)