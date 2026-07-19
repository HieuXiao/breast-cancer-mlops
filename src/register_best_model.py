import mlflow


mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)


RUN_ID = "9f78bb5533694dc8a8043e8ed0c3cb51"
MODEL_NAME = "BreastCancerClassifier"

model_uri = f"runs:/{RUN_ID}/model"
result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print("Model registered successfully")
print("Name:", result.name)
print("Version:", result.version)