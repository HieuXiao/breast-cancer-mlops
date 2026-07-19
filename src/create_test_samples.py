from sklearn.datasets import load_breast_cancer
import json

dataset = load_breast_cancer()
samples = [
    0,
    10,
    100
]

for i in samples:
    data = {
        "features": dataset.data[i].tolist(),
        "true_label": int(dataset.target[i])
    }
    print(json.dumps(data, indent=2))