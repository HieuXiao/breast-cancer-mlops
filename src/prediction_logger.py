import json
from datetime import datetime
import os


LOG_FILE = "logs/predictions.jsonl"


def save_prediction_log(
    features,
    prediction,
    confidence,
    model_version
):

    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "features": features,
        "prediction": prediction,
        "confidence": confidence,
        "model_version": model_version
    }

    os.makedirs(
        "logs",
        exist_ok=True
    )

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(log_data)
            + "\n"
        )