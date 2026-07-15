# Breast Cancer MLOps

End-to-end MLOps project for Breast Cancer classification using the Breast Cancer Wisconsin Dataset from scikit-learn.

## Tech Stack

- Python
- Scikit-learn
- MLflow
- FastAPI
- Pytest
- GitHub Actions

---

## Project Structure

```text
breast-cancer-mlops/
├── app/
├── config/
├── data/
├── logs/
├── models/
├── src/
├── tests/
└── .github/workflows/
```

---

## Getting Started

```bash
# Create virtual environment
python -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Run training

```bash
python src/train.py
```

Launch API

```bash
uvicorn app.main:app --reload
```

Launch MLflow

```bash
mlflow ui
```

---

## Git Workflow

```text
main
│
└── develop
    ├── feature/project-setup
    ├── feature/ml-pipeline
    ├── feature/mlflow-tracking
    ├── feature/model-serialization
    ├── feature/fastapi-api
    ├── feature/model-monitoring
    ├── feature/testing-ci
    └── feature/documentation
```

### Branch Overview

| Branch | Purpose |
|--------|---------|
| `main` | Stable release |
| `develop` | Integration branch |
| `feature/project-setup` | Project initialization |
| `feature/ml-pipeline` | ML pipeline & training |
| `feature/mlflow-tracking` | Experiment tracking |
| `feature/model-serialization` | Model saving/loading |
| `feature/fastapi-api` | REST API |
| `feature/model-monitoring` | Drift detection & logging |
| `feature/testing-ci` | Testing & GitHub Actions |
| `feature/documentation` | Documentation |

---

## License

Educational project for learning MLOps.
