# Module 3 — REST API Deployment

## 1. Model Serialization

### Objective

Serialize the complete machine learning pipeline using joblib.

The saved model must include:

```
StandardScaler
+
LogisticRegression
```

to guarantee consistency between training and inference.


---

## Exported Model

File:

```
models/breast_cancer_pipeline_v1.joblib
```

Export command:

```bash
python src/export_model.py
```


Output:

```
Model saved:
models/breast_cancer_pipeline_v1.joblib
```


---

## Serialization Validation

Test:

```bash
python src/test_serialization.py
```


Result:

```
Serialization test passed
```


The loaded model generated identical predictions compared with the original pipeline.


---

# 2. FastAPI Deployment


## API Framework

```
FastAPI
```

Server:

```
Uvicorn
```


Run command:

```bash
uvicorn app.main:app --reload
```


---

# API Endpoints


## GET /health

Purpose:

Check API availability.


Example response:

```json
{
  "status": "ok"
}
```


---

## GET /model-info

Purpose:

Return deployed model metadata.


Example response:

```json
{
  "model": "breast_cancer_pipeline_v1",
  "algorithm": "StandardScaler + LogisticRegression",
  "version": "1.0"
}
```


---

## POST /predict

Purpose:

Receive 30 breast cancer features and return:

- predicted class
- label
- confidence score


Input:

```json
{
 "features":[30 numerical values]
}
```


Output:

```json
{
 "prediction":0,
 "label":"malignant",
 "confidence":0.99
}
```


---

# 3. API Testing


Testing dataset:

```
sklearn.datasets.load_breast_cancer
```


Number of test samples:

```
3
```


Result:

| Sample | True Label | Prediction | Status |
|-|-|-|-|
| 1 | malignant | malignant | Passed |
| 2 | malignant | malignant | Passed |
| 3 | malignant | malignant | Passed |


Full curl logs:

See:

```
docs/module3_curl_log.md
```


---

# 4. Deployment Architecture


```
Client

   |
   |
POST /predict

   |

FastAPI

   |

joblib Pipeline

   |

StandardScaler

   |

LogisticRegression

   |

Prediction + Confidence
```


---

# 5. MLOps Considerations


## Training-serving consistency

The complete pipeline is serialized instead of only the estimator.

Reason:

The preprocessing step during training must be identical during inference.


Pipeline:

```
Raw features

    ↓

StandardScaler

    ↓

LogisticRegression

    ↓

Prediction
```


This prevents:

```
Training-serving skew
```


---

# 6. Module 3 Completion Status


| Requirement | Status |
|-|-|
| Serialize pipeline with joblib | Completed |
| Versioned model file | Completed |
| Reload and verify prediction | Completed |
| FastAPI REST API | Completed |
| POST /predict | Completed |
| GET /health | Completed |
| GET /model-info | Completed |
| curl testing ≥ 3 samples | Completed |
| Swagger documentation `/docs` | Completed |


Module 3 completed successfully.