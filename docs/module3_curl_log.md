# Module 3 — REST API Prediction Log

## Model Information

Model:

`breast_cancer_pipeline_v1.joblib`

Pipeline:

```
StandardScaler + LogisticRegression
```

API Endpoint:

```
POST http://127.0.0.1:8000/predict
```

Dataset:

```
sklearn.datasets.load_breast_cancer
```

Number of features:

```
30
```

Label mapping:

| Value | Meaning |
|---|---|
| 0 | malignant |
| 1 | benign |


---

# Test Sample 1

## Input

True label:

```
0 (malignant)
```

Features:

```json
[
17.99,
10.38,
122.8,
1001.0,
0.1184,
0.2776,
0.3001,
0.1471,
0.2419,
0.07871,
1.095,
0.9053,
8.589,
153.4,
0.006399,
0.04904,
0.05373,
0.01587,
0.03003,
0.006193,
25.38,
17.33,
184.6,
2019.0,
0.1622,
0.6656,
0.7119,
0.2654,
0.4601,
0.1189
]
```

## curl command

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d "{\"features\":[17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}"
```

## Response

```json
{
  "prediction": 0,
  "label": "malignant",
  "confidence": 0.9999999938818932
}
```

Result:

```
Prediction matched true label
```


---

# Test Sample 2

## Input

True label:

```
0 (malignant)
```

## Response

```json
{
  "prediction": 0,
  "label": "malignant",
  "confidence": 0.9530784229949721
}
```

Result:

```
Prediction matched true label
```


---

# Test Sample 3

## Input

True label:

```
0 (malignant)
```

## Response

```json
{
  "prediction": 0,
  "label": "malignant",
  "confidence": 0.9560041458362656
}
```

Result:

```
Prediction matched true label
```


---

# Summary

| Sample | True Label | Prediction | Confidence | Result |
|-|-|-|-|-|
| 1 | malignant | malignant | 0.99999999 | Correct |
| 2 | malignant | malignant | 0.95307842 | Correct |
| 3 | malignant | malignant | 0.95600415 | Correct |


Conclusion:

The deployed FastAPI service successfully loaded the serialized pipeline and returned valid predictions for three real samples.