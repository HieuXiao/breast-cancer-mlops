# Module 4 — Model Monitoring & Drift Detection Results


## 1. PSI Manual Calculation

Feature monitored:
```
mean radius
```


Expected distribution (training):

| Bin | Expected % |
|---|---|
| 1 | 0.20 |
| 2 | 0.30 |
| 3 | 0.30 |
| 4 | 0.20 |


Actual distribution (production simulation):

| Bin | Actual % |
|---|---|
| 1 | 0.10 |
| 2 | 0.20 |
| 3 | 0.40 |
| 4 | 0.30 |


Calculated:
```
PSI ≈ 0.179
```

Interpretation:
```
PSI < 0.2
No significant drift detected.
```



---

## 2. PSI Function Validation

Implemented PSI calculation in:
```
src/psi.py
```

The Python result matches the manual calculation.


---

## 3. Drift Detection Experiment

Two production scenarios were simulated.


### Scenario 1 — No Drift

Output:
```
PSI: 0.0043
KS statistic: 0.0450
KS p-value: 0.2635
```
Result:
```
No significant drift
```

Reason:

- PSI < 0.2
- KS p-value > 0.05


---

### Scenario 2 — Drift

Output:
```
PSI: 2.6426
KS statistic: 0.7010
KS p-value: 0.0000
```
Result:
```
ALERT: Drift detected
```

Reason:

- PSI >= 0.2
- KS p-value < 0.05


---

## 4. Prediction Logging Integration

Prediction logging was integrated into FastAPI `/predict` endpoint.


Logged information:

- timestamp
- input features
- prediction result
- confidence score
- model version


Log file:
```
logs/predictions.jsonl
```



Example prediction logs:


```json
{
 "timestamp": "2026-07-19T13:26:44.454566",
 "prediction": 0,
 "confidence": 0.9999999939,
 "model_version": "breast_cancer_pipeline_v1"
}

{
 "timestamp": "2026-07-19T13:27:54.692022",
 "prediction": 0,
 "confidence": 0.9530784230,
 "model_version": "breast_cancer_pipeline_v1"
}

{
 "timestamp": "2026-07-19T13:28:02.002959",
 "prediction": 0,
 "confidence": 0.9560041458,
 "model_version": "breast_cancer_pipeline_v1"
}

## 5. Conclusion

PSI detects distribution changes between training and production data.

KS-test confirms whether the observed difference is statistically significant.

Prediction logging provides traceability for production inference requests.

Monitoring input distribution and prediction behavior helps detect potential model degradation before ground-truth labels become available.