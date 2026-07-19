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
```

## 5. Conclusion

PSI detects distribution changes between training and production data.

KS-test confirms whether the observed difference is statistically significant.

Prediction logging provides traceability for production inference requests.

Monitoring input distribution and prediction behavior helps detect potential model degradation before ground-truth labels become available.


---

# 6. Phản biện

## Câu 1. Phân biệt Data Drift và Concept Drift

### Data Drift

Data Drift xảy ra khi phân phối dữ liệu đầu vào thay đổi so với dữ liệu dùng để huấn luyện mô hình, trong khi mối quan hệ giữa đầu vào và nhãn vẫn giữ nguyên.

Ví dụ trong thương mại điện tử:

- Mô hình dự đoán khả năng khách hàng mua hàng được huấn luyện từ dữ liệu năm 2024.
- Đến năm 2026, độ tuổi và khu vực của khách hàng thay đổi do mở rộng thị trường.
- Nhãn "có mua hàng hay không" vẫn được định nghĩa như cũ nhưng phân phối đặc trưng đầu vào đã khác.

Data Drift có thể được phát hiện mà **không cần nhãn thật** bằng cách theo dõi phân phối dữ liệu thông qua các chỉ số như PSI hoặc KS-test.

---

### Concept Drift

Concept Drift xảy ra khi mối quan hệ giữa dữ liệu đầu vào và nhãn đầu ra thay đổi theo thời gian.

Ví dụ trong thương mại điện tử:

- Trước đây khách hàng thường mua khi được giảm giá 10%.
- Sau một thời gian, do thay đổi hành vi tiêu dùng hoặc chiến lược của đối thủ, mức giảm giá 10% không còn đủ hấp dẫn.
- Phân phối dữ liệu đầu vào có thể gần như không đổi nhưng quy luật quyết định mua hàng đã thay đổi.

Concept Drift **không thể phát hiện chỉ bằng dữ liệu đầu vào**, mà cần có nhãn thực tế để đánh giá sự suy giảm hiệu năng của mô hình.

---

## Câu 2. Vì sao mô hình "hỏng trong im lặng"?

Một mô hình có thể tiếp tục hoạt động bình thường và vẫn trả về dự đoán mà không phát sinh bất kỳ lỗi (exception) nào, mặc dù chất lượng dự đoán đã giảm đáng kể. Hiện tượng này được gọi là **silent failure**.

Nguyên nhân là vì:

- Dữ liệu đầu vào vẫn đúng định dạng.
- Pipeline vẫn xử lý được dữ liệu.
- API vẫn trả về kết quả bình thường.

Tuy nhiên, nếu phân phối dữ liệu production khác với dữ liệu huấn luyện thì mô hình có thể đưa ra các dự đoán kém chính xác mà hệ thống không tự nhận biết.

Do nhãn thực tế thường chỉ có sau nhiều ngày hoặc nhiều tuần, việc theo dõi phân phối đầu vào và phân phối đầu ra giúp phát hiện sớm các dấu hiệu bất thường trước khi có thể đánh giá lại độ chính xác của mô hình.

---

## Câu 3. Đặt ngưỡng cảnh báo theo tác động kinh doanh

Ngưỡng cảnh báo không nên chỉ dựa trên giá trị thống kê mà còn cần xét đến mức độ ảnh hưởng của bài toán.

Ví dụ:

### Bài toán chẩn đoán ung thư

- PSI = 0.10 đã nên được cảnh báo.
- Sai sót trong dự đoán có thể ảnh hưởng trực tiếp đến sức khỏe và tính mạng bệnh nhân.
- Do đó cần phát hiện drift sớm và đánh giá lại mô hình ngay.

### Bài toán gợi ý sản phẩm

- PSI = 0.20 hoặc cao hơn mới cần cảnh báo.
- Sai lệch dự đoán chủ yếu làm giảm doanh thu hoặc trải nghiệm người dùng.
- Có thể chấp nhận mức biến động lớn hơn trước khi quyết định retrain.

Điều này cho thấy cùng một giá trị PSI nhưng mức cảnh báo có thể khác nhau tùy theo mức độ rủi ro của từng hệ thống.

---

## Câu 4. Quy trình xử lý khi hệ thống phát hiện Drift

Khi hệ thống giám sát phát hiện drift, quy trình xử lý gồm các bước:

1. Xác nhận cảnh báo bằng PSI, KS-test và các chỉ số giám sát khác.
2. Kiểm tra dữ liệu production để xác định nguyên nhân thay đổi.
3. Đánh giá hiệu năng của mô hình nếu đã có nhãn thực tế.
4. Quyết định hành động phù hợp.

Có ba chiến lược phổ biến:

### Retrain theo lịch (Scheduled Retraining)

Áp dụng khi dữ liệu thay đổi đều theo thời gian, ví dụ huấn luyện lại mỗi tháng hoặc mỗi quý.

### Retrain theo Trigger (Trigger-based Retraining)

Áp dụng khi hệ thống phát hiện drift vượt ngưỡng hoặc hiệu năng mô hình giảm xuống dưới mức cho phép.

### Rollback

Nếu mô hình mới hoạt động kém hơn mô hình đang triển khai, hệ thống sẽ quay trở lại phiên bản trước trong Model Registry để đảm bảo tính ổn định của dịch vụ.