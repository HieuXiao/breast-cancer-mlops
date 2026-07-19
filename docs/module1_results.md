# Module 1 — ML Pipeline và Thí nghiệm Data Leakage

## 1. Mục tiêu

Mục tiêu của Module 1 là xây dựng một quy trình Machine Learning có khả năng tái lập (reproducible) và tìm hiểu cách Pipeline giúp ngăn chặn hiện tượng Data Leakage.

Các nội dung thực hiện gồm:

- Nạp bộ dữ liệu Breast Cancer.
- Chia dữ liệu bằng `train_test_split` với `stratify` và `random_state`.
- Xây dựng Pipeline gồm:
  - StandardScaler
  - LogisticRegression
- Huấn luyện và đánh giá mô hình.
- Đánh giá bằng Cross Validation (5-fold).
- Thực hiện thí nghiệm Data Leakage và so sánh với quy trình đúng.

---

# 2. Chuẩn bị dữ liệu

## Dataset

- Nguồn: `sklearn.datasets.load_breast_cancer`
- Bài toán: Phân loại nhị phân
- Số mẫu: 569
- Số đặc trưng: 30

## Chia dữ liệu

| Tham số | Giá trị |
|---------|----------|
| Test size | 20% |
| Random state | 42 |
| Stratify | Có |

Kết quả:

```
Train: (455, 30)
Test : (114, 30)
```

---

# 3. Xây dựng Machine Learning Pipeline

Pipeline được xây dựng theo quy trình:

```
Input Features
        │
        ▼
StandardScaler
        │
        ▼
LogisticRegression
        │
        ▼
Prediction
```

Cài đặt:

```python
Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42))
    ]
)
```

Pipeline đảm bảo bước tiền xử lý (StandardScaler) chỉ được huấn luyện trên tập Train trước khi mô hình được huấn luyện.

---

# 4. Đánh giá trên tập Test

Mô hình được huấn luyện trên tập Train và đánh giá trên tập Test.

## Kết quả

| Chỉ số | Giá trị |
|---------|---------:|
| Accuracy | 0.9825 |
| F1-score | 0.9861 |
| ROC-AUC | 0.9954 |

## Nhận xét

- Accuracy cho thấy mô hình dự đoán đúng phần lớn mẫu trong tập kiểm tra.
- F1-score cao chứng tỏ mô hình cân bằng tốt giữa Precision và Recall.
- ROC-AUC gần bằng 1 cho thấy khả năng phân biệt hai lớp rất tốt.

---

# 5. Đánh giá bằng Cross Validation (5-fold)

Cross Validation được thực hiện với:

```python
cv = 5
```

## Accuracy

Kết quả từng Fold:

```
[0.96703297 0.97802198 0.96703297 1.00000000 0.98901099]
```

Tổng hợp:

| Chỉ số | Giá trị |
|---------|---------:|
| Mean | 0.9802 |
| Std | 0.0128 |

---

## ROC-AUC

Kết quả từng Fold:

```
[0.99638803 1.00000000 0.97213622 1.00000000 0.99896801]
```

Tổng hợp:

| Chỉ số | Giá trị |
|---------|---------:|
| Mean | 0.9935 |
| Std | 0.0108 |

## Nhận xét

Kết quả giữa các Fold có sự dao động nhỏ, chứng tỏ mô hình hoạt động ổn định và có khả năng tổng quát hóa tốt trên nhiều cách chia dữ liệu khác nhau.

---

# 6. Thí nghiệm Data Leakage

## Quy trình sai

Trong thí nghiệm này, dữ liệu được chuẩn hóa trước khi chia Train/Test.

```
Toàn bộ Dataset
        │
        ▼
StandardScaler.fit_transform()
        │
        ▼
Train/Test Split
        │
        ▼
Huấn luyện mô hình
```

Đây là quy trình **không đúng**, vì StandardScaler đã sử dụng thông tin của cả tập Test để tính toán Mean và Standard Deviation.

---

## So sánh kết quả

| Quy trình | ROC-AUC |
|------------|---------:|
| Pipeline đúng | 0.9954 |
| Data Leakage | 0.9954 |
| Chênh lệch | 0.0000 |

## Nhận xét

Trong bộ dữ liệu Breast Cancer của Scikit-learn, việc chuẩn hóa toàn bộ dữ liệu trước khi chia Train/Test không làm thay đổi ROC-AUC.

Tuy nhiên, đây vẫn là một quy trình sai vì thông tin của tập Test đã được sử dụng trong bước tiền xử lý. Trên các bộ dữ liệu khác hoặc các bài toán phức tạp hơn, Data Leakage có thể làm kết quả đánh giá trở nên quá lạc quan và không phản ánh đúng khả năng tổng quát hóa của mô hình.

---

# 7. Trả lời câu hỏi phản biện

## 7.1 Pipeline chống Data Leakage trong Cross Validation như thế nào?

Pipeline kết hợp toàn bộ các bước tiền xử lý và huấn luyện thành một quy trình thống nhất.

Trong mỗi Fold của Cross Validation:

```
Training Fold
      │
      ▼
Scaler.fit()
      │
      ▼
Scaler.transform()
      │
      ▼
Model.fit()

Validation Fold
      │
      ▼
Scaler.transform()
      │
      ▼
Prediction
```

StandardScaler chỉ được học (`fit`) trên dữ liệu Train của từng Fold.

Đối với Validation Fold, StandardScaler chỉ thực hiện `transform()` bằng các thống kê đã học từ Train mà không học lại.

Nhờ đó, thông tin từ Validation không bị rò rỉ vào quá trình huấn luyện mô hình.

---

## 7.2 Nếu thay LogisticRegression bằng RandomForest thì StandardScaler còn cần thiết không?

Thông thường là **không cần**.

RandomForest là mô hình dựa trên cây quyết định (Decision Tree).

Mỗi nút của cây chỉ quan tâm đến điều kiện dạng:

```
Feature < Threshold
```

Việc chuẩn hóa dữ liệu không làm thay đổi thứ tự giữa các giá trị nên gần như không ảnh hưởng đến cách cây quyết định phân chia dữ liệu.

StandardScaler thường chỉ cần thiết với các mô hình dựa trên khoảng cách hoặc tối ưu hóa hệ số như:

- Logistic Regression
- Support Vector Machine (SVM)
- Neural Network
- K-Nearest Neighbors (KNN)

---

## 7.3 Vì sao Data Leakage có thể cho điểm cao hơn nhưng là điểm giả?

Khi chuẩn hóa toàn bộ dữ liệu trước khi chia Train/Test, StandardScaler đã sử dụng thông tin thống kê của cả tập Test.

Điều này khiến mô hình gián tiếp tiếp cận thông tin từ dữ liệu dùng để đánh giá.

Do đó, các chỉ số đánh giá có thể cao hơn thực tế và không phản ánh đúng khả năng tổng quát hóa của mô hình trên dữ liệu chưa từng xuất hiện.

Trong dự án này, ROC-AUC của hai quy trình đều bằng **0.9954**, nhưng điều đó không có nghĩa quy trình Data Leakage là đúng. Đây chỉ là đặc điểm của bộ dữ liệu Breast Cancer, nơi sự khác biệt giữa thống kê của Train và toàn bộ dữ liệu là rất nhỏ.

---

# 8. Kết luận

Qua Module 1 có thể rút ra các kết luận sau:

- Quy trình Machine Learning cần tách biệt rõ các bước chuẩn bị dữ liệu, tiền xử lý, huấn luyện và đánh giá.
- Pipeline giúp ngăn chặn Data Leakage bằng cách đảm bảo mọi bước tiền xử lý chỉ được huấn luyện trên dữ liệu Train.
- Cross Validation giúp đánh giá tính ổn định và khả năng tổng quát hóa của mô hình tốt hơn so với một lần Train/Test Split.
- Điểm đánh giá cao chỉ có ý nghĩa khi quy trình đánh giá được thực hiện đúng và không xảy ra Data Leakage.