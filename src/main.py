from data import load_data
from pipeline import build_pipeline

X_train, X_test, y_train, y_test = load_data()

# -- Check data.py --
# print("Train:", X_train.shape)
# print("Test :", X_test.shape)

# -- Check pipeline.py --
pipeline = build_pipeline()
print(pipeline)