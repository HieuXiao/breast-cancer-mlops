from data import load_data

X_train, X_test, y_train, y_test = load_data()

print("Train:", X_train.shape)
print("Test :", X_test.shape)