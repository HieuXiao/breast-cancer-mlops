# breast-cancer-mlops\src\data.py

# -- Data loading utilities for the Breast Cancer dataset.--

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_data(test_size: float = 0.2, random_state: int = 42):
    # Load the Breast Cancer dataset and split it into training and testing sets.
    dataset = load_breast_cancer()

    x = dataset.data
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size = test_size,
        stratify = y,
        random_state = random_state,
    )

    return x_train, x_test, y_train, y_test