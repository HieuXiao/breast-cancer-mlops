# breast-cancer-mlops/tests/test_pipeline.py

import numpy as np
import pytest

from src.data import load_data
# pyrefly: ignore [missing-import]
from src.pipeline import build_pipeline


@pytest.fixture
def trained_pipeline():
    """
    Train pipeline once and reuse for all tests.
    """
    X_train, X_test, y_train, y_test = load_data()

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    return pipeline, X_test


def test_prediction_shape(trained_pipeline):
    """
    Output prediction phải có cùng số lượng với input.
    """
    pipeline, X_test = trained_pipeline

    y_pred = pipeline.predict(X_test)

    assert y_pred.shape == (len(X_test),)


def test_prediction_binary_labels(trained_pipeline):
    """
    Prediction chỉ được chứa 0 hoặc 1.
    """
    pipeline, X_test = trained_pipeline

    y_pred = pipeline.predict(X_test)

    assert set(y_pred).issubset({0, 1})


def test_prediction_probability_range(trained_pipeline):
    """
    Xác suất phải nằm trong [0, 1].
    """
    pipeline, X_test = trained_pipeline

    probabilities = pipeline.predict_proba(X_test)

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_pipeline_nan_input(trained_pipeline):
    """
    Pipeline phải báo lỗi khi đầu vào chứa NaN.
    """
    pipeline, X_test = trained_pipeline

    X_nan = X_test.copy()
    X_nan[0, 0] = np.nan

    with pytest.raises(ValueError):
        pipeline.predict(X_nan)