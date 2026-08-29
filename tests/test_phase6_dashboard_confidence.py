import numpy as np
import pandas as pd


def dashboard_status(predictions, interval_id, timestamp):
    matches = predictions.loc[
        (predictions["independent_interval_id"] == interval_id)
        & (predictions["timestamp"] == pd.Timestamp(timestamp))
    ].copy()
    probabilities = pd.to_numeric(matches["probability"], errors="coerce").dropna()
    event_probability = float(np.clip(probabilities.mean(), 0.0, 1.0))
    positive = event_probability >= 0.5
    confidence = event_probability if positive else 1.0 - event_probability
    return positive, confidence


def test_ensemble_confidence_is_not_saturated_for_sample_minute():
    ts = pd.Timestamp("2024-09-01 00:00:00")
    rows = pd.DataFrame(
        {
            "independent_interval_id": ["SEP", "SEP", "SEP"],
            "timestamp": [ts, ts, ts],
            "model": ["Logistic Regression", "Random Forest", "Gradient Boosting"],
            "probability": [0.001, 0.20, 0.08],
        }
    )
    positive, confidence = dashboard_status(rows, "SEP", ts)
    assert not positive
    assert 0.90 < confidence < 1.0
