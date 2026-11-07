"""Synthetic-data Random Forest forecasting for the Peak Prediction Agent.

This educational model is evaluated using held-out synthetic trajectories.
No claim is made about accuracy on real NEOM traffic data.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def features(history: list[float], horizon: int = 3) -> list[float]:
    """Only historical queue values are used; no future leakage."""
    if len(history) < 3:
        raise ValueError("At least 3 historical queue samples required")
    a, b, c = [float(v) for v in history[-3:]]
    return [a, b, c, b-a, c-b, (c-a)/2, float(horizon)]


def _training_data(seed: int = 461):
    rng = np.random.default_rng(seed)
    x, y, runs = [], [], []
    # Train/test are split by independent *trajectory*, not by adjacent windows.
    for run in range(320):
        initial = float(rng.uniform(0, 35))
        drift = float(rng.uniform(-4, 6))
        noise = float(rng.uniform(0.0, 1.2))
        horizon = int(rng.integers(1, 7))
        trajectory = [initial]
        for t in range(18):
            new = max(0., min(90., trajectory[-1] + drift + rng.normal(0, noise)))
            trajectory.append(new)
        for t in range(2, len(trajectory)-horizon):
            x.append(features(trajectory[t-2:t+1], horizon))
            y.append(trajectory[t+horizon])
            runs.append(run)
    return np.asarray(x), np.asarray(y), np.asarray(runs)


@lru_cache(maxsize=1)
def trained_model():
    x, y, runs = _training_data()
    training = runs < 256
    model = RandomForestRegressor(n_estimators=70, max_depth=12, min_samples_leaf=2, n_jobs=1, random_state=461)
    model.fit(x[training], y[training])
    mae = float(mean_absolute_error(y[~training], model.predict(x[~training])))
    return model, mae


def predict_queue(history: list[float], horizon: int = 3) -> float:
    model, _ = trained_model()
    return max(0., float(model.predict([features(history, horizon)])[0]))
