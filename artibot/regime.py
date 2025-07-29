"""Utilities for market regime detection."""

from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans


def detect_volatility_regime(prices: np.ndarray, n_states: int = 2) -> int:
    """Infer the current volatility regime using a Gaussian HMM.

    Parameters
    ----------
    prices : np.ndarray
        Sequence of prices (e.g., closing prices) ordered by time.
    n_states : int, default 2
        Number of hidden states for the HMM.

    Returns
    -------
    int
        Index of the inferred current regime ``0`` .. ``n_states - 1``.
    """
    if prices.size < 2:
        return 0

    returns = np.diff(np.log(prices)).reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
    model.fit(returns)
    states = model.predict(returns)
    return int(states[-1])


def classify_market_regime(prices: np.ndarray, lookback: int = 168) -> int:
    """Cluster recent volatility and trend data to label the market regime.

    Parameters
    ----------
    prices : np.ndarray
        Sequence of prices ordered by time.
    lookback : int, default 168
        Number of recent observations used for the features.

    Returns
    -------
    int
        Regime index predicted by the fitted :class:`~sklearn.cluster.KMeans`.
    """

    if prices.size < lookback:
        return 0

    recent_prices = prices[-lookback:]
    returns = np.diff(np.log(recent_prices))
    vol7d = np.std(returns)

    short_ma = np.mean(recent_prices[-20:])
    long_ma = np.mean(recent_prices[-100:])
    trend_strength = short_ma - long_ma

    features = np.array([[vol7d, trend_strength]], dtype=float)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans.fit(np.vstack([features, features * 1.01, features * 0.99]))
    return int(kmeans.predict(features)[0])
