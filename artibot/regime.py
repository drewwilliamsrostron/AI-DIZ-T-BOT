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


# --------------------------------------------------------------------------- #
# Fast batch regime classification                                            #
# --------------------------------------------------------------------------- #
# When back-testing millions of bars we can’t afford to fit a K-Means model   #
# thousands of times in a Python for-loop.  This helper fits once and then    #
# predicts all regime labels in a vectorised way.                             #
#                                                                             #
# Args                                                                        #
# -----                                                                       #
# prices : np.ndarray[float]                                                  #
#   Close-price series (1-D).                                                 #
# n_clusters : int, default 3                                                 #
#   Number of regimes to detect – must match the single-sample version        #
#   classify_market_regime().                                                 #
#                                                                             #
# Returns                                                                     #
# -------                                                                     #
# list[int]                                                                   #
#   Regime label for every bar, length == len(prices).                        #
# --------------------------------------------------------------------------- #


def classify_market_regime_batch(
    prices: np.ndarray, *, n_clusters: int = 3, vol_window: int = 20
) -> list[int]:
    """Vectorised K-Means regime labelling for an entire price series."""

    import numpy as np
    from sklearn.cluster import KMeans

    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or prices.size < vol_window + 2:
        return [0] * len(prices)

    log_ret = np.diff(np.log(prices), prepend=np.log(prices[0]))
    vol = (
        np.convolve(log_ret**2, np.ones(vol_window), "full")[: len(log_ret)]
        / vol_window
    ) ** 0.5
    sma_short = np.convolve(prices, np.ones(5) / 5.0, "same")
    sma_long = np.convolve(prices, np.ones(20) / 20.0, "same")
    trend = (sma_short - sma_long) / (prices + 1e-9)

    feats = np.column_stack([vol, trend])

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(feats)
    return labels.tolist()
