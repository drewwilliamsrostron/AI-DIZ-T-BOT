"""Utilities for market regime detection."""

from __future__ import annotations

import logging
import os
import numpy as np
from hmmlearn.hmm import GaussianHMM
from artibot.regime_encoder import RegimeEncoder


def _probs_to_labels(probs, thresh: float = 0.6):
    """
    Convert an (N, K) soft‑probability matrix to integer labels.
    If max‑probability < thresh the label is −1 (signals ‘blend’ mode).
    Returns (labels, probs) so the caller can keep the original soft scores.
    """
    labels = []
    for p in probs:  # 1‑D np.ndarray of length K
        top = int(p.argmax())
        if p[top] >= thresh:
            labels.append(top)  # confident – hard switch
        else:
            labels.append(-1)  # low‑confidence – soft blend
    return labels, probs


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


_last_regime_encoder: RegimeEncoder | None = None


def classify_market_regime(prices: np.ndarray, lookback: int = 168) -> int:
    """Return the most likely current regime using the global encoder."""

    global _last_regime_encoder

    if prices.size < lookback or _last_regime_encoder is None:
        return 0

    window = prices[-lookback:]
    try:
        probs = _last_regime_encoder.encode_sequence(window)
        if len(probs) == 0:
            return 0
        return int(np.argmax(probs[-1]))
    except Exception as exc:  # pragma: no cover - safety net
        logging.warning("regime classification failed: %s", exc)
        return 0


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
# tuple[list[int], np.ndarray]                                                #
#   Hard regime labels and the corresponding soft probabilities.              #
# --------------------------------------------------------------------------- #


def classify_market_regime_batch(
    prices: np.ndarray, *, n_clusters: int | str = 3, vol_window: int = 20
) -> tuple[list[int], np.ndarray]:
    """Vectorised regime labelling using the :class:`RegimeEncoder`."""

    global _last_regime_encoder

    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        k = int(n_clusters) if isinstance(n_clusters, int) else 3
        return [0] * len(prices), np.zeros((len(prices), k))

    if n_clusters == "auto":
        n_clusters = 2 if len(prices) < 1500 else 3
    k = int(n_clusters)

    if _last_regime_encoder is None:
        _last_regime_encoder = RegimeEncoder(n_regimes=int(n_clusters))
        if os.path.isfile("encoder.pt"):
            try:
                _last_regime_encoder.load("encoder.pt")
            except Exception as exc:  # pragma: no cover - load failure
                logging.warning("failed to load encoder: %s", exc)
        else:
            try:
                _last_regime_encoder.train_unsupervised(prices, epochs=1)
                _last_regime_encoder.save("encoder.pt")
            except Exception as exc:  # pragma: no cover - train failure
                logging.warning("failed to train encoder: %s", exc)
                return [0] * len(prices), np.zeros((len(prices), k))

    try:
        probs = _last_regime_encoder.encode_sequence(prices)
    except Exception as exc:  # pragma: no cover - encode failure
        logging.warning("failed to encode sequence: %s", exc)
        return [0] * len(prices), np.zeros((len(prices), k))

    if len(probs) == 0:
        return [0] * len(prices), np.zeros((len(prices), k))

    labels, probs = _probs_to_labels(probs, thresh=0.6)
    uniq = np.unique(labels)
    if len(uniq) == 1:
        logging.warning(
            "Encoder produced only one hard regime label – switching to blend‑mode everywhere"
        )
        labels = [-1] * len(labels)

    pad = len(prices) - len(labels)
    if pad > 0:
        labels = np.pad(labels, (pad, 0), constant_values=labels[0])

    # simple smoothing: ignore single-sample spikes
    smooth = labels.copy()
    for i in range(1, len(smooth)):
        if i >= 2 and smooth[i] != smooth[i - 1] and smooth[i] != smooth[i - 2]:
            smooth[i] = smooth[i - 1]
    labels = smooth

    transitions = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    logging.info("Regime sequence computed, transitions at indices: %s", transitions)
    return labels.tolist(), probs
