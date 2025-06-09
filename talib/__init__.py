"""Minimal TA-Lib stub for tests."""

import numpy as np


def RSI(arr, timeperiod=14):
    return np.zeros_like(arr)


def MACD(arr, fastperiod=12, slowperiod=26, signalperiod=9):
    zeros = np.zeros_like(arr)
    return zeros, zeros, zeros
