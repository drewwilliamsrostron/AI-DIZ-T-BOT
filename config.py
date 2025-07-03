FEATURE_CONFIG = {
    "expected_features": 16,
    "strict_validation": True,
    "feature_columns": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_10",
        "sma_50",
        "rsi",
        "macd",
        "boll_upper",
        "boll_lower",
        "vwap",
        "obv",
        "stoch_k",
        "stoch_d",
        "atr",
    ],
}

# Convenient alias for core modules
FEATURE_COLUMNS = FEATURE_CONFIG["feature_columns"]
