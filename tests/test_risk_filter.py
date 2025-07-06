import pandas as pd
from artibot.rules.risk_filter import apply as risk_filter


def test_risk_filter_apply():
    df = pd.DataFrame(
        {
            "unix": [0, 1, 2],
            "open": [1.0, 1.0, -1.0],
            "high": [1.1, 1.2, 0.9],
            "low": [0.9, 0.8, 0.8],
            "close": [1.05, 1.1, -1.0],
            "volume_btc": [0.1, 0.2, 0.3],
        }
    )

    filtered = risk_filter(df, enabled=True)
    assert len(filtered) == 2

    disabled = risk_filter(df, enabled=False)
    assert len(disabled) == 3

