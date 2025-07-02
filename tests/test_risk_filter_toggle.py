import artibot.globals as G
from artibot.ensemble import reject_if_risky


def test_risk_filter_toggle():
    G.set_risk_filter_enabled(False)
    assert reject_if_risky(sharpe=0.5, max_dd=-0.4, entropy=0.1) is False
    G.set_risk_filter_enabled(True)
    assert reject_if_risky(sharpe=0.5, max_dd=-0.4, entropy=0.1) is True
