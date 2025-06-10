from artibot.ensemble import reject_if_risky


def test_gate_triggered():
    assert (
        reject_if_risky(sharpe=0.8, max_dd=-0.2, entropy=0.5, profit_factor=0.9) is True
    )
