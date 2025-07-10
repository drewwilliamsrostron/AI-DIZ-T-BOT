from artibot.ensemble import reject_if_risky


def test_reject_if_risky():
    assert reject_if_risky(reward=0.5, max_dd=-0.4, entropy=0.1) is False
    assert reject_if_risky(reward=1.4, max_dd=-0.2, entropy=0.3) is False
