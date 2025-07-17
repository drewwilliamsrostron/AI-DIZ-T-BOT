import artibot.globals as G
from artibot.ensemble import reject_if_risky


def test_gate_triggered(monkeypatch):
    monkeypatch.setattr(G, "risk_filter_enabled", True)
    monkeypatch.setattr(G, "global_num_trades", 2000)
    assert reject_if_risky(reward=0.8, max_dd=-0.2, entropy=0.5) is True
