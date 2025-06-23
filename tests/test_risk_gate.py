import artibot.globals as G
from artibot.ensemble import reject_if_risky


def test_early_stage_gate_relaxed(monkeypatch):
    monkeypatch.setattr(G, "global_num_trades", 500)
    assert reject_if_risky(sharpe=0.2, max_dd=-0.5, entropy=0.5) is False
