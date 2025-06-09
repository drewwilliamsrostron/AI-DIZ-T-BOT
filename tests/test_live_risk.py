import artibot.globals as G
from artibot.live_risk import update_auto_pause


def test_auto_pause_low_sharpe():
    G.live_sharpe_history.clear()
    G.live_drawdown_history.clear()
    G.trading_paused = False
    ts = 0
    for _ in range(7):
        update_auto_pause(0.9, -0.05, ts=ts)
        ts += 86400
    assert G.trading_paused is True


def test_auto_pause_drawdown():
    G.live_sharpe_history.clear()
    G.live_drawdown_history.clear()
    G.trading_paused = False
    update_auto_pause(1.2, -0.05, ts=0)
    update_auto_pause(1.2, -0.11, ts=12 * 3600)
    assert G.trading_paused is True
