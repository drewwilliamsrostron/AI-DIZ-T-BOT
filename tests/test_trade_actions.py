import artibot.globals as G


def test_update_trade_params_and_close():
    G.update_trade_params(2.0, 3.0)
    assert G.global_SL_multiplier == 2.0
    assert G.global_TP_multiplier == 3.0
    G.cancel_open_orders()
    G.close_position()
