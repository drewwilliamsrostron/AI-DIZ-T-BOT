import artibot.globals as G


def test_bot_running_toggle():
    G.set_bot_running(False)
    assert not G.is_bot_running()
    G.set_bot_running(True)
    assert G.is_bot_running()
