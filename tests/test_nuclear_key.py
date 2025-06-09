from artibot.metrics import nuclear_key_condition


def test_nuclear_key_condition():
    assert nuclear_key_condition(1.6, -0.1, 1.6)
    assert not nuclear_key_condition(1.4, -0.1, 1.6)
    assert not nuclear_key_condition(1.6, -0.3, 1.6)
    assert not nuclear_key_condition(1.6, -0.1, 1.2)
