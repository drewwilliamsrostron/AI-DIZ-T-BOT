import artibot.globals as G
from artibot.ensemble import nuclear_key_gate


def test_gate_uses_profit_factor():
    G.set_nuclear_key(True)
    assert not nuclear_key_gate(1.6, -0.1, 1.2, 1.0)
    assert nuclear_key_gate(1.6, -0.1, 1.2, 1.6)
