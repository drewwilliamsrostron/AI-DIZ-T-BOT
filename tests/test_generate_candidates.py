from artibot.utils.markets import generate_candidates


def test_single_token_variants():
    cands = generate_candidates("BTC")
    assert "BTC/USD" in cands
    assert "BTCUSD" in cands
    assert "BTC/USD:USD" in cands
    assert "BTCUSDT" in cands


def test_two_token_variants():
    cands = generate_candidates("ETH/USDT")
    assert "ETH/USDT" in cands
    assert "ETHUSDT" in cands
