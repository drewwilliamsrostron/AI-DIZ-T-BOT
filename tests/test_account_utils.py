from artibot.utils.account import get_account_equity


class DummyEx:
    def __init__(self):
        self.calls = []

    def fetch_balance(self, params=None):
        self.calls.append(params)
        return {"BTC": {"total": 0.5}}

    def fetch_ticker(self, pair):
        assert pair == "BTC/USDT"
        return {"close": 1000.0}


def test_get_account_equity():
    ex = DummyEx()
    eq = get_account_equity(ex)
    assert eq == 1000.0  # (0.5 + 0.5) * 1000
    assert ex.calls == [None, {"type": "swap", "code": "BTC"}]


