class ExchangeConnector:
    """Thin wrapper around ccxt for simple order placement."""

    def __init__(self, config):
        import ccxt

        api_conf = config.get("API", {})
        self.live = bool(api_conf.get("LIVE_TRADING", False))
        key = (
            api_conf.get("API_KEY_LIVE") if self.live else api_conf.get("API_KEY_TEST")
        )
        secret = (
            api_conf.get("API_SECRET_LIVE")
            if self.live
            else api_conf.get("API_SECRET_TEST")
        )
        self.exchange = ccxt.phemex(
            {"apiKey": key, "secret": secret, "enableRateLimit": True}
        )
        if not self.live:
            self.exchange.set_sandbox_mode(True)
        self.exchange.load_markets()

    def create_order(self, symbol, side, amount, price=None, order_type="market"):
        return self.exchange.create_order(symbol, order_type, side, amount, price)
