"""CCXT-based helper for simplified Phemex access."""

import logging
import time


class ExchangeConnector:
    """Thin wrapper around :mod:`ccxt` for simple order placement."""

    def __init__(self, config):
        import ccxt

        self.symbol = config.get("symbol", "BTC/USDT")
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
        default_type = api_conf.get("DEFAULT_TYPE", "swap")
        self.default_type = default_type
        api_url_live = api_conf.get("API_URL_LIVE")
        api_url_test = api_conf.get("API_URL_TEST")

        params = {
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": default_type},
        }
        if api_url_live or api_url_test:
            params["urls"] = {
                "api": {
                    "live": api_url_live,
                    "test": api_url_test,
                }
            }

        self.exchange = ccxt.phemex(params)
        self.exchange.set_sandbox_mode(not self.live)
        self.exchange.load_markets()

    def fetch_latest_bars(self, timeframe: str = "1h", limit: int = 24):
        now = int(time.time())
        last_hour = now - (now % 3600)
        since_ms = (last_hour - limit * 3600) * 1000
        params = {"type": self.default_type}
        logging.debug(
            "fetch_ohlcv → symbol=%s params=%s",
            self.symbol,
            params,
        )
        try:
            bars = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=limit,
                params=params,
            )
        except Exception as primary:  # pragma: no cover - network errors
            logging.error("primary fetch failed: %s", primary)
            try:
                market_id = self.exchange.market(self.symbol)["id"]
                logging.debug("retrying fetch with market_id %s", market_id)
                bars = self.exchange.fetch_ohlcv(
                    market_id,
                    timeframe=timeframe,
                    since=since_ms,
                    limit=limit,
                )
            except Exception as fallback:  # pragma: no cover - network errors
                logging.error("fallback fetch failed: %s", fallback)
                return []
        logging.info("Fetched %d bars", len(bars) if bars else 0)
        return bars if bars else []

    def create_order(self, side: str, amount: float, price=None, order_type="market"):
        return self.exchange.create_order(self.symbol, order_type, side, amount, price)
