"""CCXT-based helper for simplified Phemex access."""

import logging
import os


class ExchangeConnector:
    """Thin wrapper around :mod:`ccxt` for simple order placement."""

    def __init__(self, config):
        import ccxt

        # trading pair is fixed for now
        self.symbol = "BTCUSD"
        api_conf = config.get("API", {})
        self.live = bool(api_conf.get("LIVE_TRADING", False))
        key = os.getenv(
            "PHEMEX_KEY",
            api_conf.get("API_KEY_LIVE") if self.live else api_conf.get("API_KEY_TEST"),
        )
        secret = os.getenv(
            "PHEMEX_SECRET",
            (
                api_conf.get("API_SECRET_LIVE")
                if self.live
                else api_conf.get("API_SECRET_TEST")
            ),
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
        params["urls"] = {"api": {"live": api_url_live, "test": api_url_test}}

        self.exchange = ccxt.phemex(params)
        self.exchange.set_sandbox_mode(not self.live)
        self.exchange.load_markets()

    def fetch_latest_bars(self, timeframe: str = "1h", limit: int = 100):
        logging.debug(
            "fetch_ohlcv → symbol=%s",
            self.symbol,
        )
        try:
            bars = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe=timeframe,
                limit=limit,
            )
        except Exception as primary:  # pragma: no cover - network errors
            logging.error(
                "fetch_ohlcv failed for %s tf=%s limit=%s: %s",
                self.symbol,
                timeframe,
                limit,
                primary,
            )
            if hasattr(self.exchange, "market"):
                market_id = self.symbol
                try:
                    market_id = self.exchange.market(self.symbol)["id"]
                    logging.debug("retrying fetch with market_id %s", market_id)
                    bars = self.exchange.fetch_ohlcv(
                        market_id,
                        timeframe=timeframe,
                        limit=limit,
                    )
                except Exception as fallback:  # pragma: no cover - network errors
                    logging.error(
                        "fetch_ohlcv failed for %s tf=%s limit=%s: %s",
                        market_id,
                        timeframe,
                        limit,
                        fallback,
                    )
                    return []
            else:
                return []
        logging.info("Fetched %d bars", len(bars) if bars else 0)
        return bars if bars else []

    def create_order(
        self,
        side: str,
        amount: float,
        price=None,
        order_type="market",
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ):
        params = {"type": self.default_type}
        if stop_loss is not None:
            params["stopLossPrice"] = stop_loss
        if take_profit is not None:
            params["takeProfitPrice"] = take_profit
        return self.exchange.create_order(
            self.symbol, order_type, side, amount, price, params
        )
