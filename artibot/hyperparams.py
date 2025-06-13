"""Hyperparameter configuration loaded from ``master_config.json``."""

from __future__ import annotations

from dataclasses import dataclass

import artibot.globals as G

import json
import os


def _load_master_config(path: str = "master_config.json") -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    cfg_path = os.path.join(root, path)
    try:
        with open(cfg_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


_CONFIG = _load_master_config()


@dataclass
class HyperParams:
    """Training and indicator settings.

    Parameters default to values in ``master_config.json`` when available.
    """

    learning_rate: float = float(_CONFIG.get("LEARNING_RATE", 3e-4))
    weight_decay: float = float(_CONFIG.get("WEIGHT_DECAY", 1e-4))
    sl: float = float(_CONFIG.get("SL", 5.0))
    tp: float = float(_CONFIG.get("TP", 5.0))
    atr_threshold_k: float = float(_CONFIG.get("ATR_THRESHOLD_K", 1.5))
    conf_threshold: float = float(_CONFIG.get("CONF_THRESHOLD", 5e-5))

    long_frac: float = 0.0  # fraction of equity in LONG contracts
    short_frac: float = 0.0  # fraction in SHORT contracts

    indicator_hp: "IndicatorHyperparams" = None

    use_sma: bool = bool(_CONFIG.get("USE_SMA", True))
    use_vortex: bool = bool(_CONFIG.get("USE_VORTEX", True))
    use_cmf: bool = bool(_CONFIG.get("USE_CMF", True))
    use_ichimoku: bool = bool(_CONFIG.get("USE_ICHIMOKU", False))
    use_atr: bool = bool(_CONFIG.get("USE_ATR", True))
    use_momentum: bool = bool(_CONFIG.get("USE_MOMENTUM", False))
    use_bbw: bool = bool(_CONFIG.get("USE_BBW", False))

    def __post_init__(self) -> None:
        if self.indicator_hp is None:
            self.indicator_hp = IndicatorHyperparams()
        self.long_frac = max(0.0, min(self.long_frac, G.MAX_SIDE_EXPOSURE_PCT))
        self.short_frac = max(0.0, min(self.short_frac, G.MAX_SIDE_EXPOSURE_PCT))
        G.sync_globals(self, self.indicator_hp)

    @property
    def atr_period(self) -> int:
        return self.indicator_hp.atr_period


###############################################################################
# Dataclass for indicator-specific hyperparams
###############################################################################


@dataclass
class IndicatorHyperparams:
    """Periods and toggles for optional indicators."""

    use_sma: bool = True
    sma_period: int = 10
    use_rsi: bool = True
    rsi_period: int = 9
    use_macd: bool = True
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    use_atr: bool = True
    atr_period: int = 14
    use_vortex: bool = True
    vortex_period: int = 14
    use_cmf: bool = True
    cmf_period: int = 20

    def __post_init__(self) -> None:
        mapping = {
            "use_sma": "USE_SMA",
            "sma_period": "SMA_PERIOD",
            "use_rsi": "USE_RSI",
            "rsi_period": "RSI_PERIOD",
            "use_macd": "USE_MACD",
            "macd_fast": "MACD_FAST",
            "macd_slow": "MACD_SLOW",
            "macd_signal": "MACD_SIGNAL",
            "use_atr": "USE_ATR",
            "atr_period": "ATR_PERIOD",
            "use_vortex": "USE_VORTEX",
            "vortex_period": "VORTEX_PERIOD",
            "use_cmf": "USE_CMF",
            "cmf_period": "CMF_PERIOD",
        }
        for attr, key in mapping.items():
            if key in _CONFIG:
                cur = getattr(self, attr)
                typ = type(cur)
                try:
                    setattr(self, attr, typ(_CONFIG[key]))
                except Exception:
                    pass
