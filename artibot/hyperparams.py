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

TRANSFORMER_HEADS = int(_CONFIG.get("TRANSFORMER_HEADS", 8))


@dataclass
class HyperParams:
    """Training and indicator settings.

    Parameters default to values in ``master_config.json`` when available.
    """

    learning_rate: float = float(_CONFIG.get("LEARNING_RATE", 3e-4))
    weight_decay: float = float(_CONFIG.get("WEIGHT_DECAY", 0.0))
    sl: float = float(_CONFIG.get("SL", 5.0))
    tp: float = float(_CONFIG.get("TP", 5.0))
    atr_threshold_k: float = float(_CONFIG.get("ATR_THRESHOLD_K", 1.5))
    conf_threshold: float = float(_CONFIG.get("CONF_THRESHOLD", 5e-5))

    # desired exposure fractions for each side (0–10 % of equity)
    long_frac: float = float(_CONFIG.get("LONG_FRAC", 0.05))
    short_frac: float = float(_CONFIG.get("SHORT_FRAC", 0.05))

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
    use_vortex: bool = False
    vortex_period: int = 14
    use_cmf: bool = False
    cmf_period: int = 20
    use_ema: bool = True
    ema_period: int = 20
    use_donchian: bool = False
    donchian_period: int = 20
    use_kijun: bool = False
    kijun_period: int = 26
    use_tenkan: bool = False
    tenkan_period: int = 9
    use_displacement: bool = False
    displacement: int = 26
    use_sentiment: bool = True
    use_macro: bool = True
    use_rvol: bool = True

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
            "use_ema": "USE_EMA",
            "ema_period": "EMA_PERIOD",
            "use_donchian": "USE_DONCHIAN",
            "donchian_period": "DONCHIAN_PERIOD",
            "use_kijun": "USE_KIJUN",
            "kijun_period": "KIJUN_PERIOD",
            "use_tenkan": "USE_TENKAN",
            "tenkan_period": "TENKAN_PERIOD",
            "use_displacement": "USE_DISPLACEMENT",
            "displacement": "DISPLACEMENT",
            "use_sentiment": "USE_SENTIMENT",
            "use_macro": "USE_MACRO",
            "use_rvol": "USE_RVOL",
        }
        for attr, key in mapping.items():
            if key in _CONFIG:
                cur = getattr(self, attr)
                typ = type(cur)
                try:
                    setattr(self, attr, typ(_CONFIG[key]))
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Risk control
# ---------------------------------------------------------------------------
RISK_FILTER = {
    "MIN_REWARD": -1.0,
    "MAX_DRAWDOWN": -0.90,
}

# floor and ceiling for optimiser learning-rate
LR_MIN = 1e-5
LR_MAX = 5e-4
# largest change allowed in a single mutate call (±20 %)
LR_FN_MAX_DELTA = 0.2

# Number of mini-batches for warm-up period
WARMUP_STEPS = 1000

# Allowed actions for the meta agent once indicator toggles are disabled.
# Keeping this list in ``hyperparams`` lets other modules share the frozen action
# space without importing :mod:`artibot.rl` during startup.
ALLOWED_META_ACTIONS = {
    "lr",
    "wd",
    "d_sma_period",
    "d_rsi_period",
    "d_macd_fast",
    "d_macd_slow",
    "d_macd_signal",
    "d_atr_period",
    "d_vortex_period",
    "d_cmf_period",
    "d_ema_period",
    "d_donchian_period",
    "d_kijun_period",
    "d_tenkan_period",
    "d_displacement",
    "d_sl",
    "d_tp",
    "d_long_frac",
    "d_short_frac",
    "d_lr",
    "d_wd",
    # allow indicator toggles so the meta agent can explore all combinations
    "toggle_sma",
    "toggle_rsi",
    "toggle_macd",
    "toggle_atr",
    "toggle_vortex",
    "toggle_cmf",
    "toggle_ichimoku",
    "toggle_ema",
    "toggle_donchian",
    "toggle_kijun",
    "toggle_tenkan",
    "toggle_disp",
}

# mapping from toggle actions to mask indices
TOGGLE_INDEX = {
    "toggle_sma": 0,
    "toggle_rsi": 1,
    "toggle_macd": 2,
    "toggle_atr": 3,
    "toggle_vortex": 4,
    "toggle_cmf": 5,
    "toggle_ichimoku": 6,
    "toggle_ema": 7,
    "toggle_donchian": 8,
    "toggle_kijun": 9,
    "toggle_tenkan": 10,
    "toggle_disp": 11,
}


def mutate_lr(old: float, delta: float) -> float:
    """Return ``old`` adjusted by ``delta`` within safe bounds."""

    if delta > 0.2:
        return 5e-4
    if delta < -0.2:
        return 1e-5
    new = old * (1 + delta)
    return max(1e-5, min(5e-4, new))


def should_freeze_features(step: int) -> bool:
    """Return ``True`` when indicator features should stay fixed."""

    return step >= WARMUP_STEPS
