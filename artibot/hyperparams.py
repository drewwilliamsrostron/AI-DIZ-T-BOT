"""Hyperparameter configuration loaded from ``master_config.json``."""

from __future__ import annotations

from dataclasses import dataclass, fields
import logging

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
        # propagate top-level ``use_*`` flags to the indicator dataclass so that
        # global defaults reflect the desired startup state
        for f in fields(self.indicator_hp):
            if f.name.startswith("use_") and hasattr(self, f.name):
                setattr(self.indicator_hp, f.name, getattr(self, f.name))
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
    sma_period: int = 1
    use_rsi: bool = True
    rsi_period: int = 1
    use_macd: bool = True
    macd_fast: int = 1
    macd_slow: int = 1
    macd_signal: int = 1
    use_atr: bool = True
    atr_period: int = 1
    use_vortex: bool = True
    vortex_period: int = 1
    use_cmf: bool = True
    cmf_period: int = 1
    use_ema: bool = True
    ema_period: int = 1
    use_donchian: bool = True
    donchian_period: int = 1
    use_kijun: bool = True
    kijun_period: int = 1
    use_tenkan: bool = True
    tenkan_period: int = 1
    use_displacement: bool = True
    displacement: int = 1
    use_sentiment: bool = True
    use_macro: bool = True
    use_rvol: bool = True

    def __post_init__(self) -> None:
        """Override defaults with any ``USE_*`` or period values in the config."""
        mapping = {
            "sma_period": "SMA_PERIOD",
            "rsi_period": "RSI_PERIOD",
            "macd_fast": "MACD_FAST",
            "macd_slow": "MACD_SLOW",
            "macd_signal": "MACD_SIGNAL",
            "atr_period": "ATR_PERIOD",
            "vortex_period": "VORTEX_PERIOD",
            "cmf_period": "CMF_PERIOD",
            "ema_period": "EMA_PERIOD",
            "donchian_period": "DONCHIAN_PERIOD",
            "kijun_period": "KIJUN_PERIOD",
            "tenkan_period": "TENKAN_PERIOD",
            "displacement": "DISPLACEMENT",
        }

        # automatically include every ``use_*`` flag for convenience
        for field in fields(self):
            if field.name.startswith("use_"):
                mapping[field.name] = field.name.upper()

        # integer overrides when config value available and attribute matches default
        for attr, key in mapping.items():
            if attr.startswith("use_"):
                continue
            if (
                key in _CONFIG
                and getattr(self, attr) == self.__dataclass_fields__[attr].default
            ):
                try:
                    val = int(_CONFIG[key])
                    setattr(self, attr, max(1, min(val, 200)))
                except Exception:
                    pass

        # ``use_*`` flags default to config values when provided and attribute matches default
        for f in fields(self):
            if f.name.startswith("use_") and f.name.upper() in _CONFIG:
                if getattr(self, f.name) == self.__dataclass_fields__[f.name].default:
                    try:
                        setattr(self, f.name, bool(_CONFIG[f.name.upper()]))
                    except Exception:
                        pass

        logging.info(
            "Indicator hyperparams: %s",
            {f.name: getattr(self, f.name) for f in fields(self)},
        )


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
WARMUP_STEPS = int(_CONFIG.get("WARMUP_STEPS", 50))

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

# allow indicator toggles through the meta action filter
ALLOWED_META_ACTIONS.update(TOGGLE_INDEX.keys())


def mutate_lr(old: float, delta: float) -> float:
    """Return ``old`` adjusted by ``delta`` within safe bounds."""

    if delta > 0.2:
        return 5e-4
    if delta < -0.2:
        return 1e-5
    new = old * (1 + delta)
    return max(1e-5, min(5e-4, new))


def should_freeze_features(step: int) -> bool:
    """Return ``True`` when indicator features should stay fixed.

    Warm-up gating has been removed so this now always returns ``False``.
    """

    return False
