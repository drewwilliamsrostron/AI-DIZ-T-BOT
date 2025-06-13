"""Hyperparameter configuration loaded from ``master_config.json``."""

from __future__ import annotations

from dataclasses import dataclass

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
    atr_period: int = int(_CONFIG.get("ATR_PERIOD", 50))
    atr_threshold_k: float = float(_CONFIG.get("ATR_THRESHOLD_K", 1.5))
    conf_threshold: float = float(_CONFIG.get("CONF_THRESHOLD", 5e-5))

    use_sma: bool = bool(_CONFIG.get("USE_SMA", True))
    use_vortex: bool = bool(_CONFIG.get("USE_VORTEX", False))
    use_cmf: bool = bool(_CONFIG.get("USE_CMF", False))
    use_ichimoku: bool = bool(_CONFIG.get("USE_ICHIMOKU", False))
    use_atr: bool = bool(_CONFIG.get("USE_ATR", False))
    use_momentum: bool = bool(_CONFIG.get("USE_MOMENTUM", False))
    use_bbw: bool = bool(_CONFIG.get("USE_BBW", False))

