"""Regime-aware caching utilities for Artibot.

Automatically stores and retrieves the most successful strategy per regime.
Regime detection is unsupervised (via HMM/KMeans) and caching occurs when
performance is positive.
"""

from __future__ import annotations

import os
import logging
import torch

from artibot.hyperparams import IndicatorHyperparams

# Directory for cached regime models
CACHE_DIR = "regime_model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def save_best_for_regime(regime: int, ensemble, result: dict) -> None:
    """Persist the ensemble state for ``regime`` when performance improves."""

    if result.get("net_pct", 0.0) <= 0:
        return

    regime_dir = os.path.join(CACHE_DIR, f"cluster_{regime}")
    os.makedirs(regime_dir, exist_ok=True)
    filepath = os.path.join(regime_dir, "best.pt")

    prev_best = float("-inf")
    if os.path.isfile(filepath):
        try:
            ckpt = torch.load(filepath)
            prev_best = ckpt.get("best_composite_reward", float("-inf"))
        except Exception:
            prev_best = float("-inf")

    new_reward = result.get("composite_reward", 0.0)
    if new_reward > prev_best:
        state_dicts = [m.state_dict() for m in ensemble.models]
        ihp = ensemble.indicator_hparams
        torch.save(
            {
                "best_composite_reward": new_reward,
                "sharpe": result.get("sharpe", 0.0),
                "net_pct": result.get("net_pct", 0.0),
                "state_dicts": state_dicts,
                "indicator_hparams": ihp and ihp.__dict__,
            },
            filepath,
        )
        logging.info(
            "CACHED_STRATEGY regime=%s reward=%.2f net_pct=%.2f",
            regime,
            new_reward,
            result.get("net_pct", 0.0),
        )


def load_best_for_regime(regime: int, ensemble):
    """Load cached weights for ``regime`` into ``ensemble`` if available."""

    filepath = os.path.join(CACHE_DIR, f"cluster_{regime}", "best.pt")
    if not os.path.isfile(filepath):
        return None

    ckpt = torch.load(filepath, map_location=ensemble.device)
    state_dicts = ckpt["state_dicts"]
    ihp = ckpt.get("indicator_hparams")
    if ihp:
        try:
            ensemble.indicator_hparams = IndicatorHyperparams(**ihp)
        except Exception:
            pass

    if regime < len(state_dicts) and regime < len(ensemble.models):
        ensemble.models[regime].load_state_dict(state_dicts[regime], strict=False)
    else:
        for model, sd in zip(ensemble.models, state_dicts):
            model.load_state_dict(sd, strict=False)

    return {
        "best_composite_reward": ckpt.get("best_composite_reward", 0.0),
        "sharpe": ckpt.get("sharpe", 0.0),
        "net_pct": ckpt.get("net_pct", 0.0),
    }
