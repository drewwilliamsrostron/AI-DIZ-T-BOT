"""Unified hyper-parameter optimisation using Optuna BOHB."""

from __future__ import annotations

from dataclasses import fields
from typing import Dict, Tuple, get_args, get_origin
import logging
import os

import torch

import optuna

try:  # pragma: no cover - optional dependency during tests
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner
except Exception:  # pragma: no cover - stubbed optuna
    TPESampler = HyperbandPruner = None

from .hyperparams import IndicatorHyperparams
from .ensemble import EnsembleModel
from .backtest import robust_backtest
from .dataset import load_csv_hourly, HourlyDataset


_DEF_LR = 1e-3
_DEF_WD = 0.0


def _trial_indicator_params(trial: optuna.trial.Trial) -> IndicatorHyperparams:
    """Sample indicator periods and toggles for ``trial``."""
    params = {}
    for f in fields(IndicatorHyperparams):
        if f.name.startswith("use_"):
            params[f.name] = trial.suggest_categorical(f.name, [True, False])
        else:
            ftype = f.type
            origin = get_origin(ftype)
            if origin is None:
                if ftype is int:
                    params[f.name] = trial.suggest_int(f.name, 1, 200)
            else:
                if int in get_args(ftype):
                    params[f.name] = trial.suggest_int(f.name, 1, 200)
    return IndicatorHyperparams(**params)


_CSV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Gemini_BTCUSD_1h.csv")
)


def _quick_backtest(
    hp: IndicatorHyperparams, bars: int = 90
) -> tuple[float, float, float]:
    """Run a light-weight backtest and return reward metrics."""

    data = load_csv_hourly(_CSV_PATH)
    if len(data) > bars:
        data = data[-bars:]
    ds = HourlyDataset(data, seq_len=24, indicator_hparams=hp, train_mode=False)
    n_features = ds[0][0].shape[-1]
    model = EnsembleModel(
        device=torch.device("cpu"),
        n_models=1,
        n_features=n_features,
    )
    model.indicator_hparams = hp
    result = robust_backtest(model, data, indicator_hp=hp)
    return (
        result.get("composite_reward", 0.0),
        result.get("net_pct", 0.0),
        result.get("sharpe", 0.0),
    )


def _objective(trial: optuna.trial.Trial) -> float:
    """Evaluate indicator parameters using a short backtest."""

    hp = _trial_indicator_params(trial)
    trial_lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    trial_wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    comp_reward, net_pct, sharpe = _quick_backtest(hp, bars=90)
    score = comp_reward if comp_reward > 0 else -1.0
    logging.info(
        "Optuna trial: hp=%s, net_pct=%.2f, sharpe=%.2f, comp_reward=%.3f, returned=%.3f",
        hp,
        net_pct,
        sharpe,
        comp_reward,
        score,
    )
    trial.set_user_attr("indicator_hp", hp)
    trial.set_user_attr("learning_rate", trial_lr)
    trial.set_user_attr("weight_decay", trial_wd)
    return score


def run_bohb(n_trials: int = 50) -> Tuple[IndicatorHyperparams, Dict[str, float]]:
    """Run BOHB search and return best parameters."""
    if TPESampler is None or HyperbandPruner is None:
        study = optuna.create_study(direction="maximize")
        study.enqueue_trial({})
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(multivariate=True),
            pruner=HyperbandPruner(),
        )
        study.optimize(_objective, n_trials=n_trials)
    trial = study.best_trial
    attrs = getattr(trial, "user_attrs", {})
    hp = attrs.get("indicator_hp", IndicatorHyperparams())
    params = {
        "learning_rate": float(attrs.get("learning_rate", _DEF_LR)),
        "weight_decay": float(attrs.get("weight_decay", _DEF_WD)),
    }
    return hp, params
