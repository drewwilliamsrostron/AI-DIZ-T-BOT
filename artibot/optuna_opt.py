"""Unified hyper-parameter optimisation using Optuna BOHB."""

from __future__ import annotations

from dataclasses import fields
from typing import Dict, Tuple
import random

import optuna

try:  # pragma: no cover - optional dependency during tests
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner
except Exception:  # pragma: no cover - stubbed optuna
    TPESampler = HyperbandPruner = None

from .hyperparams import IndicatorHyperparams


_DEF_LR = 1e-3
_DEF_WD = 0.0


def _trial_indicator_params(trial: optuna.trial.Trial) -> IndicatorHyperparams:
    """Sample indicator periods and toggles for ``trial``."""
    params = {}
    for f in fields(IndicatorHyperparams):
        if f.name.startswith("use_"):
            params[f.name] = trial.suggest_categorical(f.name, [True, False])
        elif f.type is int:
            params[f.name] = trial.suggest_int(f.name, 1, 200)
    return IndicatorHyperparams(**params)


def _objective(trial: optuna.trial.Trial) -> float:
    """Dummy objective reporting two steps for pruning."""
    hp = _trial_indicator_params(trial)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    # quick evaluation placeholder
    score = random.random()
    trial.report(score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    score += random.random() * 0.1
    trial.report(score, step=1)
    trial.set_user_attr("indicator_hp", hp)
    trial.set_user_attr("learning_rate", lr)
    trial.set_user_attr("weight_decay", wd)
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
