"""Unified Optuna-based hyperparameter optimiser."""

from __future__ import annotations

import logging
import threading
from dataclasses import fields
from typing import Iterable, Tuple

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from .ensemble import EnsembleModel
from .training import csv_training_thread
from .backtest import robust_backtest
from .hyperparams import IndicatorHyperparams
import artibot.globals as G

_LOG = logging.getLogger(__name__)


def _sample_indicator_params(trial: optuna.trial.Trial) -> IndicatorHyperparams:
    """Return ``IndicatorHyperparams`` sampled uniformly in [1, 200]."""
    params = {}
    for f in fields(IndicatorHyperparams):
        if f.name.startswith("use_"):
            params[f.name] = trial.suggest_categorical(f.name, [True, False])
        else:
            params[f.name] = trial.suggest_int(f.name, 1, 200)
    return IndicatorHyperparams(**params)


def _train_and_score(model: EnsembleModel, data: list, epochs: int) -> float:
    stop = threading.Event()
    csv_training_thread(
        model,
        data,
        stop,
        {"ADAPT_TO_LIVE": False},
        use_prev_weights=False,
        max_epochs=epochs,
        update_globals=False,
    )
    metrics = robust_backtest(model, data)
    if metrics.get("trades", 0) == 0:
        _LOG.info("IGNORED_EMPTY_BACKTEST: 0 trades in result")
    else:
        G.push_backtest_metrics(metrics)
    return float(metrics.get("composite_reward", 0.0))


def objective(trial: optuna.trial.Trial, data: list) -> float:
    """Objective used by Optuna search."""
    ind_hp = _sample_indicator_params(trial)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 0.0, 1e-3, log=True)

    model = EnsembleModel(lr=lr, weight_decay=wd, n_models=1)
    model.indicator_hparams = ind_hp

    quick_score = _train_and_score(model, data[:200], epochs=1)
    trial.report(quick_score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    final_score = _train_and_score(model, data, epochs=3)
    trial.report(final_score, step=1)
    return final_score


def run_search(data: Iterable, n_trials: int = 20, storage: str | None = None) -> Tuple[IndicatorHyperparams, float, float]:
    """Run BOHB search and return the best parameters."""
    dataset = list(data)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(multivariate=True),
        pruner=HyperbandPruner(),
        storage=storage,
    )
    study.optimize(lambda t: objective(t, dataset), n_trials=n_trials)
    best = study.best_trial.params
    hp_keys = {f.name for f in fields(IndicatorHyperparams)}
    hp_kwargs = {k: best[k] for k in best if k in hp_keys}
    hp = IndicatorHyperparams(**hp_kwargs)
    lr = best.get("lr", 1e-4)
    wd = best.get("weight_decay", 0.0)
    return hp, lr, wd
