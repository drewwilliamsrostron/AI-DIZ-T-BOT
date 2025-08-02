# Ensemble model handling and training utilities.
# Includes optimisation logic and RL loss composition.
"""Model ensemble used during training and prediction."""

# ruff: noqa: F403, F405

import inspect
from contextlib import nullcontext
from typing import Iterable, Optional, Tuple
from threading import Event
import threading
import logging
import random
import hashlib
from dataclasses import asdict

import os
import shutil
import numpy as np
import sys
import importlib.machinery as _machinery

if "openai" in sys.modules and getattr(sys.modules["openai"], "__spec__", None) is None:
    sys.modules["openai"].__spec__ = _machinery.ModuleSpec("openai", None)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    from torch.amp import GradScaler, autocast  # PyTorch >=2.3
except Exception:  # pragma: no cover - older versions
    from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    OneCycleLR,
)
from torch.utils.data import DataLoader

from .backtest import robust_backtest
from .hyperparams import HyperParams, IndicatorHyperparams
from artibot.core.device import get_device
from .utils import zero_disabled
import artibot.globals as G
from .metrics import compute_yearly_stats, compute_monthly_stats
from .model import TradingModel
from .dataset import TradeParams
from .constants import FEATURE_DIMENSION
from .feature_manager import validate_and_align_features


def update_best(epoch: int, reward: float, net_pct: float, best_ckpt_path: str) -> None:
    """Log a ``NEW_BEST_CANDIDATE`` event.

    Parameters
    ----------
    epoch:
        Current epoch number.
    reward:
        Composite reward achieved this epoch.
    net_pct:
        Net profit percentage from the latest backtest.
    best_ckpt_path:
        File path where the best weights were saved.

    Returns
    -------
    None
    """

    logging.info(
        "NEW_BEST_CANDIDATE  epoch=%d  reward=%.3f  net_pct=%.2f  saved %s",
        epoch,
        reward,
        net_pct,
        best_ckpt_path,
    )


def reject_if_risky(
    reward: float,
    max_dd: float,
    entropy: float,
    *,
    thresholds: dict | None = None,
) -> bool:
    """Check reward metrics against the risk filter.

    Parameters
    ----------
    reward:
        Composite reward value to evaluate.
    max_dd:
        Maximum drawdown from the latest backtest.
    entropy:
        Mean attention entropy for the epoch.
    thresholds:
        Optional dictionary with ``MIN_REWARD`` and ``MAX_DRAWDOWN`` limits.

    Returns
    -------
    bool
        ``True`` when metrics breach the configured risk limits.
    """

    if not G.is_risk_filter_enabled():
        return False

    if thresholds is None:
        try:  # lazy import avoids circular dependency
            from .bot_app import CONFIG

            thresholds = CONFIG.get("RISK_FILTER", CONFIG)
        except Exception:
            thresholds = {}

    min_entropy = float(thresholds.get("MIN_ENTROPY", 1.0))
    min_reward = float(thresholds.get("MIN_REWARD", -1.0))
    max_drawdown = float(thresholds.get("MAX_DRAWDOWN", -0.30))

    # Early-stage models get a looser gate until trade count builds up
    if G.global_num_trades < 1000:
        return reward <= 0.0 and max_dd <= -0.40

    if entropy < min_entropy:
        return True  # reject collapsed runs
    return max_dd < max_drawdown or reward < min_reward


def nuclear_key_gate(
    reward: float,
    max_dd: float,
    entropy: float,
    profit_factor: float,
    *,
    thresholds: dict | None = None,
) -> bool:
    """Evaluate whether trading should be enabled.

    Parameters
    ----------
    reward:
        Composite reward from the current epoch.
    max_dd:
        Maximum drawdown value.
    entropy:
        Attention entropy from the models.
    profit_factor:
        Profit factor computed from backtests.
    thresholds:
        Optional overrides for the NK gate limits.

    Returns
    -------
    bool
        ``True`` when trading is allowed based on the nuclear key state.
    """

    if thresholds is None:
        try:  # lazy import avoids circular dependency
            from .bot_app import CONFIG

            thresholds = CONFIG.get("RISK_FILTER", CONFIG)
        except Exception:
            thresholds = {}

    min_entropy = float(thresholds.get("MIN_ENTROPY", 1.0))
    min_reward = float(thresholds.get("MIN_REWARD", -1.0))
    max_drawdown = float(thresholds.get("MAX_DRAWDOWN", -0.30))
    min_profit_factor = 1.5

    if not G.is_nuclear_key_enabled():
        return True

    return (
        entropy >= min_entropy
        and reward >= min_reward
        and max_dd >= max_drawdown
        and profit_factor >= min_profit_factor
    )


def nk_gate_passes() -> bool:
    """Shortcut helper using global metrics.

    Returns
    -------
    bool
        ``True`` when the current metrics pass the NK gate.
    """
    entropy = (
        float(G.global_attention_entropy_history[-1])
        if G.global_attention_entropy_history
        else 0.0
    )
    return nuclear_key_gate(
        G.global_composite_reward,
        G.global_max_drawdown,
        entropy,
        G.global_profit_factor,
    )


def choose_best(rewards: list[float]) -> float:
    """Return the highest reward from ``rewards``.

    Raises ``ValueError`` when the list is empty.
    """
    if not rewards:
        raise ValueError("rewards cannot be empty")
    return max(rewards)


class EnsembleModel(nn.Module):
    """Simple container for multiple models and optimisers.

    Parameters
    ----------
    n_features:
        Number of columns expected in the feature matrix.  When ``None`` the
        value from :data:`~artibot.constants.FEATURE_DIMENSION` is used.
    indicator_hp:
        Optional :class:`IndicatorHyperparams` to initialise the ensemble with
        tuned indicator settings.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        n_models: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        weights_path: str = "best.pt",
        *,
        n_features: int | None = None,
        total_steps: int = 10000,
        grad_accum_steps: int = 1,
        delayed_reward_epochs: int = 0,
        warmup_steps: int | None = None,
        indicator_hp: IndicatorHyperparams | None = None,
        freeze_features: bool | None = None,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        device = torch.device(device) if device is not None else get_device()
        self.device = device
        self.weights_path = weights_path
        self.tau = float(tau)
        # use provided settings or fall back to config defaults
        self._indicator_hparams = (
            indicator_hp if indicator_hp is not None else IndicatorHyperparams()
        )
        self.hp = HyperParams(indicator_hp=self._indicator_hparams)
        if freeze_features is not None:
            self.hp.freeze_features = freeze_features
        if lr is not None:
            self.hp.learning_rate = lr

        # Determine the feature dimension either from the caller or fall back
        # to the package constant.  This allows the ensemble to align its input
        # layer with the dataset's feature matrix and prevents shape mismatches
        # during training.
        dim = n_features if n_features is not None else FEATURE_DIMENSION

        self.expected_features = dim
        self.n_features = dim

        self.models = [
            TradingModel(input_size=dim).to(device, non_blocking=True)
            for _ in range(n_models)
        ]
        for model in self.models:
            # track whether any parameters are frozen
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    logging.warning("Frozen param: %s", name)
            model.score_history = []
            model.sharpe_ema = 0.0

        if len(self.models) == 3:
            names = ["trend_following", "mean_reversion", "volatility_adaptive"]
            for i, model in enumerate(self.models):
                model.strategy_name = names[i]
        else:
            for i, model in enumerate(self.models):
                model.strategy_name = f"regime_{i}"

        if len(self.models) == 3:
            from copy import deepcopy

            base_hp = self._indicator_hparams

            hp0 = deepcopy(base_hp)
            hp0.use_momentum = True
            hp0.use_rsi = False
            hp0.sma_period = max(hp0.sma_period, 50)

            hp1 = deepcopy(base_hp)
            hp1.use_rsi = True
            hp1.rsi_period = 14
            hp1.use_momentum = False

            hp2 = deepcopy(base_hp)
            hp2.use_atr = True
            hp2.atr_period = min(hp2.atr_period, 14)
            hp2.use_bbw = True

            specialized_hps = [hp0, hp1, hp2]
            for idx, m in enumerate(self.models[:3]):
                m.indicator_hparams = specialized_hps[idx]

        logging.info(
            "Ensemble initialized with %d models: %s",
            len(self.models),
            ", ".join(
                getattr(m, "strategy_name", str(idx))
                for idx, m in enumerate(self.models)
            ),
        )
        self._mask_lock = threading.Lock()
        self.register_buffer("feature_mask", torch.ones(1, dim, device=device))
        print("[DEBUG] Model moved to device")
        from . import hyperparams as _hp

        lr = max(self.hp.learning_rate, _hp.LR_MIN)
        self.optimizers = [
            optim.AdamW(
                m.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.9999),
            )
            for m in self.models
        ]
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([2.0, 2.0, 1.0], device=device)
        )
        self.mse_loss_fn = nn.MSELoss()
        amp_on = device.type == "cuda"
        print(f"[DEBUG] Using device: {device}, AMP enabled: {amp_on}")
        # GradScaler's `device` arg is not supported on older PyTorch versions
        if "device" in inspect.signature(GradScaler).parameters:
            self.scaler = GradScaler(enabled=amp_on, device=device)
        else:
            self.scaler = GradScaler(enabled=amp_on)
        print("[DEBUG] Scaler init complete")
        self.best_val_loss = float("inf")
        self.best_composite_reward = float("-inf")
        self.best_state_dicts = None
        # Persist step count across folds so RL stays active
        self.train_steps = G.global_step
        # Start with a small reward weight and no delay
        self.reward_loss_weight = 0.05
        self.max_reward_loss_weight = 0.2
        self.reward_running_mean = 0.0
        self.reward_running_var = 1.0
        self.patience = 0
        self.delayed_reward_epochs = delayed_reward_epochs
        self.warmup_steps = warmup_steps if warmup_steps is not None else G.warmup_steps

        # per-epoch attention stats
        self.entropies: list[float] = []
        self.max_probs: list[float] = []
        self.rejection_count_this_epoch = 0

        # (3) We'll do dynamic patience mechanism, so this is an initial
        self.patience_counter = 0

        # Disable LR schedulers to avoid the step-order warning from PyTorch
        self.schedulers = []
        self.cycle = []

        self.grad_accum_steps = grad_accum_steps
        self.total_steps = total_steps

    # ------------------------------------------------------------------
    # indicator hyper-parameter accessors
    # ------------------------------------------------------------------

    @property
    def indicator_hparams(self) -> IndicatorHyperparams:
        """Return indicator settings used by the ensemble."""

        return self._indicator_hparams

    @indicator_hparams.setter
    def indicator_hparams(self, hp: IndicatorHyperparams) -> None:
        """Update indicator settings and propagate to :class:`HyperParams`."""

        self._indicator_hparams = hp
        self.hp.indicator_hp = hp

    def _align_features(self, x: torch.Tensor) -> torch.Tensor:
        """Validate feature dimension and zero disabled columns."""

        current_features = x.shape[-1]
        if current_features != self.expected_features:
            raise ValueError(
                f"Expected {self.expected_features} features, got {current_features}"
            )

        mask = self.feature_mask.to(x.device)
        return zero_disabled(x, mask)

    @validate_and_align_features
    def forward(self, x: torch.Tensor):
        """Return weighted ensemble output."""
        x = x.to(self.device)
        outs = [m(x) for m in self.models]
        scores = torch.tensor(
            [m.score_history[-1] if m.score_history else 0.0 for m in self.models],
            dtype=torch.float32,
            device=self.device,
        )
        weights = torch.softmax(scores / self.tau, dim=0)
        logits = torch.stack([o[0] for o in outs])
        w_logits = (weights.view(-1, 1, 1) * logits).sum(dim=0)
        params = [o[1] for o in outs]

        def _get_attr(p, name, default):
            val = getattr(p, name, None)
            if val is None:
                return torch.tensor(default, device=self.device)
            if not torch.is_tensor(val):
                return torch.tensor(val, device=self.device)
            return val

        risk_frac = sum(
            w * _get_attr(p, "risk_fraction", 0.1) for w, p in zip(weights, params)
        )
        sl_mult = sum(
            w * _get_attr(p, "sl_multiplier", 5.0) for w, p in zip(weights, params)
        )
        tp_mult = sum(
            w * _get_attr(p, "tp_multiplier", 5.0) for w, p in zip(weights, params)
        )
        att = sum(
            w * _get_attr(p, "attention", torch.zeros(1))
            for w, p in zip(weights, params)
        )
        reward_pred = torch.stack([o[2] for o in outs])
        pred_reward = (weights.view(-1, 1) * reward_pred).sum(dim=0)
        logging.info("MODEL_WEIGHTS %s", [round(float(w), 3) for w in weights.cpu()])
        return w_logits, TradeParams(risk_frac, sl_mult, tp_mult, att), pred_reward

    def configure_one_cycle(self, total_steps: int) -> None:
        """Initialise OneCycle schedulers with ``total_steps``."""
        self.total_steps = total_steps
        # Scheduler disabled – maintain attribute for callers
        self.cycle = []

    def rebuild_models(self, new_dim: int) -> None:
        """Validate that the requested dimension matches the frozen setting."""
        if new_dim != self.n_features:
            raise ValueError("Feature dimension mismatch")

    def train_one_epoch(
        self,
        dl_train: DataLoader,
        dl_val: Optional[DataLoader],
        data_full: Iterable,
        stop_event: Optional[Event] = None,
        *,
        features: Optional[dict] = None,
        update_globals: bool = True,
    ) -> Tuple[float, Optional[float]]:
        """Train models for one epoch and return losses.

        Parameters
        ----------
        dl_train:
            DataLoader providing training batches.
        dl_val:
            Optional validation DataLoader used to compute ``val_loss``.
        data_full:
            Full dataset for backtesting before training.
        stop_event:
            Optional threading event signalling early stop.
        features:
            Optional precomputed indicator arrays for ``data_full``.
        update_globals:
            When ``True`` update :mod:`artibot.globals` with metrics from this
            epoch.

        Returns
        -------
        tuple[float, Optional[float]]
            ``train_loss`` and ``val_loss`` (``None`` when no ``dl_val``).
        """
        first_epoch = self.train_steps == 0
        target_wd = 0.0 if first_epoch else self.hp.weight_decay
        for opt in self.optimizers:
            for pg in opt.param_groups:
                pg["weight_decay"] = target_wd
        if first_epoch:
            logging.info("Epoch 0: temporarily disabled weight decay")

        # Mutate shared state on the globals module so the GUI sees progress

        # ------------------------------------------------------------------
        # Optional indicator sweep to tune hyperparameters at runtime
        # ------------------------------------------------------------------
        sweep_every = int(os.environ.get("SWEEP_EVERY", "1"))
        run_sweep = sweep_every > 0 and self.train_steps % sweep_every == 1
        best_result: dict | None = None
        if run_sweep:
            from .optuna_opt import run_bohb
            from .hyperparams import _CONFIG

            hp, _ = run_bohb(n_trials=10)
            for name, val in vars(hp).items():
                setattr(self.hp.indicator_hp, name, val)
                setattr(self.indicator_hparams, name, val)
                _CONFIG[name.upper()] = val
            G.sync_globals(self.hp, self.hp.indicator_hp)
            best_result = robust_backtest(
                self, data_full, indicator_hp=self.indicator_hparams
            )

        # Run a back-test with the best parameters found (or current settings)
        logging.info(">>> ENTERING DEFCON 3: Full Backtest")
        logging.info(">>> Using current best hyperparams")
        current_result = best_result or robust_backtest(
            self, data_full, indicator_hp=self.indicator_hparams
        )
        # Evaluate each model individually for adaptive weighting
        for idx_m, m in enumerate(self.models):
            orig = self.models
            self.models = [m]
            try:
                res_m = robust_backtest(
                    self, data_full, indicator_hp=self.indicator_hparams
                )
            finally:
                self.models = orig
            m.score_history.append(res_m.get("composite_reward", 0.0))
            m.sharpe_ema = 0.9 * getattr(m, "sharpe_ema", 0.0) + 0.1 * res_m.get(
                "sharpe", 0.0
            )
            if hasattr(G, "current_regime"):
                r = G.current_regime
                if r is not None:
                    if not hasattr(m, "sharpe_by_regime"):
                        m.sharpe_by_regime = {}
                        m.reward_by_regime = {}
                    prev_sharpe = m.sharpe_by_regime.get(r, 0.0)
                    prev_reward = m.reward_by_regime.get(r, 0.0)
                    m.sharpe_by_regime[r] = 0.9 * prev_sharpe + 0.1 * res_m.get(
                        "sharpe", 0.0
                    )
                    m.reward_by_regime[r] = 0.9 * prev_reward + 0.1 * res_m.get(
                        "composite_reward", 0.0
                    )
        span_days = 0
        if data_full:
            try:
                start_ts = int(data_full[0][0])
                end_ts = int(data_full[-1][0])
                span_days = (end_ts - start_ts) // 86400
            except Exception:
                span_days = 0

        ignore_result = current_result.get("trades", 0) == 0
        if ignore_result:
            logging.info("IGNORED_EMPTY_BACKTEST: 0 trades in result")

        # --- ❹  Push to globals & ping GUI ------------------------------------------
        if not ignore_result:
            G.push_backtest_metrics(current_result)
            regime = getattr(G, "current_regime", None)
            if regime is not None:
                try:
                    from artibot import regime_cache

                    regime_cache.save_best_for_regime(regime, self, current_result)
                except Exception as e:  # pragma: no cover - cache failures non-critical
                    logging.error(f"Regime caching failed: {e}")
        # ---------------- END merged block ----------------

        if data_full:
            assert len(data_full[0]) >= 5, "Expect raw OHLCV rows"

        if update_globals and not ignore_result:
            G.global_equity_curve = current_result["equity_curve"]
            G.global_backtest_profit.append(current_result["effective_net_pct"])
            G.global_inactivity_penalty = current_result["inactivity_penalty"]
            G.global_composite_reward = current_result["composite_reward"]
            G.global_days_without_trading = current_result["days_without_trading"]
            G.global_trade_details = current_result["trade_details"]
            G.global_days_in_profit = current_result["days_in_profit"]
            G.global_sharpe = current_result["sharpe"]
            G.global_max_drawdown = current_result["max_drawdown"]
            G.global_net_pct = current_result["net_pct"]
            G.global_num_trades = current_result["trades"]

            G.global_win_rate = current_result["win_rate"]
            G.global_profit_factor = current_result["profit_factor"]
            G.global_avg_trade_duration = current_result["avg_trade_duration"]
            G.global_avg_win = current_result.get("avg_win", 0.0)
            G.global_avg_loss = current_result.get("avg_loss", 0.0)

        dfy, table_str = compute_yearly_stats(
            current_result["equity_curve"],
            current_result["trade_details"],
            initial_balance=100.0,
        )
        rows = len(dfy) if dfy is not None else 0
        logging.info("YEARLY_STATS rows=%d", rows)
        if update_globals:
            G.global_yearly_stats_table = table_str

        dfm, monthly_table = compute_monthly_stats(
            current_result["equity_curve"],
            current_result["trade_details"],
            initial_balance=100.0,
        )
        rows_m = len(dfm) if dfm is not None else 0
        logging.info("MONTHLY_STATS rows=%d", rows_m)
        if update_globals:
            G.global_monthly_stats_table = monthly_table

        # update live weights when the composite reward improves
        # ``global_best_composite_reward`` defaults to ``-inf`` so a separate
        # ``None`` check is unnecessary
        best = G.global_best_composite_reward
        if (
            current_result.get("full_data_run", False)
            and not ignore_result
            and current_result["composite_reward"] > best
        ):
            logging.info(
                "\u2705 PROMOTED: full_data_run=True, span=%d days, reward=%.2f",
                span_days,
                current_result["composite_reward"],
            )
            G.global_best_composite_reward = current_result["composite_reward"]

            G.global_best_sharpe = current_result["sharpe"]
            G.global_best_equity_curve = current_result["equity_curve"]
            G.global_best_drawdown = current_result["max_drawdown"]
            G.global_best_net_pct = current_result["net_pct"]
            G.global_best_num_trades = current_result["trades"]
            G.global_best_win_rate = current_result["win_rate"]
            G.global_best_profit_factor = current_result["profit_factor"]
            G.global_best_avg_trade_duration = current_result["avg_trade_duration"]
            G.global_best_avg_win = current_result.get("avg_win", 0.0)
            G.global_best_avg_loss = current_result.get("avg_loss", 0.0)
            G.global_best_trade_details = current_result["trade_details"]
            G.global_best_inactivity_penalty = current_result["inactivity_penalty"]
            G.global_best_days_in_profit = current_result["days_in_profit"]
            G.global_best_lr = self.optimizers[0].param_groups[0]["lr"]
            G.global_best_wd = self.optimizers[0].param_groups[0].get("weight_decay", 0)
            G.global_best_yearly_stats_table = table_str
            G.global_best_monthly_stats_table = monthly_table
            self.best_state_dicts = [m.state_dict() for m in self.models]
            self.save_best_weights(self.weights_path)
            logging.info(
                "SAVED_BEST_WEIGHTS epoch=%d reward=%.2f path=%s",
                self.train_steps,
                current_result["composite_reward"],
                self.weights_path,
            )
            md5 = ""
            try:
                with open(self.weights_path, "rb") as f:
                    md5 = hashlib.md5(f.read()).hexdigest()
            except Exception:
                md5 = ""
            promote = G.use_sandbox or G.nuke_armed or nk_gate_passes()
            if promote:
                live_path = os.path.join(
                    os.path.dirname(self.weights_path), "live_model.pt"
                )
                try:
                    shutil.copy(self.weights_path, live_path)
                    G.set_live_weights_updated(True)
                    logging.info("PROMOTED_TO_LIVE_MODEL hash=%s", md5)
                    logging.info(
                        "PROMOTION: Model meets Nuclear Key criteria, ready to trade."
                    )
                except Exception as exc:
                    logging.error("Live weight copy failed: %s", exc)
            elif not G.use_sandbox:
                try:
                    from .bot_app import CONFIG

                    thresholds = CONFIG.get("RISK_FILTER", CONFIG)
                except Exception:
                    thresholds = {}

                min_entropy = float(thresholds.get("MIN_ENTROPY", 1.0))
                min_reward = float(thresholds.get("MIN_REWARD", -1.0))
                max_drawdown_lim = float(thresholds.get("MAX_DRAWDOWN", -0.30))
                min_profit_factor = 1.5

                entropy = (
                    float(G.global_attention_entropy_history[-1])
                    if G.global_attention_entropy_history
                    else 0.0
                )
                reward_val = G.global_composite_reward
                pf_value = G.global_profit_factor
                max_dd = G.global_max_drawdown

                reasons = []
                if pf_value < min_profit_factor:
                    reasons.append(
                        f"profit factor {pf_value:.2f} < {min_profit_factor:.2f}"
                    )
                if reward_val < min_reward:
                    reasons.append(
                        f"composite reward {reward_val:.1f} < threshold {min_reward:.1f}"
                    )
                if max_dd < max_drawdown_lim:
                    reasons.append(
                        f"max drawdown {abs(max_dd) * 100:.0f}% exceeds {abs(max_drawdown_lim) * 100:.0f}% limit"
                    )
                if entropy < min_entropy:
                    reasons.append(f"entropy {entropy:.2f} < {min_entropy:.2f}")
                reason_str = (
                    "; ".join(reasons) if reasons else "NK gate conditions not met"
                )
                logging.info("NOT_PROMOTED: %s", reason_str)
                short_reason = "criteria not met"
                if pf_value < min_profit_factor:
                    short_reason = "profit factor below 1.5"
                elif reward_val < min_reward:
                    short_reason = "reward below threshold"
                elif max_dd < max_drawdown_lim:
                    short_reason = "drawdown exceeds limit"
                elif entropy < min_entropy:
                    short_reason = "low entropy"
                G.set_status("Training", f"Live trading locked by NK: {short_reason}")

        else:
            logging.info(
                "\u274c SKIPPED: full_data_run=%s, span=%d days, reward=%.2f",
                current_result.get("full_data_run", False),
                span_days,
                current_result["composite_reward"],
            )

        # (4) We'll define an extended state for the meta-agent,
        # but that happens in meta_control_loop.
        # For the main training, we keep your code.

        reward_val = current_result["composite_reward"]
        # Update running statistics to normalise the reward signal
        self.reward_running_mean = 0.99 * self.reward_running_mean + 0.01 * reward_val
        diff = reward_val - self.reward_running_mean
        self.reward_running_var = 0.99 * self.reward_running_var + 0.01 * diff**2
        std = float(self.reward_running_var**0.5 + 1e-6)
        norm_reward = diff / std
        scaled_target = torch.tanh(
            torch.tensor(norm_reward, dtype=torch.float32, device=self.device)
        )
        advantage = scaled_target.clone()
        total_loss = 0.0
        nb = 0
        for m in self.models:
            m.train()
        # Serialise updates with meta-control thread
        with G.model_lock:
            accum_counter = 0
            # Main mini-batch training loop
            for batch_idx, (batch_x, batch_y) in enumerate(dl_train):
                G.inc_step()
                G.bump_warmup()
                if accum_counter == 0:
                    for opt in self.optimizers:
                        opt.zero_grad()
                bx = self._align_features(batch_x.to(self.device).contiguous())
                by = batch_y.to(self.device)

                if "device_type" in inspect.signature(autocast).parameters:
                    ctx = autocast(
                        device_type=self.device.type,
                        enabled=(self.device.type == "cuda"),
                    )
                else:
                    ctx = (
                        autocast(enabled=(self.device.type == "cuda"))
                        if self.device.type == "cuda"
                        else nullcontext()
                    )

                losses = []
                ce_vals: list[float] = []
                r_vals: list[float] = []
                skip_batch = False
                with ctx, torch.autograd.set_detect_anomaly(True):
                    for model in self.models:
                        logits, _, pred_reward = model(bx.clone())
                        self.entropies.append(getattr(model, "last_entropy", 0.0))
                        self.max_probs.append(getattr(model, "last_max_prob", 0.0))
                        ce_loss = self.criterion(logits, by)

                        log_probs = F.log_softmax(logits, dim=1)
                        act_lp = log_probs.gather(1, by.view(-1, 1)).squeeze(1)

                        ce_vals.append(float(ce_loss))

                        if not torch.isfinite(pred_reward).all():
                            logging.error(
                                "Non‑finite pred_reward detected at step %s",
                                self.train_steps,
                            )
                            logging.error(
                                "pred_reward stats: min=%s max=%s",
                                pred_reward.min().item(),
                                pred_reward.max().item(),
                            )
                            continue

                        use_reward = self.train_steps >= self.warmup_steps
                        logging.info(
                            f"RL active: {use_reward} at train_step {self.train_steps}"
                        )
                        if use_reward:
                            r_loss = self.mse_loss_fn(
                                pred_reward, scaled_target.expand_as(pred_reward)
                            )

                            rl_loss = -(advantage * act_lp.mean())
                        else:
                            r_loss = torch.tensor(0.0, device=self.device)
                            rl_loss = torch.tensor(0.0, device=self.device)

                        if batch_idx < 3:
                            logging.info(
                                "DBG_LOSSES batch=%d ce=%.6f r=%.6f rl=%.6f",
                                batch_idx,
                                ce_loss.item(),
                                r_loss.item(),
                                rl_loss.item(),
                            )

                        loss = ce_loss + self.reward_loss_weight * r_loss + rl_loss

                        if not torch.isfinite(loss).all():
                            logging.error(
                                "Non‑finite loss detected at step %s", self.train_steps
                            )
                            logging.error(
                                "ce_loss=%s reward_loss=%s rl_loss=%s",
                                ce_loss.item(),
                                r_loss.item(),
                                rl_loss.item(),
                            )
                            skip_batch = True
                            break

                        losses.append(loss)
                        r_vals.append(float(r_loss))

                if skip_batch or not losses:
                    accum_counter = 0
                    for opt in self.optimizers:
                        opt.zero_grad()
                    continue

                total_batch_loss = torch.stack(losses).sum() / self.grad_accum_steps
                self.scaler.scale(total_batch_loss).backward()

                accum_counter += 1
                if accum_counter >= self.grad_accum_steps:
                    for idx_m, (model, opt_) in enumerate(
                        zip(self.models, self.optimizers)
                    ):
                        self.scaler.unscale_(opt_)
                        if idx_m == 0 and hasattr(model, "fc"):
                            g = model.fc.weight.grad
                            g_norm = g.abs().mean().item() if g is not None else 0.0
                            lr = opt_.param_groups[0]["lr"]
                            logging.debug(
                                "Epoch %d, grad_norm=%.6f, lr=%.6f",
                                self.train_steps,
                                g_norm,
                                lr,
                            )
                        torch.nn.utils.clip_grad_norm_(
                            self.models[idx_m].parameters(), 1.0
                        )
                        try:
                            self.scaler.step(opt_)
                        except AssertionError:
                            opt_.step()
                            self.scaler = GradScaler(enabled=False)
                        else:
                            self.scaler.update()
                        from . import hyperparams as _hp

                        for pg in opt_.param_groups:
                            pg["lr"] = _hp.mutate_lr(pg["lr"], 0.0)
                            pg["weight_decay"] = _hp.mutate_lr(
                                pg.get("weight_decay", 0.0), 0.0
                            )
                        opt_.zero_grad()
                        for cyc in self.cycle:
                            try:
                                if getattr(cyc, "step_num", 0) < cyc.total_steps:
                                    cyc.step()
                                    cyc.step_num = getattr(cyc, "step_num", 0) + 1
                            except ValueError as e:
                                logging.warning("LR scheduler skipped: %s", e)
                        from . import hyperparams as _hp

                        for pg in opt_.param_groups:
                            pg["lr"] = _hp.mutate_lr(pg["lr"], 0.0)
                            pg["weight_decay"] = _hp.mutate_lr(
                                pg.get("weight_decay", 0.0), 0.0
                            )
                    self.train_steps += 1
                    logging.debug(f"Incremented train_steps to {self.train_steps}")
                    batch_loss = sum(loss_i.item() for loss_i in losses)
                    total_loss += (
                        (batch_loss / len(self.models))
                        if not np.isnan(batch_loss)
                        else 0.0
                    )
                    nb += 1
                    accum_counter = 0

            # Flush any remaining gradients
            if accum_counter > 0:
                for model, opt_ in zip(self.models, self.optimizers):
                    self.scaler.unscale_(opt_)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    try:
                        self.scaler.step(opt_)
                    except AssertionError:
                        opt_.step()
                        self.scaler = GradScaler(enabled=False)
                    else:
                        self.scaler.update()
                    opt_.zero_grad()
                self.train_steps += 1
                logging.debug(f"Incremented train_steps to {self.train_steps}")
                accum_counter = 0

            train_loss = total_loss / nb

            val_loss = self.evaluate_val_loss(dl_val) if dl_val else None
            if val_loss is not None:
                for sch in self.schedulers:
                    sch.step(val_loss)
                if val_loss < self.best_val_loss - 1e-6:
                    self.best_val_loss = val_loss
                    self.patience = 0
                else:
                    self.patience += 1
                if self.patience >= 12:
                    from . import hyperparams as _hp

                    for opt in self.optimizers:
                        for pg in opt.param_groups:
                            pg["lr"] = _hp.mutate_lr(pg["lr"], pg["lr"] * -0.9)
                            pg["weight_decay"] = _hp.mutate_lr(
                                pg.get("weight_decay", 0.0), 0.0
                            )
                    self.patience = 0

            # (2) Adjust Reward Penalties => less harsh for zero trades/ negative net
            trades_now = len(current_result["trade_details"])
            raw_reward = current_result["composite_reward"]

            # ------------------------------------------------------------------
            # Update global best metrics using the raw (unpenalised) reward
            # ------------------------------------------------------------------
            if (
                update_globals
                and trades_now > 0
                and raw_reward > G.global_best_composite_reward
            ):
                G.global_best_composite_reward = raw_reward
                G.global_best_sharpe = current_result["sharpe"]
                G.global_best_equity_curve = current_result["equity_curve"]
                G.global_best_drawdown = current_result["max_drawdown"]
                G.global_best_net_pct = current_result["net_pct"]
                G.global_best_num_trades = trades_now
                G.global_best_win_rate = current_result["win_rate"]
                G.global_best_profit_factor = current_result["profit_factor"]
                G.global_best_avg_trade_duration = current_result["avg_trade_duration"]
                G.global_best_avg_win = current_result.get("avg_win", 0.0)
                G.global_best_avg_loss = current_result.get("avg_loss", 0.0)
                G.global_best_trade_details = current_result["trade_details"]
                G.global_best_inactivity_penalty = current_result["inactivity_penalty"]
                G.global_best_days_in_profit = current_result["days_in_profit"]
                G.global_best_lr = self.optimizers[0].param_groups[0]["lr"]
                G.global_best_wd = (
                    self.optimizers[0].param_groups[0].get("weight_decay", 0)
                )
                G.global_best_yearly_stats_table = table_str
                G.global_best_monthly_stats_table = monthly_table
                self.best_state_dicts = [m.state_dict() for m in self.models]
                self.save_best_weights(self.weights_path)
                logging.info(
                    "SAVED_BEST_WEIGHTS epoch=%d reward=%.2f path=%s",
                    self.train_steps,
                    current_result["composite_reward"],
                    self.weights_path,
                )

            # Use the raw reward directly without heavy trade count penalties
            cur_reward = raw_reward

            # (3) Dynamic Patience => measure improvement
            # We'll track the last 10 net profits
            avg_improvement = (
                np.mean(G.global_backtest_profit[-10:])
                if len(G.global_backtest_profit) >= 10
                else 0
            )
            attn_entropy = (
                float(np.mean(G.global_attention_entropy_history[-100:]))
                if G.global_attention_entropy_history
                else 0.0
            )
            if attn_entropy < 0.5:
                logging.warning("Attention entropy low: %.2f", attn_entropy)

                G.set_status("Warning: attention entropy < 0.5", "")

            if cur_reward > self.best_composite_reward and trades_now > 0:

                # Disable risk-based rejection of epoch improvements
                # if reject_if_risky(
                #     cur_reward,
                #     G.global_max_drawdown,
                #     attn_entropy,
                # ):
                #     self.rejection_count_this_epoch += 1
                #     logging.info(
                #         "REJECTED by risk filter",
                #         extra={
                #             "epoch": self.train_steps,
                #             "sharpe": G.global_sharpe,
                #             "max_dd": G.global_max_drawdown,
                #             "attn_entropy": attn_entropy,
                #             "lr": self.optimizers[0].param_groups[0]["lr"],
                #         },
                #     )
                #     G.set_status("Risk", "Epoch rejected")
                # else:
                self.best_composite_reward = cur_reward
                self.patience_counter = 0
                self.best_state_dicts = [m.state_dict() for m in self.models]
                self.save_best_weights()
                logging.info(
                    "NEW_BEST_CANDIDATE",
                    extra={
                        "epoch": self.train_steps,
                        "sharpe": G.global_sharpe,
                        "max_dd": G.global_max_drawdown,
                        "attn_entropy": attn_entropy,
                        "lr": self.optimizers[0].param_groups[0]["lr"],
                    },
                )

            else:
                if trades_now == 0:
                    logging.info("NOT_PROMOTED: trades = 0")
                else:
                    logging.info(
                        "NOT_PROMOTED: composite_reward=%.1f < %.1f",
                        cur_reward,
                        self.best_composite_reward,
                    )
                self.patience_counter += 1
                # If net improvements are small => bigger patience
                # If average improvement >=1 => shorter patience
                patience_threshold = 30 if avg_improvement < 1.0 else 15
                if self.patience_counter >= patience_threshold:
                    if random.random() < 0.7:
                        _ = np.random.choice([1e-5, 1e-4, 1e-3])
                    else:
                        for m in self.models:
                            for layer in m.modules():
                                if isinstance(layer, nn.Linear):
                                    with torch.no_grad():
                                        new_w = torch.empty_like(layer.weight)
                                        nn.init.kaiming_normal_(new_w)
                                        layer.weight.copy_(new_w)
                                        if layer.bias is not None:
                                            layer.bias.copy_(
                                                torch.full_like(layer.bias, 0.1)
                                            )
                        if random.random() < 0.3:
                            self.models = [
                                TradingModel(
                                    input_size=self.n_features,
                                    hidden_size=np.random.choice([128, 256]),
                                    dropout=np.random.uniform(0.3, 0.6),
                                ).to(self.device)
                                for _ in range(len(self.models))
                            ]
                            lr = self.optimizers[0].param_groups[0]["lr"]
                            wd = (
                                self.optimizers[0]
                                .param_groups[0]
                                .get("weight_decay", 0.0)
                            )
                            self.optimizers = [
                                optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
                                for m in self.models
                            ]
                            self.schedulers = [
                                ReduceLROnPlateau(
                                    opt,
                                    mode="min",
                                    patience=6,
                                    factor=0.8,
                                    min_lr=2e-5,
                                )
                                for opt in self.optimizers
                            ]
                            self.cycle = [
                                OneCycleLR(
                                    o,
                                    max_lr=lr,
                                    total_steps=self.total_steps,
                                )
                                for o in self.optimizers
                            ]
                            for sch in self.cycle:
                                sch.step_num = 0
                                sch.total_steps = self.total_steps
                            if "device" in inspect.signature(GradScaler).parameters:
                                self.scaler = GradScaler(
                                    enabled=(self.device.type == "cuda"),
                                    device=self.device,
                                )
                            else:
                                self.scaler = GradScaler(
                                    enabled=(self.device.type == "cuda")
                                )
                        self.patience_counter = 0

        # update best metrics when composite reward improves
        if (
            update_globals
            and current_result["composite_reward"] > G.global_best_composite_reward
        ):
            G.global_best_equity_curve = current_result["equity_curve"]
            G.global_best_drawdown = current_result["max_drawdown"]
            G.global_best_net_pct = current_result["net_pct"]
            G.global_best_num_trades = trades_now
            G.global_best_win_rate = current_result["win_rate"]
            G.global_best_profit_factor = current_result["profit_factor"]
            G.global_best_avg_trade_duration = current_result["avg_trade_duration"]
            G.global_best_avg_win = current_result.get("avg_win", 0.0)
            G.global_best_avg_loss = current_result.get("avg_loss", 0.0)
            G.global_best_trade_details = current_result["trade_details"]
            G.global_best_sharpe = current_result["sharpe"]
            G.global_best_inactivity_penalty = current_result["inactivity_penalty"]

            G.global_best_composite_reward = current_result["composite_reward"]

            G.global_best_days_in_profit = current_result["days_in_profit"]
            G.global_best_lr = self.optimizers[0].param_groups[0]["lr"]
            G.global_best_wd = self.optimizers[0].param_groups[0].get("weight_decay", 0)
            _, best_table = compute_yearly_stats(
                current_result["equity_curve"],
                current_result["trade_details"],
                initial_balance=100.0,
            )
            G.global_best_yearly_stats_table = best_table

            _, best_monthly = compute_monthly_stats(
                current_result["equity_curve"],
                current_result["trade_details"],
                initial_balance=100.0,
            )
            G.global_best_monthly_stats_table = best_monthly
            self.best_state_dicts = [m.state_dict() for m in self.models]
            self.save_best_weights(self.weights_path)
            logging.info(
                "SAVED_BEST_WEIGHTS epoch=%d reward=%.2f path=%s",
                self.train_steps,
                current_result["composite_reward"],
                self.weights_path,
            )
            if self.train_steps > 0:
                update_best(
                    self.train_steps,
                    current_result["composite_reward"],
                    current_result["net_pct"],
                    self.weights_path,
                )

        mean_ent = float(torch.tensor(self.entropies).mean()) if self.entropies else 0.0
        mean_mp = float(torch.tensor(self.max_probs).mean()) if self.max_probs else 0.0
        logging.info(
            {
                "event": "EPOCH_SUMMARY",
                "epoch": self.train_steps,
                "mean_entropy": mean_ent,
                "mean_max_prob": mean_mp,
                "rejections": self.rejection_count_this_epoch,
            }
        )
        self.entropies.clear()
        self.max_probs.clear()
        self.rejection_count_this_epoch = 0
        # Return average training and validation loss for this epoch
        return train_loss, val_loss

    def evaluate_val_loss(self, dl_val: DataLoader) -> float:
        """Return the average loss on ``dl_val`` across the ensemble."""

        for m in self.models:
            m.eval()
        losses = []
        with torch.no_grad():
            for bx, by in dl_val:
                bx = self._align_features(bx.to(self.device))
                by = by.to(self.device)
                model_losses = []
                for mm in self.models:
                    lg, _, _ = mm(bx)
                    l_ = self.criterion(lg, by)
                    model_losses.append(l_.item())
                losses.append(np.mean(model_losses))
        val_loss = float(np.mean(losses))
        for m in self.models:
            m.train()
        return val_loss

    def predict(
        self, x: torch.Tensor, regime_probs: np.ndarray | None = None
    ) -> Tuple[int, float, None]:
        """Predict a single sample using optional regime probabilities."""

        log = logging.getLogger(__name__)
        with torch.no_grad():
            x = self._align_features(x.to(self.device))
            outs = [m(x) for m in self.models]

            if regime_probs is not None:
                try:
                    G.global_current_regime_prob = regime_probs.tolist()
                except Exception:
                    G.global_current_regime_prob = None
                rp = torch.tensor(regime_probs, dtype=torch.float32, device=self.device)
                rp = rp[: len(self.models)]
                conf, cluster_idx = float(rp.max().item()), int(rp.argmax().item())
                if conf > 0.8 and cluster_idx < len(self.models):
                    weights = torch.zeros(len(self.models), device=self.device)
                    weights[cluster_idx] = 1.0
                    log.info("HARD_SWITCH regime=%s conf=%.2f", cluster_idx, conf)
                else:
                    weights = rp
                    log.info("BLEND regime_probs=%s", [round(float(v), 3) for v in rp])
            else:
                scores = torch.tensor(
                    [
                        m.score_history[-1] if m.score_history else 0.0
                        for m in self.models
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                regime = getattr(G, "current_regime", None)
                if regime is not None:
                    if regime < len(self.models):
                        weights = torch.zeros(len(self.models), device=self.device)
                        weights[regime] = 1.0
                    else:
                        for idx, m in enumerate(self.models):
                            if hasattr(m, "reward_by_regime"):
                                scores[idx] = m.reward_by_regime.get(
                                    regime, scores[idx]
                                )
                        weights = torch.softmax(scores / self.tau, dim=0)
                else:
                    weights = torch.softmax(scores / self.tau, dim=0)

            weights = weights.cpu()
            probs = torch.stack([torch.softmax(o[0], dim=1).cpu() for o in outs])
            avgp = (weights.view(-1, 1, 1) * probs).sum(dim=0)
            idx = int(avgp[0].argmax().item())
            conf = float(avgp[0, idx].item())
            log.debug(
                "blend_weights=%s avg_prob=%s",
                [round(float(w), 3) for w in weights],
                conf,
            )
            log.info("ACTION index=%s", idx)
            return idx, conf, None

    def vectorized_predict(
        self,
        X,
        batch_size: int = 1024,
        regime_labels=None,
        regime_probs=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Return predictions for each window in ``windows_tensor``, allowing regime-specific strategy selection."""
        windows_tensor = X
        # [FIXED]# Robust feature-dimension handling
        expected_dim = self.n_features
        actual_dim = windows_tensor.shape[2]
        print(f"[PREDICT] Expected features: {expected_dim}, Actual: {actual_dim}")
        log = logging.getLogger(__name__)
        log.info(
            "[TRACE] model.predict() tensor shape=%s  |  model.input_features=%d",
            windows_tensor.shape,
            expected_dim,
        )
        assert (
            windows_tensor.shape[2] == expected_dim
        ), f"PREDICT shape mismatch: got {windows_tensor.shape[2]}, expected {expected_dim}"

        if actual_dim != expected_dim:
            raise ValueError(f"Expected {expected_dim} features, got {actual_dim}")

        windows_tensor = self._align_features(windows_tensor)

        with torch.no_grad():
            all_probs = []
            n_samples = windows_tensor.shape[0]
            scores = torch.tensor(
                [m.score_history[-1] if m.score_history else 0.0 for m in self.models],
                dtype=torch.float32,
                device=self.device,
            )
            base_weights = torch.softmax(scores / self.tau, dim=0).cpu()
            if regime_probs is not None:
                try:
                    G.global_regime_prob_history = [list(p) for p in regime_probs]
                except Exception:
                    G.global_regime_prob_history = []
            for i in range(0, n_samples, batch_size):
                batch = self._align_features(windows_tensor[i : i + batch_size])
                batch_probs = []
                for m in self.models:
                    logits, _, _ = m(batch)
                    prob = torch.softmax(logits, dim=1).cpu()
                    batch_probs.append(prob)
                batch_probs = torch.stack(batch_probs)
                avg_probs = []
                for j in range(batch_probs.shape[1]):
                    probs_ij = (base_weights.view(-1, 1) * batch_probs[:, j, :]).sum(0)
                    if regime_labels is not None:
                        label = regime_labels[i + j]
                        if label >= 0:  # hard route
                            probs_ij = batch_probs[label, j, :]
                        else:  # soft blend
                            w = torch.tensor(regime_probs[i + j], device=self.device)
                            probs_ij = (w.view(-1, 1) * batch_probs[:, j, :]).sum(0)
                    avg_probs.append(probs_ij)
                avg_probs = torch.stack(avg_probs, dim=0)
                all_probs.append(avg_probs)

            ret_probs = torch.cat(all_probs, dim=0)
            idxs = ret_probs.argmax(dim=1)
            confs = ret_probs.max(dim=1)[0]

            # dummy TradeParams-like dict (until RL head predicts real values)
            dummy_t = {
                "risk_fraction": torch.tensor([0.1]),
                "sl_multiplier": torch.tensor([5.0]),
                "tp_multiplier": torch.tensor([5.0]),
            }
            logging.info(
                "ACTION_BATCH regime_switching=%s first_action=%s",
                (
                    "ON"
                    if (regime_labels is not None or regime_probs is not None)
                    else "OFF"
                ),
                int(idxs[0]) if len(idxs) > 0 else -1,
            )
            return idxs.cpu(), confs.cpu(), dummy_t

    def optimize_models(self, dummy_input):
        """Placeholder for optional optimisation passes.

        Parameters
        ----------
        dummy_input:
            Example input tensor used to trace the model.

        Returns
        -------
        None
        """
        pass

    def prune(self, delta: float = 0.1) -> None:
        """Replace models whose EMA Sharpe underperforms."""

        if not self.models:
            return
        ema_vals = [getattr(m, "sharpe_ema", 0.0) for m in self.models]
        best = max(ema_vals)
        threshold = best - delta
        new_models: list[TradingModel] = []
        replaced = 0
        for m, v in zip(self.models, ema_vals):
            if v < threshold:
                replaced += 1
                nm = TradingModel(input_size=self.n_features).to(self.device)
                nm.score_history = []
                nm.sharpe_ema = 0.0
                new_models.append(nm)
            else:
                new_models.append(m)
        if replaced:
            lr = self.optimizers[0].param_groups[0]["lr"]
            wd = self.optimizers[0].param_groups[0].get("weight_decay", 0.0)
            self.models = new_models
            self.optimizers = [
                optim.AdamW(m.parameters(), lr=lr, weight_decay=wd) for m in self.models
            ]
            logging.info("PRUNED %d models threshold=%.2f", replaced, threshold)

    def save_best_weights(self, path: str | None = None) -> None:
        """Persist the best-performing weights to disk.

        Parameters
        ----------
        path:
            Destination filepath. Defaults to ``self.weights_path``.
        """
        if not self.best_state_dicts:
            return
        if path is None:
            path = self.weights_path
        torch.save(
            {
                "best_composite_reward": self.best_composite_reward,
                "state_dicts": self.best_state_dicts,
                "indicator_hparams": asdict(self.indicator_hparams),
            },
            path,
        )

    def load_best_weights(self, path: str | None = None, data_full=None) -> None:
        """Load weights from ``path`` and optionally refresh metrics."""
        if path is None:
            path = self.weights_path
        if os.path.isfile(path):
            try:
                ckpt = torch.load(path, map_location=self.device)
                self.best_composite_reward = ckpt.get(
                    "best_composite_reward", float("-inf")
                )
                self.best_state_dicts = ckpt["state_dicts"]
                ihp = ckpt.get("indicator_hparams")
                if ihp:
                    self.indicator_hparams = IndicatorHyperparams(**ihp)
                for m, sd in zip(self.models, self.best_state_dicts):
                    m.load_state_dict(sd, strict=False)
                if data_full and len(data_full) > 24:
                    loaded_result = robust_backtest(
                        self, data_full, indicator_hp=self.indicator_hparams
                    )
                    G.global_equity_curve = loaded_result["equity_curve"]
                    G.global_backtest_profit.append(loaded_result["net_pct"])
                    G.global_sharpe = loaded_result["sharpe"]
                    G.global_profit_factor = loaded_result["profit_factor"]
                    G.gui_event.set()
                    G.global_max_drawdown = loaded_result["max_drawdown"]
                    G.global_net_pct = loaded_result["net_pct"]
                    G.global_num_trades = loaded_result["trades"]
                    G.global_win_rate = loaded_result["win_rate"]
                    G.global_avg_trade_duration = loaded_result["avg_trade_duration"]
                    G.global_avg_win = loaded_result.get("avg_win", 0.0)
                    G.global_avg_loss = loaded_result.get("avg_loss", 0.0)
                    G.global_inactivity_penalty = loaded_result["inactivity_penalty"]
                    G.global_composite_reward = loaded_result["composite_reward"]
                    G.global_days_without_trading = loaded_result[
                        "days_without_trading"
                    ]
                    G.global_trade_details = loaded_result["trade_details"]
                    G.global_days_in_profit = loaded_result["days_in_profit"]
                    G.global_best_equity_curve = loaded_result["equity_curve"]
                    G.global_best_drawdown = loaded_result["max_drawdown"]
                    G.global_best_net_pct = loaded_result["net_pct"]
                    G.global_best_num_trades = loaded_result["trades"]
                    G.global_best_win_rate = loaded_result["win_rate"]
                    G.global_best_profit_factor = loaded_result["profit_factor"]
                    G.global_best_avg_trade_duration = loaded_result[
                        "avg_trade_duration"
                    ]
                    G.global_best_avg_win = loaded_result.get("avg_win", 0.0)
                    G.global_best_avg_loss = loaded_result.get("avg_loss", 0.0)
                    G.global_best_trade_details = loaded_result["trade_details"]
                    G.global_best_sharpe = loaded_result["sharpe"]
                    G.global_best_inactivity_penalty = loaded_result[
                        "inactivity_penalty"
                    ]
                    G.global_best_composite_reward = loaded_result["composite_reward"]
                    G.global_best_days_in_profit = loaded_result["days_in_profit"]
                    G.global_best_lr = self.optimizers[0].param_groups[0]["lr"]
                    G.global_best_wd = (
                        self.optimizers[0].param_groups[0].get("weight_decay", 0)
                    )
                    _, best_table = compute_yearly_stats(
                        loaded_result["equity_curve"],
                        loaded_result["trade_details"],
                        initial_balance=100.0,
                    )
                    G.global_yearly_stats_table = best_table
                    G.global_best_yearly_stats_table = best_table

                    _, best_monthly = compute_monthly_stats(
                        loaded_result["equity_curve"],
                        loaded_result["trade_details"],
                        initial_balance=100.0,
                    )
                    G.global_monthly_stats_table = best_monthly
                    G.global_best_monthly_stats_table = best_monthly
            except Exception:
                pass


def main() -> None:
    """Simple sanity check when run directly."""
    EnsembleModel()
    print("[DEBUG] EnsembleModel initialised")


if __name__ == "__main__":
    main()
