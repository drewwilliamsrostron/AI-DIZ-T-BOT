# Ensemble model handling and training utilities.
# Includes optimisation logic and RL loss composition.
"""Model ensemble used during training and prediction."""

# ruff: noqa: F403, F405

import inspect
from contextlib import nullcontext
from typing import Iterable, Optional, Tuple
from threading import Event
from itertools import product
import threading
import logging
import random
import hashlib

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
    ) -> None:
        super().__init__()
        device = torch.device(device) if device is not None else get_device()
        self.device = device
        self.weights_path = weights_path
        self.indicator_hparams = IndicatorHyperparams(
            rsi_period=14, sma_period=10, macd_fast=12, macd_slow=26, macd_signal=9
        )
        self.hp = HyperParams(indicator_hp=self.indicator_hparams)

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
        self._mask_lock = threading.Lock()
        self.register_buffer("feature_mask", torch.ones(1, dim, device=device))
        print("[DEBUG] Model moved to device")
        from . import hyperparams as _hp

        lr = max(lr, _hp.LR_MIN)
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
        self.train_steps = 0
        # Start with a small reward weight and no delay
        self.reward_loss_weight = 0.05
        self.max_reward_loss_weight = 0.2
        self.patience = 0
        self.delayed_reward_epochs = 0

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
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = x.to(self.device)
        outs = [m(x) for m in self.models]
        return outs

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
        # Mutate shared state on the globals module so the GUI sees progress

        # ------------------------------------------------------------------
        # Optional indicator sweep to tune hyperparameters at runtime
        # ------------------------------------------------------------------
        sweep_every = int(os.environ.get("SWEEP_EVERY", "1"))
        run_sweep = sweep_every > 0 and self.train_steps % sweep_every == 1
        best_result: dict | None = None
        best_cfg: dict | None = None
        if run_sweep:

            # ---------------- START merged block ----------------
            # --- ❶  Build parameter grid -------------------------------------------------
            sma_opts = [10, 20]
            rsi_opts = [9, 14]
            macd_fast_opts = [12, 16]
            macd_slow_opts = [26, 30]
            macd_sig_opts = [9, 12]
            ema_opts = [20, 50]
            atr_opts = [14, 21]
            vortex_opts = [14, 21]
            cmf_opts = [20, 30]
            donchian_opts = [20, 30]
            kijun_opts = [26, 34]
            tenkan_opts = [9, 12]
            disp_opts = [26, 52]
            conf_opts = [self.hp.conf_threshold, self.hp.conf_threshold * 1.5]
            sl_mults = [1.0, 1.5]
            tp_mults = [1.0, 1.5]

            param_sets = list(
                product(
                    sma_opts,
                    rsi_opts,
                    macd_fast_opts,
                    macd_slow_opts,
                    macd_sig_opts,
                    ema_opts,
                    atr_opts,
                    vortex_opts,
                    cmf_opts,
                    donchian_opts,
                    kijun_opts,
                    tenkan_opts,
                    disp_opts,
                    conf_opts,
                    sl_mults,
                    tp_mults,
                )
            )

            # --- ❷  Sweep the grid --------------------------------------------------------
            for cfg in param_sets:
                # Try each configuration and record the best metrics
                (
                    sma_period,
                    rsi_period,
                    macd_fast,
                    macd_slow,
                    macd_sig,
                    ema_period,
                    atr_period,
                    vortex_period,
                    cmf_period,
                    donchian_period,
                    kijun_period,
                    tenkan_period,
                    disp_period,
                    conf,
                    sl_mult,
                    tp_mult,
                ) = cfg

                # apply to indicator hyper-params
                hp = self.indicator_hparams
                hp.sma_period = sma_period
                hp.rsi_period = rsi_period
                hp.macd_fast = macd_fast
                hp.macd_slow = macd_slow
                hp.macd_signal = macd_sig
                hp.ema_period = ema_period
                hp.atr_period = atr_period
                hp.vortex_period = vortex_period
                hp.cmf_period = cmf_period
                hp.donchian_period = donchian_period
                hp.kijun_period = kijun_period
                hp.tenkan_period = tenkan_period
                hp.displacement = disp_period

                self.hp.conf_threshold = conf
                sl = self.hp.sl * sl_mult
                tp = self.hp.tp * tp_mult
                G.update_trade_params(sl, tp)

                result = robust_backtest(
                    self, data_full
                )  # no “features” arg inside sweep

                if result.get("trades", 0) == 0:
                    logging.info("IGNORED_EMPTY_BACKTEST: 0 trades in result")
                else:
                    G.push_backtest_metrics(result)

                logging.info(
                    "SWEEP_CFG",
                    extra={
                        "cfg": {
                            "sma": sma_period,
                            "rsi": rsi_period,
                            "macd_fast": macd_fast,
                            "macd_slow": macd_slow,
                            "macd_sig": macd_sig,
                            "ema": ema_period,
                            "atr": atr_period,
                            "vortex": vortex_period,
                            "cmf": cmf_period,
                            "donchian": donchian_period,
                            "kijun": kijun_period,
                            "tenkan": tenkan_period,
                            "disp": disp_period,
                            "conf": conf,
                            "sl": sl,
                            "tp": tp,
                        },
                        "reward": result.get("composite_reward", 0.0),
                    },
                )

                if best_result is None or result.get(
                    "composite_reward", 0.0
                ) > best_result.get("composite_reward", 0.0):
                    best_result = result
                    best_cfg = {
                        "sma": sma_period,
                        "rsi": rsi_period,
                        "macd_fast": macd_fast,
                        "macd_slow": macd_slow,
                        "macd_sig": macd_sig,
                        "ema": ema_period,
                        "atr": atr_period,
                        "vortex": vortex_period,
                        "cmf": cmf_period,
                        "donchian": donchian_period,
                        "kijun": kijun_period,
                        "tenkan": tenkan_period,
                        "disp": disp_period,
                        "conf": conf,
                        "sl": sl,
                        "tp": tp,
                    }

        # --- ❸  Re-apply best config & run final back-test ---------------------------
        if best_cfg is not None:
            hp = self.indicator_hparams
            hp.sma_period = best_cfg["sma"]
            hp.rsi_period = best_cfg["rsi"]
            hp.macd_fast = best_cfg["macd_fast"]
            hp.macd_slow = best_cfg["macd_slow"]
            hp.macd_signal = best_cfg["macd_sig"]
            hp.ema_period = best_cfg["ema"]
            hp.atr_period = best_cfg["atr"]
            hp.vortex_period = best_cfg["vortex"]
            hp.cmf_period = best_cfg["cmf"]
            hp.donchian_period = best_cfg["donchian"]
            hp.kijun_period = best_cfg["kijun"]
            hp.tenkan_period = best_cfg["tenkan"]
            hp.displacement = best_cfg["disp"]
            self.hp.conf_threshold = best_cfg["conf"]
            G.update_trade_params(best_cfg["sl"], best_cfg["tp"])

        # Run a back-test with the best parameters found (or current settings)
        logging.info(">>> ENTERING DEFCON 3: Full Backtest")
        G.set_defcon("DEFCON 3 \u2013 Full Dataset Backtest")
        logging.info(">>> Using current best hyperparams")
        current_result = best_result or robust_backtest(
            self, data_full, indicators=features
        )

        ignore_result = current_result.get("trades", 0) == 0
        if ignore_result:
            logging.info("IGNORED_EMPTY_BACKTEST: 0 trades in result")

        # --- ❹  Push to globals & ping GUI ------------------------------------------
        if not ignore_result:
            G.push_backtest_metrics(current_result)
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
        best = (
            G.global_best_composite_reward
            if G.global_best_composite_reward is not None
            else float("-inf")
        )
        if (
            update_globals
            and not ignore_result
            and current_result["composite_reward"] > best
        ):
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
            md5 = ""
            try:
                with open(self.weights_path, "rb") as f:
                    md5 = hashlib.md5(f.read()).hexdigest()
            except Exception:
                md5 = ""
            promote = G.nuke_armed or nk_gate_passes()
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
                    G.set_defcon("DEFCON 2 \u2013 Ready to Trade (NK Safe)")
                except Exception as exc:
                    logging.error("Live weight copy failed: %s", exc)
            else:
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
                G.set_status("Training", f"Not promoted: {short_reason}")

        # (4) We'll define an extended state for the meta-agent,
        # but that happens in meta_control_loop.
        # For the main training, we keep your code.

        # The composite reward is used as training target with a baseline
        baseline = (
            G.global_composite_reward_ema
            if G.global_composite_reward_ema is not None
            else 0.0
        )
        scaled_target = F.softsign(
            torch.tensor(
                (current_result["composite_reward"] - baseline) / 100.0,
                dtype=torch.float32,
                device=self.device,
            )
        )
        advantage = torch.tensor(
            current_result["composite_reward"] - baseline,
            dtype=torch.float32,
            device=self.device,
        )
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

                        use_reward = self.train_steps > self.delayed_reward_epochs
                        if use_reward:
                            r_loss = self.mse_loss_fn(
                                pred_reward, scaled_target.expand_as(pred_reward)
                            )

                            rl_loss = -(advantage * act_lp.mean())
                        else:
                            r_loss = torch.tensor(0.0, device=self.device)
                            rl_loss = torch.tensor(0.0, device=self.device)
                        loss = ce_loss + self.reward_loss_weight * r_loss + rl_loss

                        if not torch.isfinite(loss).all():
                            logging.error(
                                "Non‑finite loss detected at step %s", self.train_steps
                            )
                            logging.error(
                                "ce_loss=%s reward_loss=%s",
                                ce_loss.item(),
                                r_loss.item(),
                            )
                            continue
                        losses.append(loss)

                if not losses:
                    continue

                debug_ce = float(np.mean(ce_vals)) if ce_vals else 0.0
                debug_r = float(np.mean(r_vals)) if r_vals else 0.0
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
                            logging.debug(
                                "GRAD_CHECK step=%d ce_loss=%.6f r_loss=%.6f grad=%.6f",
                                self.train_steps,
                                debug_ce,
                                debug_r,
                                g_norm,
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
                    batch_loss = sum(loss_i.item() for loss_i in losses)
                    total_loss += (
                        (batch_loss / len(self.models))
                        if not np.isnan(batch_loss)
                        else 0.0
                    )
                    nb += 1
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
            cur_reward = current_result["composite_reward"]

            if trades_now == 0:
                # Reduced penalty for complete inactivity
                cur_reward -= 50
            elif trades_now < 5:
                # small graduated penalty per missing trade
                cur_reward -= 10 * (5 - trades_now)

            # negative net => smaller penalty
            if current_result["net_pct"] < 0:
                cur_reward -= 50  # previously 2000 -> 500

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
                if reject_if_risky(
                    cur_reward,
                    G.global_max_drawdown,
                    attn_entropy,
                ):
                    self.rejection_count_this_epoch += 1
                    logging.info(
                        "REJECTED by risk filter",
                        extra={
                            "epoch": self.train_steps,
                            "sharpe": G.global_sharpe,
                            "max_dd": G.global_max_drawdown,
                            "attn_entropy": attn_entropy,
                            "lr": self.optimizers[0].param_groups[0]["lr"],
                        },
                    )
                    G.set_status("Risk", "Epoch rejected")
                else:
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

        # track best net
        if update_globals and current_result["net_pct"] > G.global_best_net_pct:
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

    def predict(self, x: torch.Tensor) -> Tuple[int, float, None]:
        """Predict a single sample and return ``(index, confidence, None)``."""

        with torch.no_grad():
            x = self._align_features(x.to(self.device))
            outs = []
            for m in self.models:
                lg, _, _ = m(x)
                p_ = torch.softmax(lg, dim=1).cpu()
                outs.append(p_)
            avgp = torch.mean(torch.stack(outs), dim=0)
            idx = int(avgp[0].argmax().item())
            conf = float(avgp[0, idx].item())
            return idx, conf, None

    def vectorized_predict(
        self, windows_tensor: torch.Tensor, batch_size: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Return predictions for ``windows_tensor`` in mini-batches."""
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
            n_ = windows_tensor.shape[0]
            for i in range(0, n_, batch_size):
                batch = self._align_features(windows_tensor[i : i + batch_size])
                batch_probs = []
                for m in self.models:
                    lg, tpars, _ = m(batch)
                    pr_ = torch.softmax(lg, dim=1).cpu()
                    batch_probs.append(pr_)
                avg_probs = torch.mean(torch.stack(batch_probs), dim=0)
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
                for m, sd in zip(self.models, self.best_state_dicts):
                    m.load_state_dict(sd, strict=False)
                if data_full and len(data_full) > 24:
                    loaded_result = robust_backtest(self, data_full)
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
