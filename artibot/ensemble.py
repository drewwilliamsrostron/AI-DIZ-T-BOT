"""Model ensemble used during training and prediction."""

# ruff: noqa: F403, F405

import inspect
from contextlib import nullcontext
import logging
import random

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
)

from .backtest import robust_backtest
from .dataset import IndicatorHyperparams
import artibot.globals as G
from .metrics import compute_yearly_stats
from .model import TradingModel


def reject_if_risky(sharpe: float, max_dd: float, entropy: float) -> bool:
    """Return True if metrics violate the risk thresholds."""
    if entropy < 1.0:
        return True  # reject collapsed runs
    return max_dd < -0.30 or sharpe < 1.0


def choose_best(rewards: list[float]) -> float:
    """Return the highest reward from ``rewards``.

    Raises ``ValueError`` when the list is empty.
    """
    if not rewards:
        raise ValueError("rewards cannot be empty")
    return max(rewards)


class EnsembleModel:
    """Simple container for multiple models and optimisers."""

    def __init__(self, device, n_models=2, lr=3e-4, weight_decay=1e-4):
        self.device = device

        # (6) We changed TradingModel to bigger capacity above
        self.models = [TradingModel().to(device) for _ in range(n_models)]
        self.optimizers = [
            optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
            for m in self.models
        ]
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([3.0, 3.0, 0.2]).to(device)
        )
        self.mse_loss_fn = nn.MSELoss()
        amp_on = device.type == "cuda"
        # GradScaler's `device` arg is not supported on older PyTorch versions
        if "device" in inspect.signature(GradScaler).parameters:
            self.scaler = GradScaler(enabled=amp_on, device=self.device.type)
        else:
            self.scaler = GradScaler(enabled=amp_on)
        self.best_val_loss = float("inf")
        self.best_composite_reward = float("-inf")
        self.best_state_dicts = None
        self.train_steps = 0
        self.reward_loss_weight = 0.2
        self.patience = 0
        self.delayed_reward_epochs = 25

        # (3) We'll do dynamic patience mechanism, so this is an initial
        self.patience_counter = 0

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
        self.cosine = [CosineAnnealingWarmRestarts(o, T_0=50) for o in self.optimizers]

        self.warmup_steps = 1000
        self.step_count = 0
        self.base_lrs = [opt.param_groups[0]["lr"] for opt in self.optimizers]

        # store global indicator hyperparams that meta-agent can change
        self.indicator_hparams = IndicatorHyperparams(
            rsi_period=14, sma_period=10, macd_fast=12, macd_slow=26, macd_signal=9
        )

    def train_one_epoch(self, dl_train, dl_val, data_full, stop_event=None):
        # mutate shared state on the globals module

        current_result = robust_backtest(self, data_full)

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

        dfy, table_str = compute_yearly_stats(
            current_result["equity_curve"],
            current_result["trade_details"],
            initial_balance=100.0,
        )
        G.global_yearly_stats_table = table_str

        # (4) We'll define an extended state for the meta-agent,
        # but that happens in meta_control_loop.
        # For the main training, we keep your code.

        # The composite reward is used as training target
        scaled_target = torch.tanh(
            torch.tensor(
                current_result["composite_reward"] / 100.0,
                dtype=torch.float32,
                device=self.device,
            )
        )
        total_loss = 0.0
        nb = 0
        for m in self.models:
            m.train()
        for batch_idx, (batch_x, batch_y) in enumerate(dl_train):
            bx = batch_x.to(self.device).contiguous().clone()
            by = batch_y.to(self.device)
            batch_loss = 0.0
            for idx_m, (model, opt_) in enumerate(zip(self.models, self.optimizers)):
                opt_.zero_grad()
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
                with ctx:
                    logits, _, pred_reward = model(bx)
                    ce_loss = self.criterion(logits, by)
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
                    else:
                        r_loss = torch.tensor(0.0, device=self.device)
                    loss = ce_loss + self.reward_loss_weight * r_loss
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

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(opt_)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                try:
                    self.scaler.step(opt_)
                except AssertionError:
                    # Older PyTorch versions sometimes fail to record
                    # the inf check state when using GradScaler.
                    # Fall back to a regular optimiser step.
                    opt_.step()
                    self.scaler = GradScaler(enabled=False)
                else:
                    self.scaler.update()
                self.step_count += 1
                if self.step_count <= self.warmup_steps:
                    new_lr = self.base_lrs[idx_m] * (
                        self.step_count / self.warmup_steps
                    )
                    for pg in opt_.param_groups:
                        pg["lr"] = new_lr
                else:
                    self.cosine[idx_m].step()
                batch_loss += loss.item()
            total_loss += (
                float(batch_loss / len(self.models))
                if not np.isnan(batch_loss)
                else 0.0
            )
            nb += 1
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
                for opt in self.optimizers:
                    for pg in opt.param_groups:
                        pg["lr"] = max(pg["lr"] * 0.1, 2e-6)
                self.patience = 0

        # (2) Adjust Reward Penalties => less harsh for zero trades/ negative net
        trades_now = len(current_result["trade_details"])
        cur_reward = current_result["composite_reward"]

        if trades_now == 0:
            # Reduced from 99999 => 500
            cur_reward -= 500
        elif trades_now < 5:
            # small graduated penalty
            cur_reward -= 100 * (5 - trades_now)

        # negative net => smaller penalty
        if current_result["net_pct"] < 0:
            cur_reward -= 500  # from 2000

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
        if cur_reward > self.best_composite_reward:
            if reject_if_risky(G.global_sharpe, G.global_max_drawdown, attn_entropy):
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
            else:
                self.best_composite_reward = cur_reward
                self.patience_counter = 0
                self.best_state_dicts = [m.state_dict() for m in self.models]
                self.save_best_weights("best_model_weights.pth")
        else:
            self.patience_counter += 1
            # If net improvements are small => bigger patience
            # If average improvement >=1 => shorter patience
            patience_threshold = 30 if avg_improvement < 1.0 else 15
            if self.patience_counter >= patience_threshold:
                # random approach: 70% chance => half LR, else random reinit
                # if random.random()< 0.7:
                #     new_lr= max(self.optimizers[0].param_groups[0]['lr']*0.5, 1e-6)
                #     for opt_ in self.optimizers:
                #         for grp in opt_.param_groups:
                #             grp['lr']= new_lr
                # else:
                #     # random reinit
                #     for m in self.models:
                #         for layer in m.modules():
                #             if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                #                 nn.init.xavier_uniform_(layer.weight)
                #                 if layer.bias is not None:
                #                     nn.init.zeros_(layer.bias)
                # self.patience_counter=0
                if random.random() < 0.7:
                    _ = np.random.choice([1e-5, 1e-4, 1e-3])  # More radical LR changes
                else:
                    # Full reinit with different initialization
                    for m in self.models:
                        for layer in m.modules():
                            if isinstance(layer, nn.Linear):
                                nn.init.kaiming_normal_(layer.weight)
                                if layer.bias is not None:
                                    nn.init.constant_(layer.bias, 0.1)
                    # Add random architecture modifications
                    if random.random() < 0.3:
                        self.models = [
                            TradingModel(
                                hidden_size=np.random.choice([128, 256]),
                                dropout=np.random.uniform(0.3, 0.6),
                            ).to(self.device)
                            for _ in range(len(self.models))
                        ]
                    self.patience_counter = 0

        # track best net
        if current_result["net_pct"] > G.global_best_net_pct:
            G.global_best_equity_curve = current_result["equity_curve"]
            G.global_best_sharpe = current_result["sharpe"]
            G.global_best_drawdown = current_result["max_drawdown"]
            G.global_best_net_pct = current_result["net_pct"]
            G.global_best_num_trades = trades_now
            G.global_best_inactivity_penalty = current_result["inactivity_penalty"]
            G.global_best_composite_reward = cur_reward
            G.global_best_days_in_profit = current_result["days_in_profit"]
            G.global_best_lr = self.optimizers[0].param_groups[0]["lr"]
            G.global_best_wd = self.optimizers[0].param_groups[0].get("weight_decay", 0)

        self.train_steps += 1
        return train_loss, val_loss

    def evaluate_val_loss(self, dl_val):
        for m in self.models:
            m.eval()
        losses = []
        with torch.no_grad():
            for bx, by in dl_val:
                bx = bx.to(self.device)
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

    def predict(self, x):
        with torch.no_grad():
            outs = []
            for m in self.models:
                lg, _, _ = m(x.to(self.device))
                p_ = torch.softmax(lg, dim=1).cpu()
                outs.append(p_)
            avgp = torch.mean(torch.stack(outs), dim=0)
            idx = int(avgp[0].argmax().item())
            conf = float(avgp[0, idx].item())
            return idx, conf, None

    def vectorized_predict(self, windows_tensor, batch_size=256):
        with torch.no_grad():
            all_probs = []
            n_ = windows_tensor.shape[0]
            for i in range(0, n_, batch_size):
                batch = windows_tensor[i : i + batch_size]
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
            # dummy param
            dummy_t = {
                "risk_fraction": torch.tensor([0.1]),
                "sl_multiplier": torch.tensor([5.0]),
                "tp_multiplier": torch.tensor([5.0]),
            }
            return idxs.cpu(), confs.cpu(), dummy_t

    def optimize_models(self, dummy_input):
        pass

    def save_best_weights(self, path="best_model_weights.pth"):
        if not self.best_state_dicts:
            return
        torch.save(
            {
                "best_composite_reward": self.best_composite_reward,
                "state_dicts": self.best_state_dicts,
            },
            path,
        )

    def load_best_weights(self, path="best_model_weights.pth", data_full=None):
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
                    G.global_best_equity_curve = loaded_result["equity_curve"]
                    G.global_best_sharpe = loaded_result["sharpe"]
                    G.global_best_drawdown = loaded_result["max_drawdown"]
                    G.global_best_net_pct = loaded_result["net_pct"]
                    G.global_best_num_trades = loaded_result["trades"]
                    G.global_best_inactivity_penalty = loaded_result[
                        "inactivity_penalty"
                    ]
                    G.global_best_composite_reward = loaded_result["composite_reward"]
                    G.global_best_days_in_profit = loaded_result["days_in_profit"]
                    G.global_best_lr = self.optimizers[0].param_groups[0]["lr"]
                    G.global_best_wd = (
                        self.optimizers[0].param_groups[0].get("weight_decay", 0)
                    )
            except Exception:
                pass
