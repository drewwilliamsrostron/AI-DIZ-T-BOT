"""Meta reinforcement learning agent controlling training hyperparameters."""

# ruff: noqa: F403, F405

import artibot.globals as G
from .hyperparams import HyperParams, IndicatorHyperparams
from .model import PositionalEncoding

import random as _random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import traceback
import logging
import math


ACTION_SPACE = {
    "lr": (-3e-4, 3e-4),
    "wd": (-3e-5, 3e-5),
    "toggle_sma": (0, 1),
    "toggle_rsi": (0, 1),
    "toggle_macd": (0, 1),
    "toggle_atr": (0, 1),
    "toggle_vortex": (0, 1),
    "toggle_cmf": (0, 1),
    "toggle_ichimoku": (0, 1),
    "toggle_ema": (0, 1),
    "toggle_donchian": (0, 1),
    "toggle_kijun": (0, 1),
    "toggle_tenkan": (0, 1),
    "toggle_disp": (0, 1),
    "d_sma_period": (-4, 4),
    "d_rsi_period": (-4, 4),
    "d_macd_fast": (-2, 2),
    "d_macd_slow": (-2, 2),
    "d_macd_signal": (-2, 2),
    "d_atr_period": (-4, 4),
    "d_vortex_period": (-4, 4),
    "d_cmf_period": (-4, 4),
    "d_ema_period": (-4, 4),
    "d_donchian_period": (-4, 4),
    "d_kijun_period": (-4, 4),
    "d_tenkan_period": (-4, 4),
    "d_displacement": (-4, 4),
    "d_sl": (-2.0, 2.0),
    "d_tp": (-2.0, 2.0),
    "d_long_frac": (-0.04, 0.04),
    "d_short_frac": (-0.04, 0.04),
}

ACTION_KEYS = list(ACTION_SPACE.keys())


###############################################################################
# NEW: A bigger action space that includes adjusting RSI, SMA, MACD + threshold
###############################################################################
class TransformerMetaAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.action_keys = ACTION_KEYS
        self.action_low = np.array([low for low, _ in ACTION_SPACE.values()])
        self.action_high = np.array([high for _, high in ACTION_SPACE.values()])

        self.state_dim = 6

        d_model = 32
        self.embed = nn.Linear(self.state_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=1,
            dim_feedforward=64,
            batch_first=True,
        )
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.policy_head = nn.Linear(d_model, len(self.action_keys))
        self.value_head = nn.Linear(d_model, 1)

    def sample_action(self) -> dict[str, float]:
        return {
            k: (
                _random.uniform(low, high)
                if isinstance(low, float)
                else _random.randint(low, high)
            )
            for k, (low, high) in ACTION_SPACE.items()
        }

        # (6) bigger or the same
        d_model = 32
        self.embed = nn.Linear(self.state_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=1,
            dim_feedforward=64,
            batch_first=True,
        )
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.policy_head = nn.Linear(d_model, self.n_actions)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x):
        x_emb = self.embed(x).unsqueeze(1)
        x_pe = self.pos_enc(x_emb)
        x_enc = self.transformer_enc(x_pe)
        rep = x_enc.squeeze(1)
        pol = self.policy_head(rep)
        val = self.value_head(rep).squeeze(1)
        return pol, val


class MetaTransformerRL:
    def __init__(
        self,
        ensemble,
        lr: float = 1e-3,
        *,
        value_range: tuple[float, float] = (-10.0, 10.0),
        target_range: tuple[float, float] = (-10.0, 10.0),
    ):
        self._model = TransformerMetaAgent()
        self.state_dim = self._model.state_dim
        self.action_keys = ACTION_KEYS
        self.opt = optim.AdamW(self._model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=100)
        self.gamma = 0.95
        self.ensemble = ensemble
        self.value_range = value_range
        self.target_range = target_range

        # Reward shaping baseline (EMA of |Δreward|)
        self.reward_ema = 1.0
        self.tau_inv = 1.0 / 50.0

        # (7) Scheduled exploration
        # Change epsilon parameters for longer exploration
        self.eps_start = 0.5  # Increased from 0.3
        self.eps_end = 0.15  # Increased from 0.05
        self.eps_decay = 0.995  # Slower decay
        # Add exploration reset mechanism
        self.exploration_reset_interval = 100  # not sure if this is used tho
        # Rest of init remains
        self.steps = 0
        self.last_improvement = 0

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        if not hasattr(self._model, "state_dim"):
            self._model.state_dim = self.state_dim

    def pick_action(self, state_np):
        """Return an action dictionary using an epsilon-greedy policy."""

        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, val = self.model(state)

        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * (
            self.eps_decay**self.steps
        )
        self.steps += 1

        if _random.random() < eps_threshold:
            act = self.model.sample_action()
            logp = torch.tensor(0.0)
            return act, logp, val

        dist = torch.distributions.Categorical(logits=logits)
        action_idx = dist.sample()
        logp = dist.log_prob(action_idx)
        key = self.action_keys[action_idx.item()]
        low, high = ACTION_SPACE[key]
        if isinstance(low, int) and isinstance(high, int):
            value = _random.randint(low, high)
        else:
            value = _random.uniform(low, high)
        act = {key: value}
        return act, logp, val

    def update(self, state_np, action_idx, reward, next_state_np, logprob, value):
        s = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        ns = torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0)
        pol_s, val_s = self.model(s)
        dist = torch.distributions.Categorical(logits=pol_s)
        lp_s = dist.log_prob(torch.tensor([action_idx]))
        with torch.no_grad():
            _, val_ns = self.model(ns)
        # Reward shaping using EMA of absolute reward changes
        self.reward_ema = (1 - self.tau_inv) * self.reward_ema + self.tau_inv * abs(
            reward
        )
        shaped = reward / max(self.reward_ema, 1e-8)
        target = shaped + self.gamma * val_ns.item()
        if not math.isfinite(target):
            logging.warning("Non-finite target: %s", target)
            target = self.target_range[1] if target > 0 else self.target_range[0]
        else:
            target = float(np.clip(target, self.target_range[0], self.target_range[1]))
        val_s = val_s.clamp(min=self.value_range[0], max=self.value_range[1])
        advantage = (target - val_s).detach()
        loss_p = -lp_s * advantage
        loss_v = F.mse_loss(val_s, torch.tensor([target], device=val_s.device))
        loss = loss_p + 0.5 * loss_v
        self.opt.zero_grad()
        if loss.requires_grad:
            loss.backward()
        else:
            for p in self.model.parameters():
                if p.requires_grad:
                    p.grad = torch.zeros_like(p)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.scheduler.step()
        if reward > 0:
            self.last_improvement = 0
        else:
            self.last_improvement += 1

    def apply_action(
        self,
        hp: HyperParams,
        indicator_hp: IndicatorHyperparams,
        act: dict[str, float],
    ) -> None:
        """Mutate ``hp`` and ``indicator_hp`` in-place based on ``act``."""

        if act.get("toggle_sma"):
            indicator_hp.use_sma = not indicator_hp.use_sma
        if act.get("toggle_rsi"):
            indicator_hp.use_rsi = not indicator_hp.use_rsi
        if act.get("toggle_macd"):
            indicator_hp.use_macd = not indicator_hp.use_macd
        if act.get("toggle_atr"):
            indicator_hp.use_atr = not indicator_hp.use_atr
        if act.get("toggle_vortex"):
            indicator_hp.use_vortex = not indicator_hp.use_vortex
        if act.get("toggle_cmf"):
            indicator_hp.use_cmf = not indicator_hp.use_cmf
        if act.get("toggle_ichimoku"):
            hp.use_ichimoku = not hp.use_ichimoku
        if act.get("toggle_ema"):
            indicator_hp.use_ema = not indicator_hp.use_ema
        if act.get("toggle_donchian"):
            indicator_hp.use_donchian = not indicator_hp.use_donchian
        if act.get("toggle_kijun"):
            indicator_hp.use_kijun = not indicator_hp.use_kijun
        if act.get("toggle_tenkan"):
            indicator_hp.use_tenkan = not indicator_hp.use_tenkan
        if act.get("toggle_disp"):
            indicator_hp.use_displacement = not indicator_hp.use_displacement

        indicator_hp.sma_period = max(
            2, indicator_hp.sma_period + int(act.get("d_sma_period", 0))
        )
        indicator_hp.rsi_period = max(
            2, indicator_hp.rsi_period + int(act.get("d_rsi_period", 0))
        )
        indicator_hp.macd_fast = max(
            2, indicator_hp.macd_fast + int(act.get("d_macd_fast", 0))
        )
        indicator_hp.macd_slow = max(
            indicator_hp.macd_fast + 1,
            indicator_hp.macd_slow + int(act.get("d_macd_slow", 0)),
        )
        indicator_hp.macd_signal = max(
            1, indicator_hp.macd_signal + int(act.get("d_macd_signal", 0))
        )

        indicator_hp.atr_period = max(
            4, indicator_hp.atr_period + int(act.get("d_atr_period", 0))
        )
        indicator_hp.vortex_period = max(
            4, indicator_hp.vortex_period + int(act.get("d_vortex_period", 0))
        )
        indicator_hp.cmf_period = max(
            4, indicator_hp.cmf_period + int(act.get("d_cmf_period", 0))
        )
        indicator_hp.ema_period = max(
            2, indicator_hp.ema_period + int(act.get("d_ema_period", 0))
        )
        indicator_hp.donchian_period = max(
            5, indicator_hp.donchian_period + int(act.get("d_donchian_period", 0))
        )
        indicator_hp.kijun_period = max(
            5, indicator_hp.kijun_period + int(act.get("d_kijun_period", 0))
        )
        indicator_hp.tenkan_period = max(
            5, indicator_hp.tenkan_period + int(act.get("d_tenkan_period", 0))
        )
        indicator_hp.displacement = max(
            1, indicator_hp.displacement + int(act.get("d_displacement", 0))
        )

        hp.sl = max(0.5, hp.sl + float(act.get("d_sl", 0.0)))
        hp.tp = max(0.5, hp.tp + float(act.get("d_tp", 0.0)))
        hp.long_frac = float(
            np.clip(
                hp.long_frac + float(act.get("d_long_frac", 0.0)),
                0.0,
                getattr(G, "MAX_SIDE_EXPOSURE_PCT", 1.0),
            )
        )
        hp.short_frac = float(
            np.clip(
                hp.short_frac + float(act.get("d_short_frac", 0.0)),
                0.0,
                getattr(G, "MAX_SIDE_EXPOSURE_PCT", 1.0),
            )
        )

        act_str = ", ".join(
            f"{k}={v:+.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in act.items()
        )
        logging.info(
            "META_MUTATION %s sl=%.2f tp=%.2f atr=%d vortex=%d cmf=%d, ema=%d, don=%d, kij=%d, ten=%d, disp=%d, long%%=%.3f, short%%=%.3f",
            act_str,
            hp.sl,
            hp.tp,
            indicator_hp.atr_period,
            indicator_hp.vortex_period,
            indicator_hp.cmf_period,
            indicator_hp.ema_period,
            indicator_hp.donchian_period,
            indicator_hp.kijun_period,
            indicator_hp.tenkan_period,
            indicator_hp.displacement,
            hp.long_frac,
            hp.short_frac,
        )

        G.sync_globals(hp, indicator_hp)


###############################################################################
# meta_control_loop
###############################################################################
def meta_control_loop(
    ensemble,
    dataset,
    agent,
    hp: HyperParams,
    indicator_hp: IndicatorHyperparams,
    interval=5.0,
):
    """Periodically tweak training parameters using the meta agent."""

    G.status_sleep("Starting meta agent", "", 2.0)

    # old initial state was just 2 dims. Now we add (sharpe, dd, trades, days_in_profit)
    prev_r = G.global_composite_reward if G.global_composite_reward else 0.0
    best_r = G.global_best_composite_reward if G.global_best_composite_reward else 0.0
    st_sharpe = G.global_sharpe
    st_dd = G.global_max_drawdown
    st_trades = G.global_num_trades
    st_days = G.global_days_in_profit if G.global_days_in_profit else 0.0

    state = np.array(
        [prev_r, best_r, st_sharpe, abs(st_dd), st_trades, st_days], dtype=np.float32
    )

    cycle_count = 0
    while True:
        try:
            if G.epoch_count < 1:

                G.status_sleep("Meta agent waiting for training", "", 1.0)

                continue

            curr_r = G.global_composite_reward if G.global_composite_reward else 0.0
            b_r = (
                G.global_best_composite_reward
                if G.global_best_composite_reward
                else 0.0
            )
            st_sharpe = G.global_sharpe
            st_dd = G.global_max_drawdown
            st_trades = G.global_num_trades
            st_days = G.global_days_in_profit if G.global_days_in_profit else 0.0
            new_state = np.array(
                [curr_r, b_r, st_sharpe, abs(st_dd), st_trades, st_days],
                dtype=np.float32,
            )

            act, logp, val_s = agent.pick_action(state)
            with G.model_lock:
                agent.apply_action(hp, indicator_hp, act)

            G.status_sleep("Meta agent sleeping", "", interval)

            curr2 = G.global_composite_reward if G.global_composite_reward else 0.0
            rew_delta = curr2 - curr_r
            agent.update(state, 0, rew_delta, new_state, logp, val_s)

            G.set_status(
                "Meta agent",
                f"Update {cycle_count}, last Δreward={rew_delta:.2f}",
            )
            cycle_count += 1

            act_str = ", ".join(
                f"{k}={v:+.2f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in act.items()
            )
            summary = (
                f"META_MUTATION {act_str} sl={hp.sl:.2f} tp={hp.tp:.2f}"
                f" ema={indicator_hp.ema_period} don={indicator_hp.donchian_period}"
                f" kij={indicator_hp.kijun_period} ten={indicator_hp.tenkan_period}"
                f" disp={indicator_hp.displacement}"
            )
            G.global_ai_adjustments_log += "\n" + summary
            G.global_ai_adjustments = summary

            state = new_state
            if agent.last_improvement > 20:
                # forced random reinit
                with G.model_lock, torch.no_grad():
                    for m in ensemble.models:
                        for layer in m.modules():
                            if isinstance(layer, nn.Linear):
                                nn.init.xavier_uniform_(layer.weight)
                                if layer.bias is not None:
                                    nn.init.zeros_(layer.bias)
                agent.last_improvement = 0
                msg = "\n[Stagnation] Forced random reinit of primary model!\n"
                G.global_ai_adjustments_log += msg

            G.status_sleep("Meta agent idle", "", 0.5)

        except Exception as e:
            G.set_status(f"Meta error: {e}", "")
            traceback.print_exc()
            G.status_sleep("Meta agent failed", "", 5.0)


###############################################################################
# Main
###############################################################################
