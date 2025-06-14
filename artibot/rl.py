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
import time


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
    """Transformer policy used by :class:`MetaTransformerRL`."""

    def __init__(self) -> None:
        super().__init__()

        self.toggle_keys = [k for k in ACTION_KEYS if k.startswith("toggle_")]
        self.gauss_keys = ["d_long_frac", "d_short_frac"]
        self.discrete_keys = [
            k
            for k in ACTION_KEYS
            if k not in self.toggle_keys and k not in self.gauss_keys
        ]

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
        self.policy_head = nn.Linear(d_model, len(self.discrete_keys))
        self.bandit_head = nn.Linear(d_model, len(self.toggle_keys))
        self.gauss_head = nn.Linear(d_model, len(self.gauss_keys))
        self.log_std = nn.Parameter(torch.zeros(len(self.gauss_keys)))
        self.value_head = nn.Linear(d_model, 1)

    def sample_action(self) -> dict[str, float]:
        """Return a random action within ``ACTION_SPACE``."""

        act: dict[str, float] = {}
        for k, (low, high) in ACTION_SPACE.items():
            if isinstance(low, int) and isinstance(high, int):
                act[k] = _random.randint(low, high)
            else:
                act[k] = _random.uniform(low, high)
        return act

    def forward(self, x):
        """Return policy components and value for state ``x``."""

        x_emb = self.embed(x).unsqueeze(1)
        x_pe = self.pos_enc(x_emb)
        x_enc = self.transformer_enc(x_pe)
        rep = x_enc.squeeze(1)
        pol_logits = self.policy_head(rep)
        bandit_logits = self.bandit_head(rep)
        gauss_mean = self.gauss_head(rep)
        value = self.value_head(rep).squeeze(1)
        return pol_logits, bandit_logits, gauss_mean, value


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
        self.discrete_keys = self._model.discrete_keys
        self.toggle_keys = self._model.toggle_keys
        self.gauss_keys = self._model.gauss_keys
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
        """Return an action dictionary with PPO-compatible log probability."""

        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = self.model(state)
            if len(out) == 4:
                p_logits, b_logits, g_mean, value = out
            else:
                p_logits, value = out
                b_logits = torch.zeros((1, len(self.toggle_keys)))
                g_mean = torch.zeros((1, len(self.gauss_keys)))

        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * (
            self.eps_decay**self.steps
        )
        self.steps += 1

        if _random.random() < eps_threshold:
            act = self.model.sample_action()
            logp = torch.tensor(0.0)
            self.prev_logits = (p_logits, b_logits, g_mean)
            self.prev_logprob = logp
            self.prev_action_idx = None
            self.last_bandit_probs = {}
            return act, logp, value

        # discrete action
        dist = torch.distributions.Categorical(logits=p_logits)
        action_idx = dist.sample()
        logp = dist.log_prob(action_idx)
        key = self.discrete_keys[action_idx.item()]
        low, high = ACTION_SPACE[key]
        if isinstance(low, int) and isinstance(high, int):
            value_v = _random.randint(low, high)
        else:
            value_v = _random.uniform(low, high)
        act: dict[str, float] = {key: value_v}

        # bandit toggles via gumbel-sigmoid
        probs = torch.sigmoid(b_logits).squeeze(0)
        self.last_bandit_probs = {
            k: float(probs[i].item()) for i, k in enumerate(self.toggle_keys)
        }
        u = torch.rand_like(probs)
        gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
        bandit_sample = torch.sigmoid((b_logits.squeeze(0) + gumbel) / 1.0)
        toggles = (bandit_sample > 0.5).int().tolist()
        for k, val_t in zip(self.toggle_keys, toggles):
            act[k] = int(val_t)
            logp = logp + (
                F.logsigmoid(b_logits.squeeze(0)[self.toggle_keys.index(k)])
                if val_t
                else F.logsigmoid(-b_logits.squeeze(0)[self.toggle_keys.index(k)])
            )

        # gaussian long/short fraction deltas
        mu = torch.tanh(g_mean.squeeze(0))
        std = self.model.log_std.exp()
        eps = torch.randn_like(mu)
        sample = torch.tanh(mu + std * eps)
        for i, k in enumerate(self.gauss_keys):
            low, high = ACTION_SPACE[k]
            scale = (high - low) / 2.0
            act[k] = float(sample[i].item() * scale)
            log_prob_gauss = -0.5 * ((sample[i] - mu[i]) ** 2) / (
                std[i] ** 2
            ) - torch.log(std[i] * math.sqrt(2 * math.pi))
            logp = logp + log_prob_gauss

        self.prev_logits = (p_logits, b_logits, g_mean)
        self.prev_logprob = logp.detach()
        self.prev_action_idx = action_idx
        return act, logp.detach(), value

    def update(self, state_np, action_idx, reward, next_state_np, logprob, value):
        """Update the meta policy using PPO."""

        s = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        ns = torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0)
        out = self.model(s)
        if len(out) == 4:
            p_logits, _, _, val_s = out
        else:
            p_logits, val_s = out
        dist = torch.distributions.Categorical(logits=p_logits)
        a = torch.tensor([action_idx]) if action_idx is not None else dist.sample()
        lp_s = dist.log_prob(a)
        if not hasattr(self, "prev_logits"):
            self.prev_logits = (
                p_logits.detach(),
                torch.zeros_like(p_logits),
                torch.zeros_like(p_logits),
            )
            logprob = torch.tensor(0.0)
        with torch.no_grad():
            out_ns = self.model(ns)
            val_ns = out_ns[-1]
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

        ratio = torch.exp(lp_s - logprob)
        clipped = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
        pg_loss = -torch.min(ratio * advantage, clipped * advantage)

        old_p_logits, _, _ = self.prev_logits
        old_dist = torch.distributions.Categorical(logits=old_p_logits)
        new_dist = torch.distributions.Categorical(logits=p_logits)
        kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
        entropy = new_dist.entropy().mean()

        loss_p = pg_loss + kl + (-1e-3 * entropy)
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

        for k, prob in getattr(self, "last_bandit_probs", {}).items():
            logging.info("FEATURE_IMPORTANCE %s %.3f", k, prob)

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

        for key, flag in [
            ("toggle_ema", "use_ema"),
            ("toggle_donchian", "use_donchian"),
            ("toggle_kijun", "use_kijun"),
            ("toggle_tenkan", "use_tenkan"),
            ("toggle_disp", "use_displacement"),
        ]:
            if act.get(key):
                cur = getattr(indicator_hp, flag)
                setattr(indicator_hp, flag, not cur)

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

        gross = hp.long_frac + hp.short_frac
        max_gross = getattr(G, "MAX_GROSS_PCT", 1.0)
        if gross > max_gross:
            scale = max_gross / gross
            hp.long_frac *= scale
            hp.short_frac *= scale

        act_str = ", ".join(
            f"{k}={v:+.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in act.items()
        )
        logging.info(
            "META_MUTATION %s sl=%.2f tp=%.2f atr=%d vortex=%d cmf=%d, ema=%d, don=%d, kij=%d, ten=%d, disp=%d, long=%.3f short=%.3f",
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

        ind = indicator_hp
        i = G.timeline_index % G.timeline_depth
        G.timeline_ind_on[i] = np.array(
            [
                ind.use_ema,
                ind.use_sma,
                ind.use_rsi,
                ind.use_kijun,
                ind.use_tenkan,
                ind.use_displacement,
            ],
            dtype=np.uint8,
        )
        G.timeline_trades[i] = 1 if (hp.long_frac > 0 or hp.short_frac > 0) else 0
        G.timeline_index += 1


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
    prev_reward = prev_r
    st_sharpe = G.global_sharpe
    st_dd = G.global_max_drawdown
    st_trades = G.global_num_trades
    st_days = G.global_days_in_profit if G.global_days_in_profit else 0.0

    state = np.array(
        [prev_r, best_r, st_sharpe, abs(st_dd), st_trades, st_days], dtype=np.float32
    )

    cycle_count = 0
    while True:
        if not G.is_bot_running():
            time.sleep(1.0)
            continue
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
            reward_raw = curr2 - prev_reward
            agent.reward_ema = 0.95 * agent.reward_ema + 0.05 * abs(reward_raw)
            shaped_r = reward_raw / (agent.reward_ema + 1e-8)
            agent.update(state, 0, shaped_r, new_state, logp, val_s)
            prev_reward = curr2
            rew_delta = shaped_r

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
