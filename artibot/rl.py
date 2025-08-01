"""Meta reinforcement learning agent controlling training hyperparameters."""

# ruff: noqa: F403, F405

import artibot.globals as G
import artibot.hyperparams as hyperparams
from .hyperparams import HyperParams, IndicatorHyperparams
from .model import PositionalEncoding
from .feature_store import FEATURE_DIM

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
    "d_sma_period": (-2, 2),
    "d_rsi_period": (-2, 2),
    "d_macd_fast": (-2, 2),
    "d_macd_slow": (-2, 2),
    "d_macd_signal": (-2, 2),
    "d_atr_period": (-2, 2),
    "d_vortex_period": (-2, 2),
    "d_cmf_period": (-2, 2),
    "d_ema_period": (-2, 2),
    "d_donchian_period": (-2, 2),
    "d_kijun_period": (-2, 2),
    "d_tenkan_period": (-2, 2),
    "d_displacement": (-2, 2),
    "d_sl": (-2.0, 2.0),
    "d_tp": (-2.0, 2.0),
    "d_long_frac": (-0.04, 0.04),
    "d_short_frac": (-0.04, 0.04),
}

ACTION_KEYS = list(ACTION_SPACE.keys())

# track previous parameter state for logging
_prev_param_state: dict[str, object] = {}


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

        from .hyperparams import TRANSFORMER_HEADS

        d_model = (
            (32 + TRANSFORMER_HEADS - 1) // TRANSFORMER_HEADS
        ) * TRANSFORMER_HEADS
        self.embed = nn.Linear(self.state_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=TRANSFORMER_HEADS,
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
        assert (
            x.device == self.embed.weight.device
        ), f"input{ x.device } != model{ self.embed.weight.device }"
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
    _last_instance: "MetaTransformerRL | None" = None

    def __init__(
        self,
        ensemble,
        lr: float = 1e-3,
        *,
        value_range: tuple[float, float] = (-10.0, 10.0),
        target_range: tuple[float, float] = (-10.0, 10.0),
        device: torch.device | str | None = None,
        cfg: object | None = None,
    ):
        from .core.device import get_device

        self.device = torch.device(device) if device is not None else get_device()
        self._model = TransformerMetaAgent().to(self.device)
        self.state_dim = self._model.state_dim
        self.action_keys = ACTION_KEYS
        self.discrete_keys = self._model.discrete_keys
        self.toggle_keys = self._model.toggle_keys
        self.gauss_keys = self._model.gauss_keys
        self.opt = optim.AdamW(self._model.parameters(), lr=lr)
        # Disable cosine decay to avoid lr falling below 1e-5 and PyTorch order warning
        self.scheduler = None
        self.gamma = 0.95
        self.ensemble = ensemble
        self.value_range = value_range
        self.target_range = target_range

        # Reward shaping baseline (EMA of |Δreward|)
        self.reward_ema = 1.0
        self.tau_inv = 1.0 / 50.0
        if cfg and hasattr(cfg, "entropy_beta"):
            import math as _math
            import random as _rand

            mn = getattr(
                getattr(cfg, "entropy_beta"), "min", getattr(cfg, "entropy_beta")
            )
            mx = getattr(getattr(cfg, "entropy_beta"), "max", mn)
            self.entropy_beta = 10 ** _rand.uniform(_math.log10(mn), _math.log10(mx))
        else:
            self.entropy_beta = 0.005
        self.skip_risk_epochs = getattr(cfg, "skip_risk_epochs", 3) if cfg else 3
        self.low_kl_count = 0

        # (7) Scheduled exploration
        # Change epsilon parameters for longer exploration
        self.eps_start = 0.20
        self.eps_end = 0.05
        self.eps_decay = 0.99
        # Add exploration reset mechanism
        self.exploration_reset_interval = 100  # not sure if this is used tho
        # Rest of init remains
        self.steps = 0
        self.last_improvement = 0
        self.batch_buffer: list[tuple] = []
        MetaTransformerRL._last_instance = self

    def _to_device(self, *tensors):
        """Move tensors to ``self.device``.

        DataLoader should use ``pin_memory=True`` so the non-blocking copy is
        truly asynchronous.
        """
        return [t.to(self.device, non_blocking=True) for t in tensors]

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        if not hasattr(self._model, "state_dim"):
            self._model.state_dim = self.state_dim
        self._model.to(self.device, non_blocking=True)

    def pick_action(self, state_np):
        """Return an action dictionary with PPO-compatible log probability."""

        if getattr(self, "current_epoch", 0) < getattr(self, "skip_risk_epochs", 3):
            self.skip_risk_check = True
        else:
            self.skip_risk_check = False

        state = torch.as_tensor(state_np, dtype=torch.float32).unsqueeze(0)
        (state,) = self._to_device(state)
        with torch.no_grad():
            out = self.model(state)
            if len(out) == 4:
                p_logits, b_logits, g_mean, value = out
            else:
                p_logits, value = out
                b_logits = torch.zeros((1, len(self.toggle_keys)))
                g_mean = torch.zeros((1, len(self.gauss_keys)))

        # clamp logits and strip NaN/Inf to avoid errors in ``Categorical``
        p_logits = p_logits.clamp(-10.0, 10.0)
        p_logits = torch.nan_to_num(p_logits, nan=0.0, posinf=10.0, neginf=-10.0)

        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * (
            self.eps_decay**self.steps
        )
        self.steps += 1

        if _random.random() < eps_threshold:
            act = self.model.sample_action()
            act = {
                k: v for k, v in act.items() if k in hyperparams.ALLOWED_META_ACTIONS
            }
            logp = torch.tensor(0.0, device=self.device)
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
            if k in hyperparams.ALLOWED_META_ACTIONS:
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
            if k in hyperparams.ALLOWED_META_ACTIONS:
                low, high = ACTION_SPACE[k]
                scale = (high - low) / 2.0
                act[k] = float(sample[i].item() * scale)
                log_prob_gauss = -0.5 * ((sample[i] - mu[i]) ** 2) / (
                    std[i] ** 2
                ) - torch.log(std[i] * math.sqrt(2 * math.pi))
                logp = logp + log_prob_gauss

        # allow only one hyper-parameter mutation per step
        if act:
            k_only = _random.choice(list(act.keys()))
            act = {k_only: act[k_only]}

        self.prev_logits = (p_logits, b_logits, g_mean)
        self.prev_logprob = logp.detach()
        self.prev_action_idx = action_idx
        filtered = {}
        freeze = False
        if self.ensemble is not None:
            freeze = bool(getattr(self.ensemble.hp, "freeze_features", False))
        for action_name, val in act.items():
            if freeze and (
                action_name.startswith("toggle_")
                or action_name.endswith("_period_delta")
            ):
                continue
            if action_name not in hyperparams.ALLOWED_META_ACTIONS:
                continue
            filtered[action_name] = val
        act = filtered
        return act, logp.detach(), value

    def update(self, state_np, action_idx, reward, next_state_np, logprob, value):
        """Update the meta policy using PPO."""

        self.batch_buffer.append(
            (state_np, action_idx, reward, next_state_np, logprob, value)
        )
        if len(self.batch_buffer) < 10:
            return
        state_np, action_idx, reward, next_state_np, logprob, value = (
            self.batch_buffer.pop(0)
        )

        s = torch.as_tensor(state_np, dtype=torch.float32).unsqueeze(0)
        ns = torch.as_tensor(next_state_np, dtype=torch.float32).unsqueeze(0)
        s, ns = self._to_device(s, ns)
        out = self.model(s)
        if len(out) == 4:
            p_logits, _, _, val_s = out
        else:
            p_logits, val_s = out

        # clamp logits and replace NaN/Inf before constructing ``Categorical``
        p_logits = p_logits.clamp(-10.0, 10.0)
        p_logits = torch.nan_to_num(p_logits, nan=0.0, posinf=10.0, neginf=-10.0)
        dist = torch.distributions.Categorical(logits=p_logits)
        a = (
            torch.tensor([action_idx], device=self.device)
            if action_idx is not None
            else dist.sample()
        )
        lp_s = dist.log_prob(a)
        if not hasattr(self, "prev_logits"):
            self.prev_logits = (
                p_logits.detach(),
                torch.zeros_like(p_logits),
                torch.zeros_like(p_logits),
            )
            logprob = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            out_ns = self.model(ns)
            val_ns = out_ns[-1]
        # Use raw reward without EMA shaping
        target = reward + self.gamma * val_ns.item()
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

        if kl.item() < 0.02:
            self.low_kl_count += 1
            if self.low_kl_count >= 3 and self.entropy_beta > 3e-4:
                self.entropy_beta = 3e-4
        else:
            self.low_kl_count = 0

        # KL penalty removed for simplicity
        loss_p = pg_loss + (-self.entropy_beta * entropy)
        loss_v = F.mse_loss(val_s, torch.tensor([target], device=val_s.device))
        loss = loss_p + 0.5 * loss_v
        self.opt.zero_grad()
        if loss.requires_grad:
            loss.backward()
        else:
            for p in self.model.parameters():
                if p.requires_grad:
                    p.grad = torch.zeros_like(p)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        logging.debug("PPO_GRAD_NORM %.6f", float(grad_norm))
        self.opt.step()
        if self.scheduler is not None:
            try:
                if getattr(self.scheduler, "step_num", 0) < getattr(
                    self.scheduler, "total_steps", 0
                ):
                    self.scheduler.step()
                    self.scheduler.step_num = getattr(self.scheduler, "step_num", 0) + 1
            except ValueError as e:
                logging.warning("LR scheduler skipped: %s", e)
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

        filtered = {}
        freeze = bool(getattr(hp, "freeze_features", False))
        for action_name, val in act.items():
            if freeze and (
                action_name.startswith("toggle_")
                or action_name.endswith("_period_delta")
            ):
                continue
            if action_name not in hyperparams.ALLOWED_META_ACTIONS:
                continue
            filtered[action_name] = val
        act = filtered

        if act.get("toggle_sma"):
            indicator_hp.use_sma = not indicator_hp.use_sma
            logging.info("TOGGLE SMA -> %s", "ON" if indicator_hp.use_sma else "OFF")
            if self.ensemble is not None and "toggle_sma" in hyperparams.TOGGLE_INDEX:
                self.ensemble.feature_mask[hyperparams.TOGGLE_INDEX["toggle_sma"]].xor_(
                    1
                )
        if act.get("toggle_rsi"):
            indicator_hp.use_rsi = not indicator_hp.use_rsi
            logging.info("TOGGLE RSI -> %s", "ON" if indicator_hp.use_rsi else "OFF")
            if self.ensemble is not None and "toggle_rsi" in hyperparams.TOGGLE_INDEX:
                self.ensemble.feature_mask[hyperparams.TOGGLE_INDEX["toggle_rsi"]].xor_(
                    1
                )
        if act.get("toggle_macd"):
            indicator_hp.use_macd = not indicator_hp.use_macd
            logging.info("TOGGLE MACD -> %s", "ON" if indicator_hp.use_macd else "OFF")
            if self.ensemble is not None and "toggle_macd" in hyperparams.TOGGLE_INDEX:
                self.ensemble.feature_mask[
                    hyperparams.TOGGLE_INDEX["toggle_macd"]
                ].xor_(1)
        if act.get("toggle_atr"):
            indicator_hp.use_atr = not indicator_hp.use_atr
            logging.info("TOGGLE ATR -> %s", "ON" if indicator_hp.use_atr else "OFF")
            if self.ensemble is not None and "toggle_atr" in hyperparams.TOGGLE_INDEX:
                self.ensemble.feature_mask[hyperparams.TOGGLE_INDEX["toggle_atr"]].xor_(
                    1
                )
        if act.get("toggle_vortex"):
            indicator_hp.use_vortex = not indicator_hp.use_vortex
            logging.info(
                "TOGGLE VORTEX -> %s",
                "ON" if indicator_hp.use_vortex else "OFF",
            )
            if (
                self.ensemble is not None
                and "toggle_vortex" in hyperparams.TOGGLE_INDEX
            ):
                self.ensemble.feature_mask[
                    hyperparams.TOGGLE_INDEX["toggle_vortex"]
                ].xor_(1)
        if act.get("toggle_cmf"):
            indicator_hp.use_cmf = not indicator_hp.use_cmf
            logging.info("TOGGLE CMF -> %s", "ON" if indicator_hp.use_cmf else "OFF")
            if self.ensemble is not None and "toggle_cmf" in hyperparams.TOGGLE_INDEX:
                self.ensemble.feature_mask[hyperparams.TOGGLE_INDEX["toggle_cmf"]].xor_(
                    1
                )
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
                logging.info(
                    "TOGGLE %s -> %s",
                    key.split("_")[1].upper(),
                    "ON" if not cur else "OFF",
                )
                if self.ensemble is not None and key in hyperparams.TOGGLE_INDEX:
                    idx = hyperparams.TOGGLE_INDEX[key]
                    self.ensemble.feature_mask[idx].xor_(1)

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

        if self.ensemble is not None:
            for opt in self.ensemble.optimizers:
                group = opt.param_groups[0]
                if "lr" in act:
                    old = group["lr"]
                    group["lr"] = hyperparams.mutate_lr(old, float(act["lr"]))
                if "d_lr" in act:
                    old = group["lr"]
                    group["lr"] = hyperparams.mutate_lr(old, float(act["d_lr"]))
                if "wd" in act:
                    old_wd = group.get("weight_decay", 0.0)
                    group["weight_decay"] = hyperparams.mutate_lr(
                        old_wd, float(act["wd"])
                    )
                if "d_wd" in act:
                    old_wd = group.get("weight_decay", 0.0)
                    group["weight_decay"] = hyperparams.mutate_lr(
                        old_wd, float(act["d_wd"])
                    )
            # sync learning rate and weight decay with hyperparams and globals
            hp.learning_rate = self.ensemble.optimizers[0].param_groups[0]["lr"]
            hp.weight_decay = (
                self.ensemble.optimizers[0].param_groups[0].get("weight_decay", 0.0)
            )
            G.global_lr = hp.learning_rate
            G.global_wd = hp.weight_decay

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

        # build full parameter snapshot
        curr_state = {
            "SL_multiplier": hp.sl,
            "TP_multiplier": hp.tp,
            "long_frac": hp.long_frac,
            "short_frac": hp.short_frac,
            "lr": G.global_lr,
            "wd": G.global_wd,
            "sma_period": ind.sma_period,
            "sma_active": ind.use_sma,
            "rsi_period": ind.rsi_period,
            "rsi_active": ind.use_rsi,
            "macd_fast": ind.macd_fast,
            "macd_slow": ind.macd_slow,
            "macd_signal": ind.macd_signal,
            "macd_active": ind.use_macd,
            "atr_period": ind.atr_period,
            "atr_active": ind.use_atr,
            "vortex_period": ind.vortex_period,
            "vortex_active": ind.use_vortex,
            "cmf_period": ind.cmf_period,
            "cmf_active": ind.use_cmf,
            "ema_period": ind.ema_period,
            "ema_active": ind.use_ema,
            "donchian_period": ind.donchian_period,
            "donchian_active": ind.use_donchian,
            "kijun_period": ind.kijun_period,
            "kijun_active": ind.use_kijun,
            "tenkan_period": ind.tenkan_period,
            "tenkan_active": ind.use_tenkan,
            "displacement": ind.displacement,
            "disp_active": ind.use_displacement,
        }

        import json as _json

        global _prev_param_state
        if curr_state != _prev_param_state:
            logging.info("CURRENT_PARAMS %s", _json.dumps(curr_state, sort_keys=True))
            G.global_ai_adjustments_log += "\nCURRENT_PARAMS " + _json.dumps(
                curr_state, sort_keys=True
            )
            _prev_param_state = curr_state
            active_list: list[str] = []
            if curr_state.get("sma_active"):
                active_list.append(f"SMA(p{curr_state.get('sma_period')})")
            if curr_state.get("rsi_active"):
                active_list.append(f"RSI(p{curr_state.get('rsi_period')})")
            if curr_state.get("macd_active"):
                active_list.append(
                    f"MACD(f{curr_state.get('macd_fast')}/s{curr_state.get('macd_slow')}/sig{curr_state.get('macd_signal')})"
                )
            if curr_state.get("atr_active"):
                active_list.append(f"ATR(p{curr_state.get('atr_period')})")
            if curr_state.get("vortex_active"):
                active_list.append(f"VORTEX(p{curr_state.get('vortex_period')})")
            if curr_state.get("cmf_active"):
                active_list.append(f"CMF(p{curr_state.get('cmf_period')})")
            if curr_state.get("ema_active"):
                active_list.append(f"EMA(p{curr_state.get('ema_period')})")
            if curr_state.get("donchian_active"):
                active_list.append(f"DONCHIAN(p{curr_state.get('donchian_period')})")
            if curr_state.get("kijun_active"):
                active_list.append(f"KIJUN(p{curr_state.get('kijun_period')})")
            if curr_state.get("tenkan_active"):
                active_list.append(f"TENKAN(p{curr_state.get('tenkan_period')})")
            if curr_state.get("disp_active"):
                active_list.append(f"DISP(p{curr_state.get('displacement')})")
            logging.info(
                "ACTIVE_INDICATORS now: %s",
                ", ".join(active_list) if active_list else "None",
            )

        if (
            G.global_composite_reward is not None
            and G.global_composite_reward > G.global_best_composite_reward
        ):
            G.global_best_composite_reward = G.global_composite_reward
            G.global_best_params = curr_state.copy()
            G.global_best_lr = G.global_lr
            G.global_best_wd = G.global_wd

    @classmethod
    def reset_policy(cls) -> None:
        """Clear cached policy state for new folds."""
        inst = getattr(cls, "_last_instance", None)
        if inst is None:
            return
        inst.steps = 0
        inst.last_improvement = 0
        inst.batch_buffer.clear()
        inst.prev_logits = None
        inst.prev_logprob = None
        inst.prev_action_idx = None
        inst.opt.state.clear()


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
        [
            prev_r,
            best_r,
            st_sharpe / 5,
            abs(st_dd),
            st_trades / 500,
            st_days / 365,
        ],
        dtype=np.float32,
    )

    prev_profit_factor = getattr(G, "global_profit_factor", 0.0)

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
                [
                    curr_r,
                    b_r,
                    st_sharpe / 5,
                    abs(st_dd),
                    st_trades / 500,
                    st_days / 365,
                ],
                dtype=np.float32,
            )

            state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
            act, logp, val_s = agent.pick_action(state)
            filtered = {}
            freeze = bool(getattr(hp, "freeze_features", False))
            for action_name, val in act.items():
                if freeze and (
                    action_name.startswith("toggle_")
                    or action_name.endswith("_period_delta")
                ):
                    continue
                if action_name not in hyperparams.ALLOWED_META_ACTIONS:
                    continue
                filtered[action_name] = val
            act = filtered
            with G.model_lock:
                agent.apply_action(hp, indicator_hp, act)
                if dataset is not None:
                    dataset.apply_feature_mask(indicator_hp)
                # models stay fixed at FEATURE_DIM; missing indicators are padded
                if ensemble.models[0].input_dim != FEATURE_DIM:
                    ensemble.rebuild_models(FEATURE_DIM)

            G.status_sleep("Meta agent sleeping", "", interval)

            curr2 = G.global_composite_reward if G.global_composite_reward else 0.0
            reward_val = curr2
            bonus = 0.0
            if new_state[2] > state[2]:
                bonus += 0.1
            if new_state[3] < state[3]:
                bonus += 0.1
            curr_pf = getattr(G, "global_profit_factor", None)
            if (
                curr_pf is not None
                and isinstance(curr_pf, (int, float))
                and curr_pf > prev_profit_factor
            ):
                bonus += 0.05
            val_loss = getattr(G, "global_validation_loss", None)
            if (
                isinstance(val_loss, list)
                and len(val_loss) > 1
                and val_loss[-1] < val_loss[-2]
            ):
                bonus += 0.05
            reward_val += bonus
            agent.update(state, 0, reward_val, new_state, logp, val_s)
            prev_profit_factor = curr_pf if curr_pf is not None else prev_profit_factor
            rew_delta = reward_val

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
