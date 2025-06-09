"""Meta reinforcement learning agent controlling training hyperparameters."""

# ruff: noqa: F403, F405

import artibot.globals as G
from .model import PositionalEncoding
from .dataset import IndicatorHyperparams

import itertools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traceback


###############################################################################
# NEW: A bigger action space that includes adjusting RSI, SMA, MACD + threshold
###############################################################################
class TransformerMetaAgent(nn.Module):
    def __init__(self):
        super().__init__()
        # (1) Enhanced meta-agent exploration: bigger sets
        # (5) threshold_space => optional
        # Simplified action space
        self.lr_space = [-0.3, -0.1, 0.1, 0.3]  # More aggressive adjustments
        self.wd_space = [-0.3, 0.0, 0.3]
        self.rsi_space = [-5, 0, 5]
        self.sma_space = [-5, 0, 5]
        self.macd_fast_space = [-5, 0, 5]
        self.macd_slow_space = [-5, 0, 5]
        self.macd_sig_space = [-3, 0, 3]
        self.threshold_space = [-0.0001, 0.0, 0.0001]  # Bigger threshold steps

        # Remove nested loops - use random sampling instead
        self.action_space = list(
            itertools.product(
                self.lr_space,
                self.wd_space,
                self.rsi_space,
                self.sma_space,
                self.macd_fast_space,
                self.macd_slow_space,
                self.macd_sig_space,
                self.threshold_space,
            )
        )
        # Randomly sample 1000 possible combinations instead of full cartesian product
        random.shuffle(self.action_space)
        self.action_space = self.action_space[:1000]

        # (4) State includes: [curr_reward, best_reward, sharpe, dd, trades, days_in_profit]
        self.state_dim = 6
        self.n_actions = len(self.action_space)

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
    def __init__(self, ensemble, lr=1e-3):
        self.model = TransformerMetaAgent()
        # expose the underlying action space so other methods don't
        # need to reach into the model attribute. Without this the
        # ``pick_action`` method fails with ``AttributeError``.
        self.action_space = self.model.action_space
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = 0.95
        self.ensemble = ensemble

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

    def pick_action(self, state_np):
        # scheduled exploration
        self.steps += 1
        current_eps = max(self.eps_end, self.eps_start * (self.eps_decay**self.steps))
        noise_scale = max(0.1, 1.0 - self.steps / 10000)  # added random noise

        s = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pol, val = self.model(s)
            dist = torch.distributions.Categorical(logits=pol)
        if random.random() < current_eps:
            # Add noise to action selection
            a_idx = np.random.choice(len(self.action_space))
            # Apply Gaussian noise to action parameters
            selected_action = list(self.action_space[a_idx])
            selected_action = [
                x + np.random.normal(0, noise_scale) for x in selected_action
            ]
            a_idx = self._find_nearest_action(selected_action)
            lp = dist.log_prob(torch.tensor([a_idx]))
        else:
            a_idx = dist.sample().item()
            lp = dist.log_prob(torch.tensor([a_idx]))
        return a_idx, lp, val.item()

    def _find_nearest_action(self, candidate):
        """Return the index of the closest action in the action space."""
        if not isinstance(candidate, np.ndarray):
            candidate = np.array(candidate, dtype=float)
        actions = np.array(self.action_space, dtype=float)
        dists = np.linalg.norm(actions - candidate, axis=1)
        return int(np.argmin(dists))

    def update(self, state_np, action_idx, reward, next_state_np, logprob, value):
        s = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        ns = torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0)
        pol_s, val_s = self.model(s)
        dist = torch.distributions.Categorical(logits=pol_s)
        lp_s = dist.log_prob(torch.tensor([action_idx]))
        with torch.no_grad():
            pol_ns, val_ns = self.model(ns)
        target = reward + self.gamma * val_ns.item()
        advantage = target - val_s.item()
        loss_p = -lp_s * advantage
        loss_v = 0.5 * (val_s - target) ** 2
        loss = loss_p + loss_v
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if reward > 0:
            self.last_improvement = 0
        else:
            self.last_improvement += 1

    def apply_action(self, action_idx):
        # decode
        (lr_adj, wd_adj, rsi_adj, sma_adj, mf_adj, ms_adj, sig_adj, thr_adj) = (
            self.action_space[action_idx]
        )
        # 1) LR/WD
        old_lr = self.ensemble.optimizers[0].param_groups[0]["lr"]
        new_lr = old_lr * (1 + lr_adj)
        new_lr = max(1e-6, min(new_lr, 1e-1))

        old_wd = self.ensemble.optimizers[0].param_groups[0].get("weight_decay", 0)
        new_wd = old_wd * (1 + wd_adj)
        new_wd = max(0, min(new_wd, 1e-2))

        for opt_ in self.ensemble.optimizers:
            for grp in opt_.param_groups:
                grp["lr"] = new_lr
                grp["weight_decay"] = new_wd

        # 2) indicator hyperparams
        old_hp = self.ensemble.indicator_hparams
        new_rsi = old_hp.rsi_period + rsi_adj
        new_sma = old_hp.sma_period + sma_adj
        new_mf = old_hp.macd_fast + mf_adj
        new_ms = old_hp.macd_slow + ms_adj
        new_sig = old_hp.macd_signal + sig_adj
        # 5) also let meta-agent change threshold
        new_threshold = G.GLOBAL_THRESHOLD + thr_adj

        self.ensemble.indicator_hparams = IndicatorHyperparams(
            rsi_period=new_rsi,
            sma_period=new_sma,
            macd_fast=new_mf,
            macd_slow=new_ms,
            macd_signal=new_sig,
        )
        # If you want the dataset to re-init => we would do self.ensemble.dynamic_threshold = new_threshold
        # But for demonstration, we won't forcibly re-load the dataset now.

        return (
            new_lr,
            new_wd,
            new_rsi,
            new_sma,
            new_mf,
            new_ms,
            new_sig,
            new_threshold,
        )


###############################################################################
# meta_control_loop
###############################################################################
def meta_control_loop(ensemble, dataset, agent, interval=5.0):
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

            G.set_status("Meta agent updating", "")

            a_idx, logp, val_s = agent.pick_action(state)
            with G.model_lock, torch.no_grad():
                (
                    new_lr,
                    new_wd,
                    nrsi,
                    nsma,
                    nmacdf,
                    nmacds,
                    nmacdsig,
                    nthr,
                ) = agent.apply_action(a_idx)

            G.status_sleep("Meta agent sleeping", "", interval)

            curr2 = G.global_composite_reward if G.global_composite_reward else 0.0
            rew_delta = curr2 - curr_r
            agent.update(state, a_idx, rew_delta, new_state, logp, val_s)

            summary = (
                f"Meta Update => s:{state} => a_idx={a_idx}, r:{rew_delta:.2f}\n"
                f" newLR={new_lr:.2e}, newWD={new_wd:.2e}, rsi={nrsi}, sma={nsma}, "
                f"macdF={nmacdf}, macdS={nmacds}, macdSig={nmacdsig}, threshold={nthr:.5f}"
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
