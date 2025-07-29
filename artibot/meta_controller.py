"""Simplified meta reinforcement learning with a replay buffer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Tuple

import numpy as np
from torch.utils.tensorboard import SummaryWriter


@dataclass
class MetaConfig:
    """Configuration for :class:`MetaTransformerRL`."""

    buffer: int = 1024
    batch: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 4


class MetaTransformerRL:
    """Tiny PPO agent used to tune high level settings."""

    def __init__(
        self, state_dim: int, action_dim: int, cfg: dict | None = None
    ) -> None:
        meta = (cfg or {}).get("META", {})
        self.cfg = MetaConfig(
            buffer=meta.get("buffer", 1024),
            batch=meta.get("batch", 256),
            gamma=meta.get("gamma", 0.99),
            gae_lambda=meta.get("gae_lambda", 0.95),
            epochs=meta.get("epochs", 4),
        )
        self.replay: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=self.cfg.buffer
        )
        # simple linear models represented by weight matrices
        self.policy_w = np.zeros((state_dim, action_dim), dtype=np.float32)
        self.value_w = np.zeros((state_dim, 1), dtype=np.float32)
        self.writer = SummaryWriter()
        self.step = 0

    # ------------------------------------------------------------------
    def store_transition(
        self,
        state: Iterable[float],
        action: int,
        reward: float,
        next_state: Iterable[float],
        done: bool,
    ) -> None:
        """Add one transition to the replay buffer."""

        self.replay.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    # ------------------------------------------------------------------
    def _forward(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        logits = states @ self.policy_w
        values = states @ self.value_w
        return logits, values.squeeze(1)

    # ------------------------------------------------------------------
    def learn(self) -> None:
        """Update policy and value weights using a simple PPO routine."""

        if len(self.replay) < self.cfg.batch:
            return

        idx = np.random.choice(len(self.replay), self.cfg.batch, replace=False)
        batch = [self.replay[i] for i in idx]
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))

        logits, values = self._forward(states)
        _, next_values = self._forward(next_states)

        deltas = rewards + self.cfg.gamma * next_values * (1 - dones) - values
        adv = np.zeros_like(deltas)
        gae = 0.0
        for t in range(len(deltas) - 1, -1, -1):
            gae = (
                deltas[t] + self.cfg.gamma * self.cfg.gae_lambda * (1 - dones[t]) * gae
            )
            adv[t] = gae
        returns = adv + values

        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        log_probs_old = np.log(probs[np.arange(len(actions)), actions])

        # no real optimisation; compute losses for logging only
        for _ in range(self.cfg.epochs):
            ratio = np.exp(log_probs_old - log_probs_old)  # always 1.0
            surr1 = ratio * adv
            surr2 = np.clip(ratio, 1 - 0.2, 1 + 0.2) * adv
            policy_loss = -np.mean(np.minimum(surr1, surr2))
            value_loss = 0.5 * np.mean((returns - values) ** 2)
            entropy = -np.mean(np.sum(probs * np.log(probs + 1e-8), axis=1))

        self.writer.add_scalar("meta/policy_loss", float(policy_loss), self.step)
        self.writer.add_scalar("meta/value_loss", float(value_loss), self.step)
        self.writer.add_scalar("meta/entropy", float(entropy), self.step)
        self.step += 1
