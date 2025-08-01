"""Lightweight market regime encoder using an LSTM autoencoder."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


_LOG = logging.getLogger(__name__)


class _LSTMAutoEncoder(nn.Module):
    """Simple LSTM autoencoder used for feature compression."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc_out, _ = self.encoder(x)
        h = enc_out[:, -1]
        z = self.latent(h)
        dec_in = z.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_in)
        recon = self.out(dec_out)
        return recon, z


class RegimeEncoder(nn.Module):
    """Learn latent market regimes with a small autoencoder."""

    def __init__(
        self,
        seq_len: int = 32,
        input_dim: int = 3,
        hidden_dim: int = 16,
        latent_dim: int = 8,
        n_regimes: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_regimes = n_regimes
        self.autoencoder = _LSTMAutoEncoder(input_dim, hidden_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, n_regimes)
        self.kmeans: Optional[KMeans] = None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _feature_windows(prices: np.ndarray, window: int) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        if prices.ndim != 1:
            raise ValueError("prices must be 1-D array")
        log_ret = np.diff(np.log(prices), prepend=np.log(prices[0]))
        vol = (
            np.convolve(log_ret**2, np.ones(20), "full")[: len(log_ret)] / 20
        ) ** 0.5
        sma_short = np.convolve(prices, np.ones(5) / 5.0, "same")
        sma_long = np.convolve(prices, np.ones(20) / 20.0, "same")
        trend = (sma_short - sma_long) / (prices + 1e-9)
        feats = np.column_stack([log_ret, vol, trend])
        num = len(feats) - window + 1
        if num <= 0:
            return np.empty((0, window, feats.shape[1]), dtype=np.float32)
        out = np.lib.stride_tricks.sliding_window_view(feats, (window, feats.shape[1]))
        return out.reshape(num, window, feats.shape[1]).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train_unsupervised(self, prices: np.ndarray, epochs: int = 5) -> None:
        """Train the encoder on a price series without supervision."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        windows = self._feature_windows(prices, self.seq_len)
        if len(windows) == 0:
            raise ValueError("not enough data for the chosen sequence length")
        ds = TensorDataset(torch.tensor(windows))
        dl = DataLoader(ds, batch_size=32, shuffle=True)
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.to(device)
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in dl:
                batch = batch.to(device)
                opt.zero_grad()
                recon, _ = self.autoencoder(batch)
                loss = nn.functional.mse_loss(recon, batch)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * batch.size(0)
            _LOG.info("[RegimeEncoder] epoch %d recon loss %.6f", epoch, epoch_loss / len(ds))
        # cluster latent space
        self.eval()
        with torch.no_grad():
            all_z = []
            for (batch,) in dl:
                batch = batch.to(device)
                _, z = self.autoencoder(batch)
                all_z.append(z.cpu())
            Z = torch.cat(all_z).numpy()
        self.kmeans = KMeans(n_clusters=self.n_regimes, n_init=10, random_state=42)
        labels = self.kmeans.fit_predict(Z)
        cl_ds = TensorDataset(torch.tensor(Z, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
        cl_dl = DataLoader(cl_ds, batch_size=32, shuffle=True)
        opt = torch.optim.Adam(self.classifier.parameters(), lr=5e-4)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in cl_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = self.classifier(x)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * x.size(0)
            _LOG.info(
                "[RegimeEncoder] classifier epoch %d loss %.6f",
                epoch,
                epoch_loss / len(cl_ds),
            )
        self.to("cpu")

    @torch.no_grad()
    def encode_sequence(self, prices: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Return regime probabilities for each window in the series."""
        if self.kmeans is None:
            raise RuntimeError("encoder not trained")
        windows = self._feature_windows(prices, self.seq_len)
        if len(windows) == 0:
            return np.empty((0, self.n_regimes), dtype=np.float32)
        dl = DataLoader(torch.tensor(windows), batch_size=batch_size)
        self.eval()
        device = next(self.parameters()).device
        probs = []
        for batch in dl:
            batch = batch.to(device)
            _, z = self.autoencoder(batch)
            logits = self.classifier(z)
            probs.append(nn.functional.softmax(logits, dim=1).cpu())
        return torch.cat(probs).numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str = "encoder.pt") -> None:
        """Save model weights to disk."""
        state = {
            "model": self.state_dict(),
            "kmeans": None if self.kmeans is None else self.kmeans.cluster_centers_,
        }
        torch.save(state, path)

    def load(self, path: str = "encoder.pt") -> None:
        """Load model weights from disk."""
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state["model"])
        if state.get("kmeans") is not None:
            self.kmeans = KMeans(n_clusters=self.n_regimes)
            self.kmeans.cluster_centers_ = state["kmeans"]


__all__ = ["RegimeEncoder"]
