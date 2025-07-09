"""Transformer model definition for Artibot."""

# ruff: noqa: F403, F405

import logging
import numpy as np
import torch
import torch.nn as nn

import artibot.globals as G
from .constants import FEATURE_DIMENSION
from .dataset import TradeParams
from .utils import attention_entropy


def _ver_tuple(ver: str) -> tuple[int, int]:
    major, minor, *_ = ver.split(".")
    return int(major), int(minor)


logger = logging.getLogger(__name__)


###############################################################################
# PositionalEncoding
###############################################################################
class PositionalEncoding(nn.Module):
    """Add positional encoding on the fly for any feature dimension."""

    def __init__(self, d_model: int | None = None, max_len: int = 5000) -> None:
        super().__init__()
        self.max_len = max_len
        self._dim = None
        if d_model is not None:
            self._build_pe(d_model)

    def _build_pe(self, d_model: int) -> None:
        pe = torch.zeros(self.max_len, d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        if hasattr(self, "pe"):
            self.pe.resize_(pe.size())
            self.pe.copy_(pe)
        else:
            self.register_buffer("pe", pe)
        self._dim = d_model

    def forward(self, x):
        if x.size(-1) != getattr(self, "pe", torch.empty(0)).size(-1):
            self._build_pe(x.size(-1))
        return x + self.pe[:, : x.size(1)]


###############################################################################
# TradingModel (Transformer)
###############################################################################
# (6) Increase model capacity & dropout. For instance:
# hidden_size=128, dropout=0.4, nhead=4, num_layers=4
class TradingModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_classes: int = 3,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if input_size != FEATURE_DIMENSION:
            raise ValueError("Feature dimension mismatch")
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.d_model = FEATURE_DIMENSION
        self.input_dim = input_size
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.fc_proj = nn.Linear(self.d_model, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes + 4)

    def forward(self, x):
        if x.size(-1) != self.d_model:
            raise ValueError("Feature dimension mismatch")
        x = self.pos_encoder(x)
        # inspect attention from the first encoder layer
        # Use a detached tensor so fastpath inference in ``no_grad`` does not mutate ``x``
        with torch.no_grad():
            x_detached = x.detach()
            attn_output, attn_weights = self.transformer_encoder.layers[0].self_attn(
                x_detached,
                x_detached,
                x_detached,
                need_weights=True,
                average_attn_weights=False,
            )
        p = attn_weights.mean(dim=(0, 1))
        entropy = attention_entropy(p)
        max_prob = p.max().item()
        self.last_entropy = float(entropy)
        self.last_max_prob = float(max_prob)
        logger.debug({"event": "ATTN_STATS", "entropy": entropy, "max_prob": max_prob})
        attn_mean = p.mean().item()
        try:
            G.global_attention_weights_history.append(attn_mean)
            G.global_attention_entropy_history.append(entropy)
        except Exception:
            pass
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc_proj(x)
        x = self.layernorm(x)
        context = self.dropout(x)
        out_all = self.fc(context)
        w = torch.nan_to_num(p.unsqueeze(0))
        try:
            G.global_last_attention = w.squeeze(0).cpu().tolist()
        except Exception:
            pass

        # Risk fraction, SL/TP are scaled
        logits = out_all[:, :3]
        logits = torch.nan_to_num(logits)
        logits = logits.clamp(min=-10.0, max=10.0)
        risk_frac = 0.001 + 0.499 * torch.sigmoid(out_all[:, 3])
        sl_mult = 0.5 + 9.5 * torch.sigmoid(out_all[:, 4])
        tp_mult = 0.5 + 9.5 * torch.sigmoid(out_all[:, 5])
        # The reward head can explode early in training which often results in
        # NaNs propagating through the loss.  Scale the raw value and strip any
        # non finite numbers to keep the optimisation stable.
        pred_reward = (
            out_all[:, 6] if out_all.shape[1] > 6 else torch.zeros_like(out_all[:, 0])
        )
        pred_reward = 0.1 * torch.nan_to_num(pred_reward)
        return logits, TradeParams(risk_frac, sl_mult, tp_mult, w), pred_reward


class TradingTransformer(TradingModel):
    """Backward-compatible alias for :class:`TradingModel`."""

    pass


###############################################################################


def build_model(*args, **kwargs) -> TradingModel:
    """Return a ``TradingModel`` compiled on PyTorch 2+."""

    model = TradingModel(*args, **kwargs)
    if hasattr(torch, "compile") and _ver_tuple(torch.__version__) >= (2, 0):
        try:  # pragma: no cover - optional feature
            model = torch.compile(model)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - compile may fail
            logger.info("torch.compile failed: %s", exc)
    return model


###############################################################################
