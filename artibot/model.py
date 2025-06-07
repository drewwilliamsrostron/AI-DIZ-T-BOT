"""Transformer model definition for Artibot."""

# ruff: noqa: F403, F405

import numpy as np
import torch
import torch.nn as nn

from .dataset import TradeParams


###############################################################################
# PositionalEncoding
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


###############################################################################
# TradingModel (Transformer)
###############################################################################
# (6) Increase model capacity & dropout. For instance:
# hidden_size=128, dropout=0.4, nhead=4, num_layers=4
class TradingModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_classes=3, dropout=0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.pos_encoder = PositionalEncoding(d_model=input_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.fc_proj = nn.Linear(input_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.attn = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes + 4)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = x.transpose(0, 1).contiguous()
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        x = self.fc_proj(x)
        x = self.layernorm(x)
        raw_attn = self.attn(x).unsqueeze(1)
        w = torch.softmax(raw_attn, dim=1)
        context = self.dropout(x)
        out_all = self.fc(context)

        # Risk fraction, SL/TP are scaled
        logits = out_all[:, :3]
        logits = torch.nan_to_num(logits).clamp_(-10.0, 10.0)
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


###############################################################################
