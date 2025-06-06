#!/usr/bin/env python3
"""
Complex AI Trading + Continuous Training + Robust Back-test Each Epoch
+ Live Phemex + Tkinter GUI + Meta-Control (Transformer-RL)

This file is self-contained – run it once, it installs required libs and
launches the GUI / training loop.  NOT INVESTMENT ADVICE; demo purposes only.
"""

###############################################################################
# 1 – One-time dependency installer (runs only if modules missing)
###############################################################################
import sys, subprocess, importlib, json, os, itertools, math, logging, random, \
       threading, queue, time, datetime, re, itertools
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", *args])

def install_dependencies():
    # CUDA-enabled torch if possible
    try:
        import torch
        if torch.cuda.is_available():
            pass
        else:
            print("Torch found but CUDA not available – re-installing with cu118 …")
            _pip("install", "torch==2.2.1+cu118",
                 "--extra-index-url", "https://download.pytorch.org/whl/cu118")
            _pip("install", "torchvision==0.17.1+cu118",
                 "--extra-index-url", "https://download.pytorch.org/whl/cu118")
            _pip("install", "torchaudio==2.2.2+cu118",
                 "--extra-index-url", "https://download.pytorch.org/whl/cu118")
    except ModuleNotFoundError:
        print("Installing torch (+CUDA) …")
        _pip("install", "torch==2.2.1+cu118",
             "--extra-index-url", "https://download.pytorch.org/whl/cu118")
        _pip("install", "torchvision==0.17.1+cu118",
             "--extra-index-url", "https://download.pytorch.org/whl/cu118")
        _pip("install", "torchaudio==2.2.2+cu118",
             "--extra-index-url", "https://download.pytorch.org/whl/cu118")

    other = ["openai", "ccxt", "pandas", "numpy", "matplotlib", "TA-Lib", "scikit-learn"]
    for pkg in other:
        mod = "talib" if pkg == "TA-Lib" else pkg
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            print(f"Installing {pkg} …")
            _pip("install", pkg)

install_dependencies()

###############################################################################
# 2 – Imports
###############################################################################
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import ccxt, talib, openai

###############################################################################
# 3 – Global hyper-parameters & runtime state (shared across threads / GUI)
###############################################################################
global_SL_multiplier   = 5
global_TP_multiplier   = 5
global_ATR_period      = 50
risk_fraction          = 0.03
GLOBAL_THRESHOLD       = 0.0001

# live metrics updated each epoch
(global_training_loss, global_validation_loss, global_backtest_profit,
 global_equity_curve,  global_best_equity_curve,
 global_attention_weights_history, global_trade_details) = ([] for _ in range(7))

(global_sharpe, global_max_drawdown, global_net_pct, global_num_trades,
 global_inactivity_penalty, global_composite_reward,
 global_days_without_trading, global_days_in_profit) = (0, 0, 0, 0, None, None, None, 0.0)

(global_best_sharpe, global_best_drawdown, global_best_net_pct,
 global_best_num_trades, global_best_inactivity_penalty,
 global_best_composite_reward, global_best_days_in_profit) = (0, 0, 0, 0, None, None, None)

global_best_lr = global_best_wd = None
global_yearly_stats_table = ""

# AI meta-controller logs
global_ai_adjustments_log = "No adjustments yet"
global_ai_adjustments      = ""
global_ai_confidence       = None
global_ai_epoch_count      = 0
global_current_prediction  = None

# live price feed container
global_phemex_data: list = []

# thread-safe queue used by CSV-training thread when ADAPT_TO_LIVE = True
live_bars_queue: "queue.Queue[list]" = queue.Queue()

###############################################################################
# 4 – Type helpers
###############################################################################
from typing import NamedTuple, Tuple

class TradeParams(NamedTuple):
    """Per-bar outputs returned by the model (for position sizing)."""
    risk_fraction: torch.Tensor
    sl_multiplier: torch.Tensor
    tp_multiplier: torch.Tensor
    attention    : torch.Tensor          # currently unused in back-test

class IndicatorHyperparams(NamedTuple):
    """Hyper-params the meta-controller is allowed to tweak on-line."""
    rsi_period  : int
    sma_period  : int
    macd_fast   : int
    macd_slow   : int
    macd_signal : int
###############################################################################
# 5 – CSV utilities & PyTorch dataset
###############################################################################
def load_csv_hourly(csv_path: str) -> list:
    """
    Load an hourly-resolution OHLCV CSV where price columns are *scaled* by 1e5.
    Expected header: unix, open, high, low, close, volume_btc (case-insensitive).
    The function returns a **sorted** list of rows:
        [timestamp, open, high, low, close, volume]
    All price fields are re-scaled back to float price (÷ 100 000).
    """
    if not os.path.isfile(csv_path):
        logging.warning("CSV file ‘%s’ not found.", csv_path)
        return []

    try:
        df = pd.read_csv(csv_path, sep=r"[,\t]+", engine="python",
                         skiprows=1, header=0)
    except Exception as exc:
        logging.warning("Error reading CSV: %s", exc)
        return []

    df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(" ", "_"))

    rows: list = []
    for _, r in df.iterrows():
        try:
            ts = int(r["unix"])
            if ts > 1e12:            # some providers use ms epoch
                ts //= 1000
            o = float(r["open"])  / 1e5
            h = float(r["high"])  / 1e5
            l = float(r["low"])   / 1e5
            c = float(r["close"]) / 1e5
            v = float(r["volume_btc"]) if "volume_btc" in df.columns else 0.0
            rows.append([ts, o, h, l, c, v])
        except Exception:
            continue

    return sorted(rows, key=lambda x: x[0])           # chronological

# ─────────────────────────────────────────────────────────────────────────────
class HourlyDataset(Dataset):
    """
    Converts raw OHLCV + indicator stacks into (seq_len × features) tensors and
    multi-class labels (0 BUY, 1 SELL, 2 HOLD) according to GLOBAL_THRESHOLD.
    """
    def __init__(self,
                 data: list,
                 seq_len: int = 24,
                 threshold: float = GLOBAL_THRESHOLD,
                 sma_period: int = 10) -> None:
        self.data       = data
        self.seq_len    = seq_len
        self.threshold  = threshold
        self.sma_period = sma_period
        self.samples, self.labels = self._preprocess()

    # --------------------------------------------------------------------- #
    def _preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        closes = np.array([row[4] for row in self.data], dtype=np.float64)

        # technical indicators
        sma  = np.convolve(closes,
                           np.ones(self.sma_period) / self.sma_period,
                           mode="same")
        rsi  = talib.RSI(closes, timeperiod=14)
        macd, _, _ = talib.MACD(closes)                              # fast-12/slow-26/sig-9

        feats = [list(r[1:6]) + [sma[i], rsi[i], macd[i]]
                 for i, r in enumerate(self.data)]
        feats = np.asarray(feats, dtype=np.float32)

        scaler       = StandardScaler()
        scaled_feats = scaler.fit_transform(feats)

        X, Y = [], []
        for i in range(self.seq_len, len(scaled_feats) - 1):
            window      = scaled_feats[i - self.seq_len: i]
            curr_close  = window[-1][3]
            next_close  = scaled_feats[i][3]
            ret         = (next_close - curr_close) / (curr_close + 1e-8)

            if   ret >  self.threshold: lbl = 0      # BUY
            elif ret < -self.threshold: lbl = 1      # SELL
            else:                        lbl = 2      # HOLD
            X.append(window)
            Y.append(lbl)

        return np.asarray(X), np.asarray(Y)

    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.samples)

    # --------------------------------------------------------------------- #
    def __getitem__(self, idx: int):
        sample = self.samples[idx].copy()
        if random.random() < 0.5:                                  # heavier augm.
            sample += np.random.normal(0, 0.02, sample.shape)
        return torch.tensor(sample), torch.tensor(self.labels[idx], dtype=torch.long)

###############################################################################
# 6 – Model architecture
###############################################################################
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (unchanged from ‘Attention is All
    You Need’).
    """
    def __init__(self, d_model: int, max_len: int = 5_000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2:                                            # odd dim
            pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# ─────────────────────────────────────────────────────────────────────────────
class TradingModel(nn.Module):
    """
    Transformer encoder → projection → head that outputs:
      • logits (3) for BUY/SELL/HOLD
      • risk-fraction, SL-multiplier, TP-multiplier (scaled sigmoid)
      • predicted reward (regression head)
    """
    def __init__(self,
                 input_size: int   = 8,
                 hidden_size: int  = 128,
                 num_classes: int  = 3,
                 dropout: float    = 0.4) -> None:
        super().__init__()
        self.pos_enc   = PositionalEncoding(d_model=input_size)

        encoder_layer  = nn.TransformerEncoderLayer(d_model=input_size,
                                                    nhead=4,
                                                    dim_feedforward=256,
                                                    dropout=dropout,
                                                    batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.proj      = nn.Linear(input_size, hidden_size)
        self.norm      = nn.LayerNorm(hidden_size)
        self.dropout   = nn.Dropout(dropout)

        self.attn_head = nn.Linear(hidden_size, 1)                 # (not used yet)
        self.out_head  = nn.Linear(hidden_size, num_classes + 4)

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, TradeParams, torch.Tensor]:
        """
        x: [batch, seq_len, input_size]
        returns:
            logits (B×3),
            TradeParams,
            predicted reward (B)
        """
        x = self.pos_enc(x)                  # add PE
        x = x.transpose(0, 1)                # seq_len × batch × feat for torch encoder
        x = self.transformer(x).mean(dim=0)  # pool
        x = self.dropout(self.norm(self.proj(x)))

        # auxiliary attention (not used for routing decisions)
        _attn = torch.softmax(self.attn_head(x).unsqueeze(1), dim=1)

        raw = self.out_head(x)
        logits      = raw[:, :3]
        risk_frac   = 0.001 + 0.499 * torch.sigmoid(raw[:, 3])
        sl_mult     = 0.5   + 9.5   * torch.sigmoid(raw[:, 4])
        tp_mult     = 0.5   + 9.5   * torch.sigmoid(raw[:, 5])
        pred_reward = raw[:, 6] if raw.shape[1] > 6 else torch.zeros_like(raw[:, 0])

        return logits, TradeParams(risk_frac, sl_mult, tp_mult, _attn), pred_reward

###############################################################################
# 7 – Helper analytics
###############################################################################
def compute_yearly_stats(equity_curve: list,
                         trades: list,
                         initial_balance: float = 100.0) -> Tuple[pd.DataFrame, str]:
    """
    Compute per-calendar-year Net %, Sharpe, Max-DD, trade count.
    Returns (DataFrame, formatted_string) — formatted for the Tkinter tab.
    """
    if not equity_curve:
        return pd.DataFrame(), "No data"

    eq = pd.DataFrame(equity_curve, columns=["ts", "balance"])
    eq["dt"] = pd.to_datetime(eq["ts"], unit="s")
    eq.set_index("dt", inplace=True)
    eq = eq.resample("1D").last().dropna()
    if eq.empty:
        return pd.DataFrame(), "No data"

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tdf["entry_dt"] = pd.to_datetime(tdf["entry_time"], unit="s")
        tdf.set_index("entry_dt", inplace=True)

    rows = []
    for yr, grp in eq.groupby(eq.index.year):
        if len(grp) < 2:
            continue
        start_bal, end_bal = grp["balance"].iloc[[0, -1]]
        net_pct = 100.0 * (end_bal - start_bal) / (start_bal or 1e-8)

        daily_ret = grp["balance"].pct_change().dropna()
        mu, sigma = daily_ret.mean(), daily_ret.std()
        sharpe    = (mu * 252) / (sigma * math.sqrt(252)) if sigma > 1e-12 else 0.0

        roll_max  = grp["balance"].cummax()
        mdd       = ((grp["balance"] - roll_max) / roll_max).min() if not grp.empty else 0.0

        n_trades  = len(tdf.loc[str(yr)]) if not tdf.empty and str(yr) in tdf.index.year.astype(str) else 0

        rows.append(dict(Year=yr, NetPct=net_pct, Sharpe=sharpe,
                         MaxDD=mdd, Trades=n_trades))

    if not rows:
        return pd.DataFrame(), "No yearly data"

    dfy = pd.DataFrame(rows).set_index("Year")
    return dfy, dfy.to_string(float_format=lambda x: f"{x: .2f}")

# ─────────────────────────────────────────────────────────────────────────────
def inactivity_exponential_penalty(gap_sec: int,
                                   max_penalty: float = 100.0) -> float:
    """Penalty grows 2× every 30-day inactivity block, capped at max_penalty."""
    month = 30 * 24 * 3600
    months = int(gap_sec // month)
    base = 0.01
    penalty = sum(min(base * 2 ** i, max_penalty) for i in range(months))
    return min(penalty, max_penalty)

# ─────────────────────────────────────────────────────────────────────────────
def compute_days_in_profit(equity_curve: list,
                           init_balance: float) -> float:
    """
    Time-integral of equity being ≥ initial balance.
    Return value in *days* (float).
    """
    if len(equity_curve) < 2:
        return 0.0

    seconds = 0.0
    for (t0, b0), (t1, b1) in zip(equity_curve[:-1], equity_curve[1:]):
        if (b0 >= init_balance) == (b1 >= init_balance):       # both same side
            if b0 >= init_balance:
                seconds += t1 - t0
        else:                                                  # crossing the line
            # linear interpolation to crossing point
            if b1 != b0:
                frac = (init_balance - b0) / (b1 - b0)
            else:
                frac = 0.0
            if b0 < init_balance < b1:         # crossing up
                seconds += (1 - frac) * (t1 - t0)
            elif b1 < init_balance < b0:       # crossing down
                seconds += frac * (t1 - t0)

    return seconds / 86_400.0

###############################################################################
# 8 – Robust, position-sizing back-test
###############################################################################
def robust_backtest(ensemble,
                    data_full: list) -> dict:
    """
    Deterministic vectorised back-test used every epoch for on-policy training
    feedback.  Includes leverage, slippage, commissions, inactivity penalties,
    funding, and composite reward calculation.
    """
    if len(data_full) < 24:
        return dict(net_pct=0.0, trades=0, effective_net_pct=0.0,
                    equity_curve=[])

    # exchange / strategy settings
    LEVERAGE        = 10
    MIN_HOLD        = 2 * 3600
    COMM            = 0.0001
    SLIPPAGE_PCT    = 0.0002
    FUNDING_RATE    = 0.0001

    # composite-reward weights
    α, β, γ, δ = 3.0, 0.5, 0.8, 0.1

    hp = ensemble.indicator_hparams
    rsi_p  = max(2,  min(hp.rsi_period, 50))
    sma_p  = max(2,  min(hp.sma_period, 100))
    fast   = max(2,  min(hp.macd_fast, hp.macd_slow - 1))
    slow   = max(fast + 1, min(hp.macd_slow, 200))
    sig    = max(1,  min(hp.macd_signal, 50))

    raw = np.asarray(data_full, dtype=np.float64)
    closes = raw[:, 4]

    sma   = np.convolve(closes, np.ones(sma_p) / sma_p, mode="same")
    rsi   = talib.RSI(closes, timeperiod=rsi_p)
    macd_, _, _ = talib.MACD(closes,
                             fastperiod=fast,
                             slowperiod=slow,
                             signalperiod=sig)

    feats = np.column_stack([raw[:, 1:6], sma, rsi, macd_]).astype(np.float32)
    ts    = raw[:, 0]

    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(feats, (24, 8)).squeeze()
    win_t   = torch.tensor(windows, dtype=torch.float32, device=ensemble.device)

    pred_idx, _, avg_params = ensemble.vectorized_predict(win_t, batch_size=512)
    preds = [2] * 23 + pred_idx.tolist()                 # pad front

    df = pd.DataFrame({
        "ts"   : ts,
        "open" : feats[:, 0] * 1e5,
        "high" : feats[:, 1] * 1e5,
        "low"  : feats[:, 2] * 1e5,
        "close": feats[:, 3] * 1e5,
        "pred" : preds
    })
    df["prev_close"] = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum(np.abs(df["high"] - df["prev_close"]),
                               np.abs(df["low"]  - df["prev_close"])))
    df["ATR"] = tr.rolling(global_ATR_period, min_periods=1).mean()

    bal          = init_bal = 100.0
    equity_curve = []
    trades       = []

    position = dict(size=0.0, side=None, entry_price=0.0,
                    SL=0.0, TP=0.0, entry_ts=None)
    last_exit_ts = None

    SLm = avg_params["sl_multiplier"].item()
    TPm = avg_params["tp_multiplier"].item()
    RF  = avg_params["risk_fraction"].item()

    # ---------------- main bar-loop ------------------------------------- #
    for _, row in df.iterrows():
        ts_, price = row["ts"], row["close"]

        # ── manage open position ─────────────────────────────────────────
        if position["size"]:
            hit_SL, hit_TP = False, False
            exit_reason    = ""

            if position["side"] == "long":
                hit_SL = row["low"]  <= position["SL"]
                hit_TP = row["high"] >= position["TP"]
                reversal = row["pred"] == 1
            else:                                # short
                hit_SL = row["high"] >= position["SL"]
                hit_TP = row["low"]  <= position["TP"]
                reversal = row["pred"] == 0

            if hit_SL:
                exit_reason = "SL"
                fill_px = position["SL"]
            elif hit_TP:
                exit_reason = "TP"
                fill_px = position["TP"]
            elif reversal:
                exit_reason = "REV"
                fill_px = price
            else:
                fill_px = price                                               # unrealised

            if exit_reason:
                # apply slippage on exit
                fill_px *= (1 - SLIPPAGE_PCT) if position["side"] == "long" else (1 + SLIPPAGE_PCT)

                if position["side"] == "long":
                    proceeds = position["size"] * fill_px
                    fee      = proceeds * COMM
                    pnl      = proceeds - position["size"] * position["entry_price"] - fee
                else:                                                         # short
                    fee  = abs(position["size"]) * fill_px * COMM
                    pnl  = abs(position["size"]) * (position["entry_price"] - fill_px) - fee
                    hours = (ts_ - position["entry_ts"]) / 3600
                    pnl  -= abs(position["size"]) * position["entry_price"] * FUNDING_RATE * hours

                bal += pnl
                trades.append(dict(entry_time=position["entry_ts"],
                                   exit_time=ts_,
                                   side=position["side"],
                                   entry_price=position["entry_price"],
                                   exit_price=fill_px,
                                   exit_reason=exit_reason,
                                   return_pct=(fill_px / position["entry_price"] - 1)
                                               * (1 if position["side"] == "long" else -1),
                                   duration_sec=ts_ - position["entry_ts"]))
                position.update(size=0.0, side=None)
                last_exit_ts = ts_

        # ── open new position if flat ────────────────────────────────────
        if position["size"] == 0 and row["pred"] in (0, 1):
            if last_exit_ts is None or ts_ - last_exit_ts >= MIN_HOLD:
                atr   = max(row["ATR"], 1.0)
                side  = "long" if row["pred"] == 0 else "short"
                fill  = price * (1 + SLIPPAGE_PCT if side == "long" else 1 - SLIPPAGE_PCT)
                SL_px = fill - SLm * atr if side == "long" else fill + SLm * atr
                TP_px = fill + TPm * atr if side == "long" else fill - TPm * atr

                risk_cap  = bal * RF
                pos_size  = risk_cap / (abs(fill - SL_px) + 1e-8)
                max_size  = bal * LEVERAGE / fill
                pos_size  = min(pos_size, max_size)

                if pos_size > 0:
                    bal -= pos_size * fill * COMM
                    position.update(size=pos_size if side == "long" else -pos_size,
                                    side=side,
                                    entry_price=fill,
                                    SL=SL_px,
                                    TP=TP_px,
                                    entry_ts=ts_)

        # ── mark-to-market equity ───────────────────────────────────────
        eq = bal
        if position["size"]:
            if position["side"] == "long":
                eq += position["size"] * (price - position["entry_price"])
            else:
                eq += abs(position["size"]) * (position["entry_price"] - price)
        equity_curve.append((ts_, eq))

    # force-close any open position at final bar
    if position["size"]:
        final_px = df.iloc[-1]["close"]
        final_px *= (1 - SLIPPAGE_PCT) if position["side"] == "long" else (1 + SLIPPAGE_PCT)

        if position["side"] == "long":
            fee = position["size"] * final_px * COMM
            pnl = position["size"] * (final_px - position["entry_price"]) - fee
        else:
            fee = abs(position["size"]) * final_px * COMM
            hrs = (df.iloc[-1]["ts"] - position["entry_ts"]) / 3600
            pnl = abs(position["size"]) * (position["entry_price"] - final_px) \
                  - fee - abs(position["size"]) * position["entry_price"] * FUNDING_RATE * hrs
        bal += pnl
        trades.append(dict(entry_time=position["entry_ts"],
                           exit_time=df.iloc[-1]["ts"],
                           side=position["side"],
                           entry_price=position["entry_price"],
                           exit_price=final_px,
                           exit_reason="FINAL",
                           return_pct=(final_px / position["entry_price"] - 1)
                                       * (1 if position["side"] == "long" else -1),
                           duration_sec=df.iloc[-1]["ts"] - position["entry_ts"]))
        equity_curve[-1] = (df.iloc[-1]["ts"], bal)

    # ── stats & reward ──────────────────────────────────────────────────
    net_pct   = 100.0 * (bal - init_bal) / init_bal
    eqdf      = pd.DataFrame(equity_curve, columns=["ts", "bal"]).set_index(
                    pd.to_datetime([t for t, _ in equity_curve], unit="s"))
    daily_ret = eqdf["bal"].pct_change().dropna()
    sharpe    = ((daily_ret.mean() * 252) /
                 (daily_ret.std()  * math.sqrt(252))) if not daily_ret.empty else 0.0
    mdd       = ((eqdf["bal"] - eqdf["bal"].cummax()) / eqdf["bal"].cummax()).min() if not eqdf.empty else 0.0

    # inactivity penalty
    gaps = []
    if trades:
        entries = sorted(t["entry_time"] for t in trades)
        gaps.append(entries[0] - df.iloc[0]["ts"])
        gaps.extend(max(0, e1 - e0) for e0, e1 in zip(entries, entries[1:]))
        gaps.append(df.iloc[-1]["ts"] - entries[-1])
    else:
        gaps.append(df.iloc[-1]["ts"] - df.iloc[0]["ts"])
    inact_pen = sum(inactivity_exponential_penalty(g) for g in gaps)

    days_pf  = compute_days_in_profit(equity_curve, init_bal)
    trades_n = len(trades)

    composite = (α * (net_pct / 100.0)
                 + β * sharpe * 2
                 + γ * (1 - abs(mdd))
                 + δ * trades_n * 3
                 + (days_pf / 365) * 1000
                 - inact_pen * 0.01)                            # small offset

    return dict(net_pct           = net_pct,
                trades            = trades_n,
                effective_net_pct = net_pct,
                sharpe            = sharpe,
                max_drawdown      = mdd,
                equity_curve      = equity_curve,
                trade_details     = trades,
                composite_reward  = composite,
                inactivity_penalty= inact_pen,
                days_without_trading=0.0,
                days_in_profit    = days_pf)

###############################################################################
# 9 – Ensemble wrapper around multiple TradingModel instances
#     (includes loss, optimisation, LR-scheduler, checkpointing, etc.)
###############################################################################
class EnsembleModel:
    def __init__(self,
                 device: torch.device,
                 n_models: int = 2,
                 lr: float = 3e-4,
                 weight_decay: float = 1e-4) -> None:
        self.device = device
        self.models = [TradingModel().to(device) for _ in range(n_models)]
        self.opts   = [optim.Adam(m.parameters(),
                                  lr=lr,
                                  weight_decay=weight_decay) for m in self.models]
        self.scaler       = GradScaler(enabled=(device.type == "cuda"))
        self.criterion    = nn.CrossEntropyLoss(
                                weight=torch.tensor([2.0, 2.0, 0.8]).to(device))
        self.reg_loss_fn  = nn.MSELoss()
        self.schedulers   = [ReduceLROnPlateau(opt, mode="min",
                                              patience=3, factor=0.5)
                             for opt in self.opts]

        self.best_state_dicts: list = []
        self.best_composite_reward  = float("-inf")
        self.train_steps            = 0
        self.reward_loss_weight     = 0.2
        self.patience               = 0

        self.indicator_hparams = IndicatorHyperparams(
            rsi_period=14, sma_period=10, macd_fast=12, macd_slow=26, macd_signal=9)

    def train(self, mode: bool = True):
        for m in self.models:
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)
    # ------------------------------------------------------------------ #
    def _forward_batch(self, batch_x, batch_y, target_reward):
        """
        Single forward/back-prop through **all** sub-models.  Returns mean loss.
        """
        total = 0.0
        for model, opt in zip(self.models, self.opts):
            opt.zero_grad()
            with autocast():
                logits, _, pred_r = model(batch_x)
                ce_loss  = self.criterion(logits, batch_y)
                r_loss   = self.reg_loss_fn(pred_r, target_reward.expand_as(pred_r))
                loss     = ce_loss + self.reward_loss_weight * r_loss
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            self.scaler.step(opt)
            self.scaler.update()
            total += loss.item()
        return total / len(self.models)

    # ------------------------------------------------------------------ #
    def train_one_epoch(self,
                        dl_train: DataLoader,
                        dl_val  : DataLoader,
                        raw_data: list) -> Tuple[float, float]:
        """
        Forward pass, back-test, loss computation, LR scheduling, patience logic,
        checkpointing.  Updates the many global-state variables used by GUI.
        """
        global global_training_loss, global_validation_loss, global_backtest_profit, \
           global_equity_curve, global_inactivity_penalty, global_composite_reward, \
           global_days_without_trading, global_trade_details, global_days_in_profit, \
           global_sharpe, global_max_drawdown, global_net_pct, global_num_trades, \
           global_yearly_stats_table, global_best_equity_curve, \
           global_best_sharpe, global_best_drawdown, global_best_net_pct, \
           global_best_num_trades, global_best_inactivity_penalty, \
           global_best_composite_reward, global_best_days_in_profit, \
           global_best_lr, global_best_wd

        bt = robust_backtest(self, raw_data)

        # write global metrics for GUI
        global_equity_curve       = bt["equity_curve"]
        global_inactivity_penalty = bt["inactivity_penalty"]
        global_composite_reward   = bt["composite_reward"]
        global_trade_details      = bt["trade_details"]
        global_days_in_profit     = bt["days_in_profit"]
        global_sharpe             = bt["sharpe"]
        global_max_drawdown       = bt["max_drawdown"]
        global_net_pct            = bt["net_pct"]
        global_num_trades         = bt["trades"]
        global_backtest_profit.append(bt["effective_net_pct"])

        # yearly stats table for GUI
        dfy, tstr                 = compute_yearly_stats(bt["equity_curve"],
                                                         bt["trade_details"])
        global_yearly_stats_table = tstr

        # scaled reward target
        target_r = torch.tanh(torch.tensor(bt["composite_reward"] / 100.0,
                                           dtype=torch.float32,
                                           device=self.device))

        # ---- training loop -------------------------------------------
        self.train_steps += 1
        tot_loss, n_batches = 0.0, 0
        for batch_x, batch_y in dl_train:
            tot_loss += self._forward_batch(batch_x.to(self.device),
                                            batch_y.to(self.device),
                                            target_r)
            n_batches += 1
        train_loss = tot_loss / n_batches
        global_training_loss.append(train_loss)

        # ---- validation ---------------------------------------------
        if dl_val:
            self.eval()
            with torch.no_grad():
                v_losses = []
                for bx, by in dl_val:
                    bx, by = bx.to(self.device), by.to(self.device)
                    losses = []
                    for model in self.models:
                        lg, _, _ = model(bx)
                        losses.append(self.criterion(lg, by).item())
                    v_losses.append(np.mean(losses))
                val_loss = float(np.mean(v_losses))
            self.train()
            global_validation_loss.append(val_loss)
            for sch in self.schedulers:
                sch.step(val_loss)
        else:
            val_loss = float("nan")
            global_validation_loss.append(None)

        # ---- update best-so-far snapshot ----------------------------
        if bt["composite_reward"] > self.best_composite_reward:
            self.best_composite_reward = bt["composite_reward"]
            self.best_state_dicts = [m.state_dict() for m in self.models]
            self.save_best("best_model_weights.pth")
            self.patience = 0
        else:
            self.patience += 1

        # early-stopping / LR-anneal policy
        if self.patience >= 30:
            for opt in self.opts:
                for pg in opt.param_groups:
                    pg["lr"] = max(pg["lr"] * 0.5, 1e-6)
            self.patience = 0

        # update global “best run” metrics
        if bt["net_pct"] > global_best_net_pct:
            global_best_equity_curve      = bt["equity_curve"]
            global_best_sharpe            = bt["sharpe"]
            global_best_drawdown          = bt["max_drawdown"]
            global_best_net_pct           = bt["net_pct"]
            global_best_num_trades        = bt["trades"]
            global_best_inactivity_penalty= bt["inactivity_penalty"]
            global_best_composite_reward  = bt["composite_reward"]
            global_best_days_in_profit    = bt["days_in_profit"]
            global_best_lr = self.opts[0].param_groups[0]["lr"]
            global_best_wd = self.opts[0].param_groups[0]["weight_decay"]

        return train_loss, val_loss

    # ------------------------------------------------------------------ #
    def save_best(self, path: str) -> None:
        if self.best_state_dicts:
            torch.save(dict(state_dicts=self.best_state_dicts,
                            best_reward=self.best_composite_reward),
                       path)

    # ------------------------------------------------------------------ #
    def load_best(self, path: str, raw_data: list) -> None:
        if os.path.isfile(path):
            ckpt = torch.load(path, map_location=self.device)
            self.best_state_dicts      = ckpt["state_dicts"]
            self.best_composite_reward = ckpt.get("best_reward", -np.inf)
            for m, sd in zip(self.models, self.best_state_dicts):
                m.load_state_dict(sd, strict=False)
            if raw_data:
                robust_backtest(self, raw_data)               # refresh global best
        else:
            logging.warning("No previous checkpoint '%s'.", path)

    # ------------------------------------------------------------------ #
    def vectorized_predict(self, windows_tensor: torch.Tensor,
                           batch_size: int = 256):
        self.eval()
        with torch.no_grad():
            probs_all = []
            n = windows_tensor.shape[0]
            for i in range(0, n, batch_size):
                batch = windows_tensor[i:i + batch_size]
                p = []
                for m in self.models:
                    lg, tpars, _ = m(batch)
                    p.append(torch.softmax(lg, dim=1))
                probs_all.append(torch.mean(torch.stack(p), dim=0))
            full = torch.cat(probs_all, dim=0)
            idx  = full.argmax(dim=1)
            conf = full.max(dim=1)[0]
            dummy = dict(risk_fraction=torch.tensor([0.1]),
                         sl_multiplier=torch.tensor([5.0]),
                         tp_multiplier=torch.tensor([5.0]))
            return idx.cpu(), conf.cpu(), dummy
###############################################################################
# 10 – Tkinter dashboard GUI
###############################################################################
class TradingGUI:
    """
    Live dashboard that embeds four Matplotlib charts + two text tabs showing
    trades & yearly stats, plus live AI-meta-adjustment logs on the right.
    This class *polls* the global variables the training / meta threads update.
    """
    def __init__(self,
                 root: tk.Tk,
                 ensemble: "EnsembleModel",
                 update_ms: int = 2_000) -> None:
        self.ensemble = ensemble
        self.update_ms = update_ms

        root.title("Complex AI Trading – Live Dashboard")
        # notebooks -----------------------------------------------------------------
        note = ttk.Notebook(root)
        note.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── tab 1: training/validation loss + equity --------------------------------
        frame_tv = ttk.Frame(note)
        note.add(frame_tv, text="Training / Equity")
        fig_tv, (ax_loss, ax_eq) = plt.subplots(2, 1, figsize=(5, 6))
        self.ax_loss, self.ax_eq = ax_loss, ax_eq
        self.canvas_tv = FigureCanvasTkAgg(fig_tv, master=frame_tv)
        self.canvas_tv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── tab 2: live price --------------------------------------------------------
        frame_live = ttk.Frame(note)
        note.add(frame_live, text="Live Price")
        fig_live, ax_live = plt.subplots(figsize=(5, 3))
        self.ax_live = ax_live
        self.canvas_live = FigureCanvasTkAgg(fig_live, master=frame_live)
        self.canvas_live.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── tab 3: net-profit over epochs -------------------------------------------
        frame_np = ttk.Frame(note)
        note.add(frame_np, text="Net-Profit (%)")
        fig_np, ax_np = plt.subplots(figsize=(5, 3))
        self.ax_np = ax_np
        self.canvas_np = FigureCanvasTkAgg(fig_np, master=frame_np)
        self.canvas_np.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── tab 4: attention weights placeholder ------------------------------------
        frame_attn = ttk.Frame(note)
        note.add(frame_attn, text="Attention Weights")
        fig_at, ax_at = plt.subplots(figsize=(5, 3))
        self.ax_at = ax_at
        self.canvas_at = FigureCanvasTkAgg(fig_at, master=frame_attn)
        self.canvas_at.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── tab 5: trade-details JSON ----------------------------------------------
        frame_trades = ttk.Frame(note)
        note.add(frame_trades, text="Trade Details")
        self.txt_trades = tk.Text(frame_trades, width=60, height=24, wrap="none")
        self.txt_trades.pack(fill=tk.BOTH, expand=True)

        # ── tab 6: yearly performance table -----------------------------------------
        frame_year = ttk.Frame(note)
        note.add(frame_year, text="Yearly Stats")
        self.txt_year = tk.Text(frame_year, width=40, height=24, wrap="none")
        self.txt_year.pack(fill=tk.BOTH, expand=True)

        # ── sidebar – AI meta-logs ---------------------------------------------------
        side = ttk.Frame(root)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        ttk.Label(side, text="AI Meta-Adjustments (latest):",
                  font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.txt_latest = tk.Text(side, width=46, height=6, wrap="word")
        self.txt_latest.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(side, text="AI Adjustments Log:",
                  font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.txt_log = tk.Text(side, width=46, height=20, wrap="word")
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        # footer – small stats grid ---------------------------------------------------
        footer = ttk.Frame(root)
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        self.lbl_pred  = ttk.Label(footer, text="Pred: –")
        self.lbl_conf  = ttk.Label(footer, text="Conf: –")
        self.lbl_step  = ttk.Label(footer, text="Step: –")
        self.lbl_lr    = ttk.Label(footer, text="LR: –")
        for i, w in enumerate((self.lbl_pred, self.lbl_conf, self.lbl_step, self.lbl_lr)):
            w.grid(row=0, column=i, sticky="w", padx=4)

        root.after(self.update_ms, self._refresh)

    # ------------------------------------------------------------------------- #
    def _refresh(self):
        # update loss / equity chart
        self.ax_loss.clear()
        self.ax_loss.plot(global_training_loss, label="Train", color="blue")
        if any(v is not None for v in global_validation_loss):
            self.ax_loss.plot(
                [v if v is not None else np.nan for v in global_validation_loss],
                label="Val", color="orange")
        self.ax_loss.set_title("Loss"); self.ax_loss.legend()

        self.ax_eq.clear()
        if global_equity_curve:
            t, b = zip(*global_equity_curve)
            self.ax_eq.plot([datetime.datetime.fromtimestamp(x) for x in t],
                            b, color="red", label="Current")
        if global_best_equity_curve:
            t, b = zip(*global_best_equity_curve)
            self.ax_eq.plot([datetime.datetime.fromtimestamp(x) for x in t],
                            b, color="green", label="Best")
        self.ax_eq.set_title("Equity"); self.ax_eq.legend()
        self.canvas_tv.draw()

        # live price
        self.ax_live.clear()
        if global_phemex_data:
            t, close = zip(*[(bar[0] / 1_000, bar[4]) for bar in global_phemex_data])
            self.ax_live.plot([datetime.datetime.fromtimestamp(x) for x in t], close)
        self.ax_live.set_title("Phemex 1h Close")
        self.canvas_live.draw()

        # net-profit history
        self.ax_np.clear()
        if global_backtest_profit:
            self.ax_np.plot(global_backtest_profit, color="green")
        self.ax_np.set_title("Net-Profit (%)")
        self.canvas_np.draw()

        # attention weights placeholder
        self.ax_at.clear()
        if global_attention_weights_history:
            self.ax_at.plot(global_attention_weights_history, color="purple")
        self.ax_at.set_title("Avg Attention (placeholder)")
        self.canvas_at.draw()

        # trades JSON
        self.txt_trades.delete("1.0", tk.END)
        self.txt_trades.insert(tk.END,
            json.dumps(global_trade_details[-50:], indent=2) if global_trade_details else "No trades yet")

        # yearly stats
        self.txt_year.delete("1.0", tk.END)
        self.txt_year.insert(tk.END, global_yearly_stats_table or "—")

        # meta logs
        self.txt_latest.delete("1.0", tk.END)
        self.txt_latest.insert(tk.END, global_ai_adjustments or "—")
        self.txt_log.delete("1.0", tk.END)
        self.txt_log.insert(tk.END, global_ai_adjustments_log[-5_000:])

        # footer
        self.lbl_pred.config(text=f"Pred: {global_current_prediction or '–'}")
        self.lbl_conf.config(text=f"Conf: {global_ai_confidence or 0:.2f}")
        self.lbl_step.config(text=f"Step: {global_ai_epoch_count}")
        lr_now = self.ensemble.opts[0].param_groups[0]["lr"]
        self.lbl_lr.config(text=f"LR: {lr_now:.2e}")

        self.txt_trades.after(self.update_ms, self._refresh)


###############################################################################
# 11 – Meta-controller (Transformer RL) to tweak hyper-params on-line
###############################################################################
class TransformerMetaAgent(nn.Module):
    """
    Tiny transformer that maps a 6-element state → categorical distribution over
    1 000 randomly-sampled hyper-parameter deltas.
    """
    def __init__(self) -> None:
        super().__init__()
        self.lr_deltas       = [-0.3, -0.1, 0.1, 0.3]
        self.wd_deltas       = [-0.3,  0.0, 0.3]
        self.rsi_deltas      = [-5, 0, 5]
        self.sma_deltas      = [-5, 0, 5]
        self.macd_fast_d     = [-5, 0, 5]
        self.macd_slow_d     = [-5, 0, 5]
        self.macd_sig_d      = [-3, 0, 3]
        self.thr_deltas      = [-0.0001, 0.0, 0.0001]

        full_space = list(itertools.product(
            self.lr_deltas, self.wd_deltas, self.rsi_deltas, self.sma_deltas,
            self.macd_fast_d, self.macd_slow_d, self.macd_sig_d, self.thr_deltas))
        random.shuffle(full_space)
        self.action_space = full_space[:1000]
        self.n_actions    = len(self.action_space)

        self.embed = nn.Linear(6, 32)
        self.pos   = PositionalEncoding(32)
        enc_layer  = nn.TransformerEncoderLayer(d_model=32, nhead=1,
                                                dim_feedforward=64,
                                                dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.pi_head = nn.Linear(32, self.n_actions)
        self.v_head  = nn.Linear(32, 1)

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor):
        x = self.embed(x).unsqueeze(1).transpose(0, 1)
        x = self.encoder(self.pos(x)).squeeze(0)
        return self.pi_head(x), self.v_head(x).squeeze(1)


class MetaController:
    """
    Simple on-policy REINFORCE with bootstrap value-baseline.
    Schedules epsilon-greedy exploration w/ Gaussian perturbation on the sampled
    action’s deltas.
    """
    def __init__(self, ensemble: EnsembleModel, lr: float = 1e-3) -> None:
        self.net       = TransformerMetaAgent()
        self.optim     = optim.Adam(self.net.parameters(), lr=lr)
        self.ensemble  = ensemble

        self.eps_start = 0.5
        self.eps_end   = 0.15
        self.eps_decay = 0.995
        self.steps     = 0

    # --------------------------------------------------------------------- #
    def select_action(self, state_np: np.ndarray):
        self.steps += 1
        eps = max(self.eps_end, self.eps_start * (self.eps_decay ** self.steps))
        s   = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, val = self.net(s)
            dist = torch.distributions.Categorical(logits=logits)
        if random.random() < eps:
            idx = random.randrange(self.net.n_actions)
            logp = torch.log(torch.tensor(1 / self.net.n_actions))
        else:
            idx = dist.sample().item()
            logp = dist.log_prob(torch.tensor(idx))
        return idx, logp, val.item()

    # --------------------------------------------------------------------- #
    def apply_action(self, idx: int):
        (d_lr, d_wd, d_rsi, d_sma, d_fast, d_slow, d_sig, d_thr) = \
            self.net.action_space[idx]

        # LR / WD
        for opt in self.ensemble.opts:
            pg = opt.param_groups[0]
            pg["lr"] = max(min(pg["lr"] * (1 + d_lr), 1e-1), 1e-6)
            pg["weight_decay"] = max(pg["weight_decay"] * (1 + d_wd), 0.0)

        # indicator params
        hp = self.ensemble.indicator_hparams
        self.ensemble.indicator_hparams = IndicatorHyperparams(
            rsi_period   = max(2,  hp.rsi_period   + d_rsi),
            sma_period   = max(2,  hp.sma_period   + d_sma),
            macd_fast    = max(2,  hp.macd_fast    + d_fast),
            macd_slow    = max(3,  hp.macd_slow    + d_slow),
            macd_signal  = max(1,  hp.macd_signal  + d_sig)
        )
        # global threshold tweak
        globals()["GLOBAL_THRESHOLD"] = max(0.0, GLOBAL_THRESHOLD + d_thr)

    # --------------------------------------------------------------------- #
    def update(self,
               state: np.ndarray,
               action_idx: int,
               reward: float,
               next_state: np.ndarray,
               logp: torch.Tensor,
               value: float):
        s  = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        ns = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        _, v_s   = self.net(s)
        _, v_ns  = self.net(ns)
        target   = reward + 0.95 * v_ns.item()
        adv      = target - v_s.item()
        loss_p   = -logp * adv
        loss_v   = 0.5 * (v_s - target) ** 2
        loss     = loss_p + loss_v

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


###############################################################################
# 12 – Background threads
###############################################################################
def csv_training_thread(ensemble: EnsembleModel,
                        raw_data: list,
                        stop_evt: threading.Event) -> None:
    """
    Train → back-test loop reading local CSV.
    """
    ds_full = HourlyDataset(raw_data, seq_len=24, threshold=GLOBAL_THRESHOLD)
    n = len(ds_full)
    tr, val = random_split(ds_full, [int(n * 0.9), n - int(n * 0.9)])
    dl_tr = DataLoader(tr, batch_size=128, shuffle=True)
    dl_val = DataLoader(val, batch_size=128)

    # load previous weights
    ensemble.load_best("best_model_weights.pth", raw_data)

    while not stop_evt.is_set():
        tl, vl = ensemble.train_one_epoch(dl_tr, dl_val, raw_data)
        time.sleep(0.1)                         # small breather


def phemex_live_thread(connector: "PhemexConnector",
                       stop_evt: threading.Event) -> None:
    """
    Pull 1-hour candles every ~60 s → feed queue for live price pane + optional
    incremental training.
    """
    while not stop_evt.is_set():
        try:
            bars = connector.fetch_latest_bars(limit=200)
            if bars:
                global_phemex_data.clear()
                global_phemex_data.extend(bars)
            time.sleep(60)
        except Exception as exc:
            logging.warning("Live-fetch error: %s", exc)
            time.sleep(60)


###############################################################################
# 13 – Exchange connector (only public OHLCV used)
###############################################################################
class PhemexConnector:
    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self.ex = ccxt.phemex({"enableRateLimit": True})
        self.ex.load_markets()
        if self.symbol not in self.ex.markets:
            raise RuntimeError(f"Symbol {self.symbol} not found on Phemex.")

    # --------------------------------------------------------------------- #
    def fetch_latest_bars(self, limit: int = 200):
        """
        Returns list of [ts, open, high, low, close, volume] with ts in *ms*.
        """
        bars = self.ex.fetch_ohlcv(self.symbol, timeframe="1h", limit=limit)
        return bars


###############################################################################
# 14 – Misc helpers
###############################################################################
def save_checkpoint():
    with open("checkpoint.json", "w") as fp:
        json.dump(dict(training_loss      = global_training_loss,
                       validation_loss    = global_validation_loss,
                       backtest_profit    = global_backtest_profit,
                       equity_curve       = global_equity_curve,
                       ai_log             = global_ai_adjustments_log),
                  fp, indent=2)


###############################################################################
# 15 – main()
###############################################################################
def main():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = load_csv_hourly("Gemini_BTCUSD_1h.csv")
    if not raw:
        print("No CSV data, exit."); return

    ensemble = EnsembleModel(device=device, n_models=2, lr=3e-4, weight_decay=1e-4)
    controller = MetaController(ensemble)

    stop_evt = threading.Event()
    threading.Thread(target=csv_training_thread,
                     args=(ensemble, raw, stop_evt), daemon=True).start()
    phemex = PhemexConnector()
    threading.Thread(target=phemex_live_thread,
                     args=(phemex, stop_evt), daemon=True).start()

    # meta-loop in main thread (simpler)
    def run_meta():
        prev_r = global_composite_reward
        while not stop_evt.is_set():
            state = np.array([global_composite_reward or 0.0,
                              global_best_composite_reward or 0.0,
                              global_sharpe,
                              abs(global_max_drawdown),
                              global_num_trades,
                              global_days_in_profit or 0.0],
                             dtype=np.float32)
            a_idx, logp, val = controller.select_action(state)
            controller.apply_action(a_idx)
            time.sleep(5)
            reward = (global_composite_reward or 0.0) - (prev_r or 0.0)
            next_state = np.array([global_composite_reward or 0.0,
                                   global_best_composite_reward or 0.0,
                                   global_sharpe,
                                   abs(global_max_drawdown),
                                   global_num_trades,
                                   global_days_in_profit or 0.0],
                                  dtype=np.float32)
            controller.update(state, a_idx, reward, next_state, logp, val)
            prev_r = global_composite_reward

    threading.Thread(target=run_meta, daemon=True).start()

    # Tk GUI
    root = tk.Tk()
    TradingGUI(root, ensemble)
    try:
        root.mainloop()
    finally:
        stop_evt.set()
        ensemble.save_best("best_model_weights.pth")
        save_checkpoint()


if __name__ == "__main__":
    main()
