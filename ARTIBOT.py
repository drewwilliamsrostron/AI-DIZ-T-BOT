#!/usr/bin/env python3
"""
Complex AI Trading + Continuous Training + Robust Backtest Each Epoch + Live Phemex + Tkinter GUI
+ Meta–Control via Neural Network (Policy Gradient w/ Transformer) for Hyperparameter Adjustment
(Now includes all 9 improvements requested)
"""
###############################################################################
# NumPy 2.x compatibility shim – restores removed constants for old libraries
###############################################################################
import sys, math

try:
    import numpy as _np
except ModuleNotFoundError:
    # fresh venv? – install NumPy first
    import subprocess, sys as _sys
    subprocess.check_call([_sys.executable, "-m", "pip", "install", "numpy>=2.2"])
    import numpy as _np

# Re-export the aliases deleted in NumPy 2
for _name, _value in {
        "NaN":  _np.nan,
        "Inf":  _np.inf,
        "PINF": _np.inf,
        "NINF": -_np.inf,
}.items():
    setattr(_np, _name, _value)
    sys.modules["numpy"].__dict__[_name] = _value

###############################################################################
# Smart installer – works on Python 3.9 → 3.13, CPU or CUDA
###############################################################################
import sys, subprocess, platform, itertools

def _install_pytorch_for_env() -> None:
    """
    Install torch + torchvision + torchaudio that **exist** for the
    interpreter that is running this script.

    • 3.9 – 3.12 → stable 2.2.1 wheels (+cu118 or +cpu)
    • 3.13       → current nightly wheels (>= 2.6.0.dev…)
    """
    major, minor = sys.version_info[:2]

    # crude CUDA check: we’ll try GPU wheels first only when a NVIDIA adapter
    # is visible to Windows.  Fall back to CPU if the install fails later.
    cuda_ok = (
        platform.system() == "Windows" and
        "NVIDIA" in subprocess.getoutput(
            "wmic path win32_VideoController get name"
        )
    )

    if (major, minor) >= (3, 13):
        # 🟡 Nightly wheels already publish cp313 tags
        index = "https://download.pytorch.org/whl/nightly/cu118" if cuda_ok \
                else "https://download.pytorch.org/whl/nightly/cpu"
        pkg_line = ["torch", "torchvision", "torchaudio", "--pre", "-f", index]
    else:
        # 🟢 Stable LTS wheels (2.2.x)
        suffix   = "+cu118" if cuda_ok else "+cpu"
        index    = "https://download.pytorch.org/whl/cu118" if cuda_ok \
                   else "https://download.pytorch.org/whl/cpu"
        pkg_line = [
            f"torch==2.2.1{suffix}",
            f"torchvision==0.17.1{suffix}",
            f"torchaudio==2.2.2{suffix}",
            "--extra-index-url", index
        ]

    print("• Installing PyTorch trio for this environment …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkg_line])

def install_dependencies() -> None:
    """Install PyTorch (if missing) plus the rest of the requirements."""
    try:
        import torch  # noqa: F401
    except ImportError:
        _install_pytorch_for_env()

    # ---------------- other pure-Python dependencies ----------------
    pkgs = {
        "openai":       "openai",
        "ccxt":         "ccxt",
        "pandas":       "pandas",
        "numpy":        "numpy",
        "matplotlib":   "matplotlib",
        "TA-Lib":       "TA-Lib"   # imported as `talib`
    }
    for import_name, pip_name in pkgs.items():
        try:
            __import__("talib" if import_name == "TA-Lib" else import_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

###############################################################################
#  ❒  TA-Lib fallback for Python 3.13 (no binary wheels yet)
###############################################################################
try:
    import talib                                    # ← works on 3.9-3.12 if wheels exist
except ModuleNotFoundError:
    # no wheel → install pandas-ta once and build a tiny shim
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas-ta"])
    import pandas as pd, pandas_ta as pta

    class _TaShim:                                 # exposes *only* the funcs you use
        @staticmethod
        def RSI(arr, timeperiod=14):
            return pta.rsi(pd.Series(arr), length=timeperiod).values
        
        @staticmethod
        def MACD(arr, fastperiod=12, slowperiod=26, signalperiod=9):
            series = pd.Series(arr)
            try:
                res = pta.macd(series, fast=fastperiod, slow=slowperiod,
                                signal=signalperiod)
                if res is not None:
                    return (
                        res[f"MACD_{fastperiod}_{slowperiod}_{signalperiod}"].values,
                        res[f"MACDs_{fastperiod}_{slowperiod}_{signalperiod}"].values,
                        res[f"MACDh_{fastperiod}_{slowperiod}_{signalperiod}"].values,
                    )
            except Exception:
                pass

            # Fallback manual calculation when pandas_ta returns None or errors
            ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
            ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=signalperiod, adjust=False).mean()
            hist = macd - signal
            return macd.values, signal.values, hist.values

    import sys
    sys.modules["talib"] = _TaShim()               # ✅ calls like talib.RSI(...) keep working


install_dependencies()
###############################################################################
# Imports
###############################################################################
import os, random, json, time, datetime, threading, queue, re
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ccxt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
import openai
from typing import NamedTuple
from sklearn.preprocessing import StandardScaler
import talib  # For RSI and MACD

# Reduce default logging to warnings only
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')
client = openai  # alias

###############################################################################
# Global Hyperparameters and Starting Values
###############################################################################
global_SL_multiplier = 5
global_TP_multiplier = 5
global_ATR_period = 50
risk_fraction = 0.03
GLOBAL_THRESHOLD = 0.0001

global_best_params = {
    "SL_multiplier": global_SL_multiplier,
    "TP_multiplier": global_TP_multiplier,
    "ATR_period": global_ATR_period,
    "risk_fraction": risk_fraction,
    "learning_rate": 1e-4
}

# Global performance metrics:
global_sharpe = 0.0
global_max_drawdown = 0.0
global_net_pct = 0.0
global_num_trades = 0
global_inactivity_penalty = None
global_composite_reward = None
global_days_without_trading = None
global_trade_details = []
global_days_in_profit = None

# Global best performance stats:
global_best_equity_curve = []
global_best_sharpe = 0.0
global_best_drawdown = 0.0
global_best_net_pct = 0.0
global_best_num_trades = 0
global_best_inactivity_penalty = None
global_best_composite_reward = None
global_best_days_in_profit = None

# Global best hyperparameters:
global_best_lr = None
global_best_wd = None

global_yearly_stats_table = ""

###############################################################################
# GPT Memories (unchanged)
###############################################################################
gpt_memory_squirtle = []
gpt_memory_wartorttle = []
gpt_memory_bigmanblastoise = []
gpt_memory_moneymaker = []

###############################################################################
# Additional global variables for meta agent logs
###############################################################################
global_ai_adjustments_log = "No adjustments yet"
global_ai_adjustments = ""
global_ai_confidence = None
global_ai_epoch_count = 0
global_current_prediction = None
global_training_loss = []
global_validation_loss = []
global_backtest_profit = []
global_equity_curve = []
global_attention_weights_history = []
global_phemex_data = []
global_days_in_profit = 0.0
live_bars_queue = queue.Queue()

###############################################################################
# NamedTuple for per-bar trade parameters
###############################################################################
class TradeParams(NamedTuple):
    risk_fraction: torch.Tensor
    sl_multiplier: torch.Tensor
    tp_multiplier: torch.Tensor
    attention: torch.Tensor

###############################################################################
# NamedTuple for global indicator hyperparams
###############################################################################
class IndicatorHyperparams(NamedTuple):
    rsi_period: int
    sma_period: int
    macd_fast: int
    macd_slow: int
    macd_signal: int

###############################################################################
# CSV Loader and Dataset
###############################################################################
def load_csv_hourly(csv_path):
    if not os.path.isfile(csv_path):
        logging.warning(f"CSV file '{csv_path}' not found.")
        return []
    try:
        df = pd.read_csv(csv_path, sep=r'[,\t]+', engine='python', skiprows=1, header=0)
    except Exception as e:
        logging.warning(f"Error reading CSV: {e}")
        return []
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    data = []
    for i, row in df.iterrows():
        try:
            ts = int(row['unix'])
            if ts > 1e12:
                ts = ts // 1000
            o = float(row['open']) / 100000.0
            h = float(row['high']) / 100000.0
            l = float(row['low']) / 100000.0
            c = float(row['close']) / 100000.0
            v = float(row['volume_btc']) if 'volume_btc' in df.columns else 0.0
            data.append([ts, o, h, l, c, v])
        except:
            pass
    return sorted(data, key=lambda x: x[0])

###############################################################################
# HourlyDataset
###############################################################################
class HourlyDataset(Dataset):
    def __init__(self, data, seq_len=24, threshold=GLOBAL_THRESHOLD, sma_period=10):
        self.data = data
        self.seq_len = seq_len
        self.threshold = threshold
        self.sma_period = sma_period
        self.samples, self.labels = self.preprocess()

    def preprocess(self):
        data_np = np.array(self.data, dtype=np.float32)
        closes = data_np[:, 4]
        sma = np.convolve(closes, np.ones(self.sma_period) / self.sma_period, mode='same')
        rsi = talib.RSI(closes, timeperiod=14)
        macd, _, _ = talib.MACD(closes)

        feats = np.column_stack([
            data_np[:, 1:6],
            sma.astype(np.float32),
            rsi.astype(np.float32),
            macd.astype(np.float32),
        ])
        scaler = StandardScaler()
        scaled_feats = scaler.fit_transform(feats)

        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(scaled_feats, (self.seq_len, scaled_feats.shape[1]))[:, 0]
        windows = windows[:-1]
        last_close = windows[:, -1, 3]
        next_close = scaled_feats[self.seq_len:, 3]
        rets = (next_close - last_close) / (last_close + 1e-8)
        labels = np.where(rets > self.threshold, 0, np.where(rets < -self.threshold, 1, 2))
        return windows.astype(np.float32), labels.astype(np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].copy()
        # (8) Data Augmentation: bigger probability + bigger noise
        # from 0.2 => 0.5 probability, and 0.01 => 0.02 stdev
        if random.random() < 0.5:
            sample += np.random.normal(0, 0.02, sample.shape)
        return torch.tensor(sample), torch.tensor(self.labels[idx], dtype=torch.long)

###############################################################################
# PositionalEncoding
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

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
        enc_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=4, dropout=dropout, dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.fc_proj = nn.Linear(input_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.attn = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes + 4)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = x.transpose(0,1).contiguous()
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
        risk_frac = 0.001 + 0.499 * torch.sigmoid(out_all[:, 3])
        sl_mult   = 0.5   + 9.5   * torch.sigmoid(out_all[:, 4])
        tp_mult   = 0.5   + 9.5   * torch.sigmoid(out_all[:, 5])
        pred_reward = out_all[:, 6] if out_all.shape[1] > 6 else torch.zeros_like(out_all[:,0])
        return logits, TradeParams(risk_frac, sl_mult, tp_mult, w), pred_reward

###############################################################################
# Yearly Stats (unchanged)
###############################################################################
def compute_yearly_stats(equity_curve, trades, initial_balance=100.0):
    if not equity_curve:
        return pd.DataFrame(), "No data"
    eq_df = pd.DataFrame(equity_curve, columns=["timestamp","balance"])
    eq_df["dt"] = pd.to_datetime(eq_df["timestamp"], unit="s")
    eq_df.set_index("dt", inplace=True)
    eq_df = eq_df.resample("1D").last().dropna()
    if eq_df.empty:
        return pd.DataFrame(), "No data"
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["entry_dt"] = pd.to_datetime(trades_df["entry_time"], unit="s")
        trades_df.set_index("entry_dt", inplace=True)
    year_rows = []
    grouped = eq_df.groupby(eq_df.index.year)
    for year, grp in grouped:
        if len(grp) < 2:
            continue
        start_balance = grp["balance"].iloc[0]
        end_balance = grp["balance"].iloc[-1]
        net_pct_year = 100.0 * (end_balance - start_balance) / (start_balance if start_balance!=0 else 1e-8)
        daily_returns = grp["balance"].pct_change().dropna()
        mu = daily_returns.mean()
        sigma = daily_returns.std()
        sharpe_year = (mu*252)/(sigma*np.sqrt(252)) if sigma>1e-12 else 0.0
        rolling_max = grp["balance"].cummax()
        dd = (grp["balance"] - rolling_max) / rolling_max
        mdd_year = dd.min() if not dd.empty else 0.0

        if not trades_df.empty:
            try:
                trades_in_year = trades_df.loc[str(year)]
                num_trades_year = len(trades_in_year) if not trades_in_year.empty else 0
            except KeyError:
                num_trades_year = 0
        else:
            num_trades_year = 0
        year_rows.append({
            "Year": year,
            "NetPct": net_pct_year,
            "Sharpe": sharpe_year,
            "MaxDD": mdd_year,
            "Trades": num_trades_year
        })
    if not year_rows:
        return pd.DataFrame(), "No yearly data"
    dfy = pd.DataFrame(year_rows).set_index("Year")
    table_str = dfy.to_string(float_format=lambda x:f"{x: .2f}")
    return dfy, table_str

###############################################################################
# inactivity_exponential_penalty (unchanged)
###############################################################################
def inactivity_exponential_penalty(gap_in_secs, max_penalty=100.0):
    month_secs = 30*24*3600
    months = int(gap_in_secs//month_secs)
    base = 0.01
    penalty = 0.0
    for i in range(months):
        penalty += base*(2**i)
        if penalty>max_penalty:
            penalty = max_penalty
            break
    return penalty

###############################################################################
# compute_days_in_profit (unchanged)
###############################################################################
def compute_days_in_profit(equity_curve, init_balance):
    total_s = 0.0
    if len(equity_curve)<2:
        return 0.0
    for i in range(1,len(equity_curve)):
        t_prev, b_prev = equity_curve[i-1]
        t_curr, b_curr = equity_curve[i]
        dt = t_curr - t_prev
        if b_prev>=init_balance and b_curr>=init_balance:
            total_s+= dt
        elif b_prev<init_balance and b_curr<init_balance:
            pass
        else:
            if b_curr!=b_prev:
                f = (init_balance - b_prev)/(b_curr - b_prev)
                if b_prev<init_balance and b_curr>=init_balance:
                    total_s += (1 - f)*dt
                else:
                    total_s += f*dt
    return total_s/86400.0

###############################################################################
# robust_backtest
###############################################################################
def robust_backtest(ensemble, data_full):
    if len(data_full)<24:
        return {"net_pct":0.0,"trades":0,"effective_net_pct":0.0,"equity_curve":[]}

    LEVERAGE=10
    min_hold_seconds=2*3600
    commission_rate=0.0001
    slippage=0.0002
    FUNDING_RATE=0.0001
    device = ensemble.device

    hp = ensemble.indicator_hparams

    # (5) If meta-agent is adjusting threshold, store it in ensemble or define a separate variable.
    # For a simpler demonstration, we keep using GLOBAL_THRESHOLD, but you could do:
    # threshold = ensemble.dynamic_threshold if ensemble.dynamic_threshold is not None else GLOBAL_THRESHOLD
    # or pass it in the function signature.

    # Clamps for RSI, SMA, MACD
    rsi_period = max(2, min(hp.rsi_period, 50))
    sma_period = max(2, min(hp.sma_period, 100))
    fast_macd  = max(2, min(hp.macd_fast, hp.macd_slow-1))
    slow_macd  = max(fast_macd+1, min(hp.macd_slow, 200))
    sig_macd   = max(1, min(hp.macd_signal, 50))

    # (9) Composite Reward: alpha=3.0, beta=0.5, gamma=0.8, delta=0.1
    alpha=3.0
    beta=0.5
    gamma=0.8
    delta=0.1

    raw_data = np.array(data_full,dtype=np.float64)
    closes = raw_data[:,4]

    sma = np.convolve(closes, np.ones(sma_period)/sma_period, mode='same')
    rsi = talib.RSI(closes, timeperiod=rsi_period)
    macd_, macdsig_, macdhist_ = talib.MACD(closes, fastperiod=fast_macd,
                                           slowperiod=slow_macd,
                                           signalperiod=sig_macd)

    extd = []
    for i,row in enumerate(raw_data):
        extd.append(list(row[1:6]) + [float(sma[i]), float(rsi[i]), float(macd_[i])])
    extd = np.array(extd,dtype=np.float32)
    timestamps = raw_data[:,0]

    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(extd, (24,8)).squeeze()
    windows_t = torch.tensor(windows, dtype=torch.float32, device=device)
    pred_indices, _, avg_params = ensemble.vectorized_predict(windows_t,batch_size=512)
    preds = [2]*23 + pred_indices.tolist()

    df = pd.DataFrame({
        'timestamp':timestamps,
        'open':extd[:,0]*100000,
        'high':extd[:,1]*100000,
        'low': extd[:,2]*100000,
        'close':extd[:,3]*100000,
        'prediction': preds
    })
    df['previous_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(
        df['high']-df['low'],
        np.maximum(
            np.abs(df['high']-df['previous_close']),
            np.abs(df['low']-df['previous_close'])
        )
    )
    df['ATR'] = df['tr'].rolling(global_ATR_period, min_periods=1).mean()

    init_bal=100.0
    bal=init_bal
    eq_curve=[]
    trades=[]
    pos={'size':0.0,'side':None,'entry_price':0.0,
         'stop_loss':0.0,'take_profit':0.0,'entry_time':None}
    last_exit_t=None

    sl_m = avg_params["sl_multiplier"].item()
    tp_m = avg_params["tp_multiplier"].item()
    rf   = avg_params["risk_fraction"].item()

    for i,row in df.iterrows():
        cur_t = row['timestamp']
        cur_p = row['close']
        if pos['size'] != 0:
            exit_condition=False
            exit_price= cur_p
            exit_reason=""
            if pos['side']=='long':
                if row['low']<= pos['stop_loss']:
                    exit_price= pos['stop_loss']
                    exit_reason="SL hit"
                    exit_condition=True
                elif row['high']>= pos['take_profit']:
                    exit_price= pos['take_profit']
                    exit_reason="TP hit"
                    exit_condition=True
                elif row['prediction']==1:
                    exit_reason="Signal reversal"
                    exit_condition=True
            else:
                if row['high']>= pos['stop_loss']:
                    exit_price= pos['stop_loss']
                    exit_reason="SL hit"
                    exit_condition=True
                elif row['low']<= pos['take_profit']:
                    exit_price= pos['take_profit']
                    exit_reason="TP hit"
                    exit_condition=True
                elif row['prediction']==0:
                    exit_reason="Signal reversal"
                    exit_condition=True
            if exit_condition:
                last_exit_t= cur_t
                if pos['side']=='long':
                    exit_price*= (1- slippage)
                    proceeds= pos['size']* exit_price
                    comm_exit= proceeds* commission_rate
                    profit= proceeds- (pos['size']* pos['entry_price'])- comm_exit
                    bal+= profit
                else:
                    exit_price*= (1+ slippage)
                    comm_exit= abs(pos['size'])* exit_price* commission_rate
                    profit= abs(pos['size'])*(pos['entry_price']- exit_price)- comm_exit
                    hrs= (cur_t- pos['entry_time'])/3600.0
                    funding= abs(pos['size'])*pos['entry_price']*FUNDING_RATE* hrs
                    profit-= funding
                    bal+= profit
                trades.append({
                    'entry_time':pos['entry_time'],
                    'exit_time':cur_t,
                    'side':pos['side'],
                    'entry_price':pos['entry_price'],
                    'stop_loss':pos['stop_loss'],
                    'take_profit':pos['take_profit'],
                    'exit_price':exit_price,
                    'exit_reason':exit_reason,
                    'return': (exit_price/pos['entry_price']-1)*(1 if pos['side']=='long' else -1),
                    'duration':cur_t- pos['entry_time']
                })
                pos={'size':0.0,'side':None,'entry_price':0.0,
                     'stop_loss':0.0,'take_profit':0.0,'entry_time':None}
        if pos['size']==0 and row['prediction'] in (0,1):
            if last_exit_t is not None and (cur_t- last_exit_t)< min_hold_seconds:
                pass
            else:
                atr= row['ATR'] if not np.isnan(row['ATR']) else 1.0
                atr= max(1.0,atr)
                fill_p= cur_p*(1+slippage if row['prediction']==0 else 1- slippage)
                st_dist= sl_m*atr
                tp_val= fill_p+ tp_m*atr if row['prediction']==0 else fill_p- tp_m*atr
                risk_cap= bal*rf
                pos_size_risk= risk_cap/(st_dist+1e-8)
                max_sz= (bal* LEVERAGE)/ fill_p
                pos_sz= min(pos_size_risk, max_sz)
                if pos_sz>0:
                    comm_entry= pos_sz* fill_p* commission_rate
                    bal-= comm_entry
                    if row['prediction']==0:
                        pos.update({'size':pos_sz,'side':'long','entry_price':fill_p,
                                    'stop_loss': fill_p- st_dist, 'take_profit': tp_val,
                                    'entry_time': cur_t})
                    else:
                        pos.update({'size':-pos_sz,'side':'short','entry_price':fill_p,
                                    'stop_loss': fill_p+ st_dist, 'take_profit': tp_val,
                                    'entry_time': cur_t})
        curr_eq= bal
        if pos['size']!=0:
            if pos['side']=='long':
                curr_eq+= pos['size']*(cur_p- pos['entry_price'])
            else:
                curr_eq+= abs(pos['size'])*(pos['entry_price']- cur_p)
        eq_curve.append((cur_t, curr_eq))

    if pos['size']!=0:
        final_price= df.iloc[-1]['close']
        if pos['side']=='long':
            exit_price= final_price*(1- slippage)
            comm_exit= pos['size']* exit_price* commission_rate
            pf= pos['size']*(exit_price- pos['entry_price'])- comm_exit
            bal+= pf
        else:
            exit_price= final_price*(1+ slippage)
            comm_exit= abs(pos['size'])* exit_price* commission_rate
            pf= abs(pos['size'])*(pos['entry_price']- exit_price)- comm_exit
            hrs= (df.iloc[-1]['timestamp']- pos['entry_time'])/3600.0
            fund= abs(pos['size'])* pos['entry_price']* FUNDING_RATE*hrs
            pf-= fund
            bal+= pf
        trades.append({
            'entry_time':pos['entry_time'],
            'exit_time': df.iloc[-1]['timestamp'],
            'side':pos['side'],
            'entry_price':pos['entry_price'],
            'stop_loss':pos['stop_loss'],
            'take_profit':pos['take_profit'],
            'exit_price':exit_price,
            'exit_reason': "Final",
            'return': (exit_price/pos['entry_price']-1)*(1 if pos['side']=='long' else -1),
            'duration': df.iloc[-1]['timestamp']- pos['entry_time']
        })
        eq_curve[-1]=(df.iloc[-1]['timestamp'], bal)

    final_profit= bal- init_bal
    net_pct= (final_profit/init_bal)*100.0
    eff_net_pct= net_pct

    # inactivity penalty
    tot_inact_pen= 0.0
    if trades:
        te_sorted= sorted([t['entry_time'] for t in trades])
        gaps=[]
        start_gap= te_sorted[0]- df.iloc[0]['timestamp']
        if start_gap>0:
            gaps.append(start_gap)
        for i in range(1,len(te_sorted)):
            gp= te_sorted[i]- te_sorted[i-1]
            if gp>0:
                gaps.append(gp)
        end_gap= df.iloc[-1]['timestamp']- te_sorted[-1]
        if end_gap>0:
            gaps.append(end_gap)
        for g in gaps:
            tot_inact_pen+= inactivity_exponential_penalty(g)
    else:
        if len(df)>1:
            total_s= df.iloc[-1]['timestamp']- df.iloc[0]['timestamp']
            tot_inact_pen+= inactivity_exponential_penalty(total_s)

    days_in_pf= compute_days_in_profit(eq_curve, init_bal)

    eqdf= pd.DataFrame(eq_curve, columns=["timestamp","balance"])
    eqdf["dt"]= pd.to_datetime(eqdf["timestamp"], unit="s")
    eqdf.set_index("dt", inplace=True)
    eqdf= eqdf.resample("1D").last().dropna()
    if len(eqdf)<2:
        sharpe=0.0
        mdd=0.0
    else:
        dret= eqdf["balance"].pct_change().dropna()
        mu= dret.mean()
        sigma= dret.std()
        sharpe= (mu*252)/(sigma* np.sqrt(252)) if sigma>1e-12 else 0.0
        rollmax= eqdf["balance"].cummax()
        dd= (eqdf["balance"]-rollmax)/rollmax
        mdd= dd.min()

    net_score= net_pct/100.0
    shr_score= sharpe
    dd_pen= abs(mdd)
    trade_count= len(trades)
    trade_term= trade_count* delta
    # final composite
    # composite_reward= (alpha* net_score + beta* shr_score - gamma* dd_pen + trade_term)
    # composite_reward-= tot_inact_pen
    composite_reward = ( # new stuff
        alpha * net_score + 
        beta * shr_score * 2 +  # Sharper reward for risk-adjusted returns
        gamma * (1 - abs(mdd)) +  # Better drawdown handling
        trade_term * 3 +  # Stronger incentive for reasonable trade frequency
        (days_in_pf/365) * 1000  # Strong bonus for consistent profitability
    )
    return {
        "net_pct": net_pct,
        "trades": trade_count,
        "effective_net_pct": eff_net_pct,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "equity_curve": eq_curve,
        "trade_details": trades,
        "composite_reward": composite_reward,
        "inactivity_penalty": tot_inact_pen,
        "days_without_trading": 0.0,
        "days_in_profit": days_in_pf
    }

###############################################################################
# EnsembleModel
###############################################################################
class EnsembleModel:
    def __init__(self, device, n_models=2, lr=3e-4, weight_decay=1e-4):
        self.device = device

        # (6) We changed TradingModel to bigger capacity above
        self.models = [TradingModel().to(device) for _ in range(n_models)]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay) for m in self.models]
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0,2.0,0.8]).to(device))
        self.mse_loss_fn = nn.MSELoss()
        amp_on = device.type == 'cuda'
        self.scaler = GradScaler(enabled=amp_on,
                                 device=(device.type if amp_on else 'cpu'))
        self.best_val_loss = float('inf')
        self.best_composite_reward = float('-inf')
        self.best_state_dicts = None
        self.train_steps=0
        self.reward_loss_weight=0.2

        # (3) We'll do dynamic patience mechanism, so this is an initial
        self.patience_counter=0

        self.schedulers = [ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5)
                           for opt in self.optimizers]

        # store global indicator hyperparams that meta-agent can change
        self.indicator_hparams = IndicatorHyperparams(
            rsi_period=14,
            sma_period=10,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9
        )

    def train_one_epoch(self, dl_train, dl_val, data_full, stop_event=None):
        global global_equity_curve, global_backtest_profit
        global global_inactivity_penalty, global_composite_reward
        global global_days_without_trading, global_trade_details
        global global_days_in_profit
        global global_sharpe, global_max_drawdown, global_net_pct, global_num_trades
        global global_yearly_stats_table

        global global_best_equity_curve, global_best_sharpe, global_best_drawdown
        global global_best_net_pct, global_best_num_trades, global_best_inactivity_penalty
        global global_best_composite_reward, global_best_days_in_profit
        global global_best_lr, global_best_wd

        current_result= robust_backtest(self, data_full)

        global_equity_curve = current_result["equity_curve"]
        global_backtest_profit.append(current_result["effective_net_pct"])
        global_inactivity_penalty = current_result["inactivity_penalty"]
        global_composite_reward = current_result["composite_reward"]
        global_days_without_trading = current_result["days_without_trading"]
        global_trade_details = current_result["trade_details"]
        global_days_in_profit = current_result["days_in_profit"]
        global_sharpe = current_result["sharpe"]
        global_max_drawdown = current_result["max_drawdown"]
        global_net_pct = current_result["net_pct"]
        global_num_trades = current_result["trades"]

        dfy, table_str= compute_yearly_stats(
            current_result["equity_curve"],
            current_result["trade_details"],
            initial_balance=100.0
        )
        global_yearly_stats_table= table_str

        # (4) We'll define an extended state for the meta-agent, but that happens in meta_control_loop.
        # For the main training, we keep your code.

        # The composite reward is used as training target
        scaled_target= torch.tanh(torch.tensor(current_result["composite_reward"]/100.0,
                                               dtype=torch.float32, device=self.device))
        total_loss=0.0
        nb=0
        for m in self.models:
            m.train()
        for batch_x, batch_y in dl_train:
            bx = batch_x.to(self.device, non_blocking=True).contiguous().clone()
            by = batch_y.to(self.device, non_blocking=True)
            batch_loss=0.0
            for model,opt_ in zip(self.models,self.optimizers):
                opt_.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    with autocast(device_type=self.device.type,
                                   enabled=(self.device.type=="cuda")):
                        logits, _, pred_reward = model(bx)
                        ce_loss= self.criterion(logits, by)
                        reward_loss= self.mse_loss_fn(pred_reward, scaled_target.expand_as(pred_reward))
                        loss= ce_loss + self.reward_loss_weight* reward_loss
                    self.scaler.scale(loss).backward()
                self.scaler.unscale_(opt_)
                torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=0.5,  # Changed from 0.5 to max_norm=0.5
                norm_type=2.0  # Add norm type
                )
                self.scaler.step(opt_)
                self.scaler.update()
                batch_loss+= loss.item()
            total_loss+= batch_loss/ len(self.models)
            nb+=1
        train_loss= total_loss/ nb

        val_loss= self.evaluate_val_loss(dl_val) if dl_val else None
        if val_loss is not None:
            for sch in self.schedulers:
                sch.step(val_loss)

        # (2) Adjust Reward Penalties => less harsh for zero trades/ negative net
        trades_now= len(current_result["trade_details"])
        cur_reward= current_result["composite_reward"]

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
        avg_improvement = np.mean(global_backtest_profit[-10:]) if len(global_backtest_profit)>=10 else 0
        if cur_reward > self.best_composite_reward:
            self.best_composite_reward= cur_reward
            self.patience_counter=0
            self.best_state_dicts= [m.state_dict() for m in self.models]
            self.save_best_weights("best_model_weights.pth")
        else:
            self.patience_counter+=1
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
                    new_lr = np.random.choice([1e-5, 1e-4, 1e-3])  # More radical LR changes
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
                        self.models = [TradingModel(
                            hidden_size=np.random.choice([128, 256]),
                            dropout=np.random.uniform(0.3, 0.6)
                        ).to(self.device) for _ in range(len(self.models))]
                    self.patience_counter=0

        # track best net
        if current_result["net_pct"]> global_best_net_pct:
            global_best_equity_curve= current_result["equity_curve"]
            global_best_sharpe= current_result["sharpe"]
            global_best_drawdown= current_result["max_drawdown"]
            global_best_net_pct= current_result["net_pct"]
            global_best_num_trades= trades_now
            global_best_inactivity_penalty= current_result["inactivity_penalty"]
            global_best_composite_reward= cur_reward
            global_best_days_in_profit= current_result["days_in_profit"]
            global_best_lr= self.optimizers[0].param_groups[0]['lr']
            global_best_wd= self.optimizers[0].param_groups[0].get('weight_decay', 0)

        self.train_steps+=1
        return train_loss, val_loss

    def evaluate_val_loss(self, dl_val):
        for m in self.models:
            m.train()
        losses=[]
        with torch.no_grad():
            for bx, by in dl_val:
                bx= bx.to(self.device)
                by= by.to(self.device)
                model_losses=[]
                for mm in self.models:
                    lg,_,_ = mm(bx)
                    l_= self.criterion(lg, by)
                    model_losses.append(l_.item())
                losses.append(np.mean(model_losses))
        return float(np.mean(losses))

    def predict(self, x):
        with torch.no_grad():
            outs=[]
            for m in self.models:
                lg,_,_= m(x.to(self.device))
                p_= torch.softmax(lg, dim=1).cpu().numpy()
                outs.append(p_)
            avgp= np.mean(outs, axis=0)
            idx= int(np.argmax(avgp[0]))
            conf= float(avgp[0][idx])
            return idx, conf, None

    def vectorized_predict(self, windows_tensor, batch_size=256):
        with torch.no_grad():
            all_probs=[]
            n_= windows_tensor.shape[0]
            for i in range(0,n_,batch_size):
                batch= windows_tensor[i:i+batch_size]
                batch_probs=[]
                for m in self.models:
                    lg, tpars, _= m(batch)
                    pr_= torch.softmax(lg, dim=1).cpu()
                    batch_probs.append(pr_)
                avg_probs= torch.mean(torch.stack(batch_probs), dim=0)
                all_probs.append(avg_probs)
            ret_probs= torch.cat(all_probs, dim=0)
            idxs= ret_probs.argmax(dim=1)
            confs= ret_probs.max(dim=1)[0]
            # dummy param
            dummy_t = {
                "risk_fraction": torch.tensor([0.1]),
                "sl_multiplier": torch.tensor([5.0]),
                "tp_multiplier": torch.tensor([5.0])
            }
            return idxs.cpu(), confs.cpu(), dummy_t

    def optimize_models(self, dummy_input):
        pass

    def save_best_weights(self, path="best_model_weights.pth"):
        if not self.best_state_dicts:
            return
        torch.save({
            "best_composite_reward": self.best_composite_reward,
            "state_dicts": self.best_state_dicts
        }, path)

    def load_best_weights(self, path="best_model_weights.pth", data_full=None):
        if os.path.isfile(path):
            try:
                ckpt= torch.load(path, map_location=self.device)
                self.best_composite_reward= ckpt.get("best_composite_reward", float('-inf'))
                self.best_state_dicts= ckpt["state_dicts"]
                for m, sd in zip(self.models, self.best_state_dicts):
                    m.load_state_dict(sd, strict=False)
                if data_full and len(data_full)>24:
                    loaded_result= robust_backtest(self, data_full)
                    global global_best_equity_curve, global_best_sharpe, global_best_drawdown
                    global global_best_net_pct, global_best_num_trades, global_best_inactivity_penalty
                    global global_best_composite_reward, global_best_days_in_profit
                    global global_best_lr, global_best_wd
                    global_best_equity_curve= loaded_result["equity_curve"]
                    global_best_sharpe= loaded_result["sharpe"]
                    global_best_drawdown= loaded_result["max_drawdown"]
                    global_best_net_pct= loaded_result["net_pct"]
                    global_best_num_trades= loaded_result["trades"]
                    global_best_inactivity_penalty= loaded_result["inactivity_penalty"]
                    global_best_composite_reward= loaded_result["composite_reward"]
                    global_best_days_in_profit= loaded_result["days_in_profit"]
                    global_best_lr= self.optimizers[0].param_groups[0]['lr']
                    global_best_wd= self.optimizers[0].param_groups[0].get('weight_decay', 0)
            except:
                pass

###############################################################################
# Tkinter GUI
###############################################################################
class TradingGUI:
    def __init__(self, root, ensemble):
        self.root = root
        self.ensemble = ensemble
        self.root.title("Complex AI Trading w/ Robust Backtest + Live Phemex")
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.frame_train = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_train, text="Training vs. Validation")
        self.fig_train, (self.ax_loss, self.ax_equity_train) = plt.subplots(2, 1, figsize=(5,6))
        self.canvas_train = FigureCanvasTkAgg(self.fig_train, master=self.frame_train)
        self.canvas_train.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.frame_live = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_live, text="Phemex Live Price")
        self.fig_live, self.ax_live = plt.subplots(figsize=(5,3))
        self.canvas_live = FigureCanvasTkAgg(self.fig_live, master=self.frame_live)
        self.canvas_live.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.frame_backtest = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_backtest, text="Backtest Results")
        self.fig_backtest, self.ax_net_profit = plt.subplots(figsize=(5,3))
        self.canvas_backtest = FigureCanvasTkAgg(self.fig_backtest, master=self.frame_backtest)
        self.canvas_backtest.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.frame_details = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_details, text="Attention Weights")
        self.fig_details, self.ax_details = plt.subplots(figsize=(5,3))
        self.canvas_details = FigureCanvasTkAgg(self.fig_details, master=self.frame_details)
        self.canvas_details.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.frame_trades = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_trades, text="Trade Details")
        self.trade_text = tk.Text(self.frame_trades, width=50, height=20)
        self.trade_text.pack(fill=tk.BOTH, expand=True)

        self.frame_yearly_perf = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_yearly_perf, text="Yearly Perf")
        self.yearly_perf_text = tk.Text(self.frame_yearly_perf, width=50, height=20)
        self.yearly_perf_text.pack(fill=tk.BOTH, expand=True)

        self.info_frame = ttk.Frame(root)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        disclaimer_text = ("NOT INVESTMENT ADVICE! Demo only.\nUse caution.")
        self.disclaimer_label = ttk.Label(self.info_frame, text=disclaimer_text,
            font=("Helvetica", 9, "italic"), foreground="darkred")
        self.disclaimer_label.grid(row=0,column=0,sticky=tk.W,padx=5,pady=2)

        self.pred_label = ttk.Label(self.info_frame, text="AI Prediction: N/A", font=("Helvetica", 12))
        self.pred_label.grid(row=1,column=0,sticky=tk.W,padx=5,pady=5)
        self.conf_label = ttk.Label(self.info_frame, text="Confidence: N/A", font=("Helvetica",12))
        self.conf_label.grid(row=2,column=0,sticky=tk.W,padx=5,pady=5)
        self.epoch_label= ttk.Label(self.info_frame, text="Training Steps: 0", font=("Helvetica",12))
        self.epoch_label.grid(row=3,column=0,sticky=tk.W,padx=5,pady=5)

        self.current_hyper_label=ttk.Label(self.info_frame, text="Current Hyperparameters:", font=("Helvetica",12,"underline"))
        self.current_hyper_label.grid(row=4,column=0,sticky=tk.W,padx=5,pady=5)
        self.lr_label= ttk.Label(self.info_frame, text="LR: N/A", font=("Helvetica",12))
        self.lr_label.grid(row=5,column=0,sticky=tk.W,padx=5,pady=5)
        self.atr_label= ttk.Label(self.info_frame, text=f"ATR: {global_ATR_period}", font=("Helvetica",12))
        self.atr_label.grid(row=6,column=0,sticky=tk.W,padx=5,pady=5)
        self.sl_label= ttk.Label(self.info_frame, text=f"SL: {global_SL_multiplier}", font=("Helvetica",12))
        self.sl_label.grid(row=7,column=0,sticky=tk.W,padx=5,pady=5)
        self.tp_label= ttk.Label(self.info_frame, text=f"TP: {global_TP_multiplier}", font=("Helvetica",12))
        self.tp_label.grid(row=8,column=0,sticky=tk.W,padx=5,pady=5)

        self.best_hyper_label= ttk.Label(self.info_frame, text="Best Hyperparameters:", font=("Helvetica",12,"underline"), foreground="darkgreen")
        self.best_hyper_label.grid(row=4,column=1,sticky=tk.W,padx=5,pady=5)
        self.best_lr_label= ttk.Label(self.info_frame, text="Best LR: N/A", font=("Helvetica",12), foreground="darkgreen")
        self.best_lr_label.grid(row=5,column=1,sticky=tk.W,padx=5,pady=5)
        self.best_wd_label= ttk.Label(self.info_frame, text="Weight Decay: N/A", font=("Helvetica",12), foreground="darkgreen")
        self.best_wd_label.grid(row=6,column=1,sticky=tk.W,padx=5,pady=5)

        self.current_sharpe_label = ttk.Label(self.info_frame, text="Sharpe: N/A", font=("Helvetica",12))
        self.current_sharpe_label.grid(row=9,column=0,sticky=tk.W,padx=5,pady=5)
        self.current_drawdown_label=ttk.Label(self.info_frame, text="Max DD: N/A", font=("Helvetica",12))
        self.current_drawdown_label.grid(row=10,column=0,sticky=tk.W,padx=5,pady=5)
        self.current_netprofit_label=ttk.Label(self.info_frame, text="Net Profit (%): N/A", font=("Helvetica",12))
        self.current_netprofit_label.grid(row=11,column=0,sticky=tk.W,padx=5,pady=5)
        self.current_trades_label= ttk.Label(self.info_frame, text="Trades: N/A", font=("Helvetica",12))
        self.current_trades_label.grid(row=12,column=0,sticky=tk.W,padx=5,pady=5)
        self.current_inactivity_label= ttk.Label(self.info_frame, text="Inactivity Penalty: N/A", font=("Helvetica",12))
        self.current_inactivity_label.grid(row=13,column=0,sticky=tk.W,padx=5,pady=5)
        self.current_composite_label= ttk.Label(self.info_frame, text="Composite: N/A", font=("Helvetica",12))
        self.current_composite_label.grid(row=14,column=0,sticky=tk.W,padx=5,pady=5)
        self.current_days_profit_label=ttk.Label(self.info_frame, text="Days in Profit: N/A", font=("Helvetica",12))
        self.current_days_profit_label.grid(row=15,column=0,sticky=tk.W,padx=5,pady=5)

        self.best_sharpe_label= ttk.Label(self.info_frame, text="Best Sharpe: N/A", font=("Helvetica",12), foreground="darkgreen")
        self.best_sharpe_label.grid(row=9,column=1,sticky=tk.W,padx=5,pady=5)
        self.best_drawdown_label=ttk.Label(self.info_frame, text="Best Max DD: N/A", font=("Helvetica",12), foreground="darkgreen")
        self.best_drawdown_label.grid(row=10,column=1,sticky=tk.W,padx=5,pady=5)
        self.best_netprofit_label=ttk.Label(self.info_frame, text="Best Net Pct: N/A", font=("Helvetica",12), foreground="darkgreen")
        self.best_netprofit_label.grid(row=11,column=1,sticky=tk.W,padx=5,pady=5)
        self.best_trades_label= ttk.Label(self.info_frame, text="Best Trades: N/A", font=("Helvetica",12), foreground="darkgreen")
        self.best_trades_label.grid(row=12,column=1,sticky=tk.W,padx=5,pady=5)
        self.best_inactivity_label=ttk.Label(self.info_frame, text="Best Inactivity: N/A", font=("Helvetica",12), foreground="darkgreen")
        self.best_inactivity_label.grid(row=13,column=1,sticky=tk.W,padx=5,pady=5)
        self.best_composite_label=ttk.Label(self.info_frame, text="Best Composite: N/A", font=("Helvetica",12), foreground="darkgreen")
        self.best_composite_label.grid(row=14,column=1,sticky=tk.W,padx=5,pady=5)
        self.best_days_profit_label=ttk.Label(self.info_frame, text="Best Days in Profit: N/A", font=("Helvetica",12), foreground="darkgreen")
        self.best_days_profit_label.grid(row=15,column=1,sticky=tk.W,padx=5,pady=5)

        self.frame_ai = ttk.Frame(root)
        self.frame_ai.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.ai_output_label= ttk.Label(self.frame_ai, text="Latest AI Adjustments:", font=("Helvetica",12,"bold"))
        self.ai_output_label.pack(anchor="n")
        self.ai_output_text = tk.Text(self.frame_ai, width=40, height=10, wrap="word")
        self.ai_output_text.pack(fill=tk.BOTH, expand=True)

        self.frame_ai_log = ttk.Frame(root)
        self.frame_ai_log.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.ai_log_label=ttk.Label(self.frame_ai_log, text="AI Adjustments Log:", font=("Helvetica",12,"bold"))
        self.ai_log_label.pack(anchor="n")
        self.ai_log_text = tk.Text(self.frame_ai_log, width=40, height=10, wrap="word")
        self.ai_log_text.pack(fill=tk.BOTH, expand=True)

        self.update_interval=2000
        self.root.after(self.update_interval, self.update_dashboard)

    def update_dashboard(self):
        global global_equity_curve, global_best_equity_curve
        self.ax_loss.clear()
        self.ax_loss.set_title("Training vs. Validation Loss")
        x1= range(1, len(global_training_loss)+1)
        self.ax_loss.plot(x1, global_training_loss, color='blue', marker='o', label='Train')
        val_filtered= [(i+1,v) for i,v in enumerate(global_validation_loss) if v is not None]
        if val_filtered:
            xv,yv= zip(*val_filtered)
            self.ax_loss.plot(xv,yv, color='orange', marker='x', label='Val')
        self.ax_loss.legend()

        self.ax_equity_train.clear()
        self.ax_equity_train.set_title("Equity: Current (red) vs Best (green)")
        try:
            valid_eq= [(t,b) for (t,b) in global_equity_curve if isinstance(t,(int,float))]
            if valid_eq:
                ts_, bs_= zip(*valid_eq)
                ts_dt= [datetime.datetime.fromtimestamp(t_) for t_ in ts_]
                self.ax_equity_train.plot(ts_dt, bs_, color='red', marker='.', label="Current")
            if global_best_equity_curve:
                best_eq= [(t,b) for (t,b) in global_best_equity_curve if isinstance(t,(int,float))]
                if best_eq:
                    t2,b2= zip(*best_eq)
                    t2dt= [datetime.datetime.fromtimestamp(x) for x in t2]
                    self.ax_equity_train.plot(t2dt, b2, color='green', marker='.', label="Best")
            self.ax_equity_train.legend()
        except:
            pass
        self.canvas_train.draw()

        self.ax_live.clear()
        self.ax_live.set_title("Phemex Live Price (1h)")
        try:
            times, closes= [],[]
            for bar in global_phemex_data:
                if len(bar)>=5 and bar[0]>0:
                    t_= bar[0]
                    c_= bar[4]
                    times.append(datetime.datetime.fromtimestamp(t_/1000))
                    closes.append(c_)
            if times and closes:
                self.ax_live.plot(times, closes, marker='o')
        except:
            pass
        self.canvas_live.draw()

        self.ax_net_profit.clear()
        self.ax_net_profit.set_title("Net Profit (%)")
        if global_backtest_profit:
            x2= range(1, len(global_backtest_profit)+1)
            self.ax_net_profit.plot(x2, global_backtest_profit, marker='o', color='green')
        self.canvas_backtest.draw()

        self.ax_details.clear()
        self.ax_details.set_title("Avg Attention Weights (placeholder)")
        if global_attention_weights_history:
            x_= list(range(1,len(global_attention_weights_history)+1))
            self.ax_details.plot(x_, global_attention_weights_history, marker='o', color='purple')
        self.canvas_details.draw()

        self.trade_text.delete("1.0", tk.END)
        if global_trade_details:
            self.trade_text.insert(tk.END, json.dumps(global_trade_details, indent=2))
        else:
            self.trade_text.insert(tk.END,"No Trade Details")

        self.yearly_perf_text.delete("1.0", tk.END)
        if global_yearly_stats_table:
            self.yearly_perf_text.insert(tk.END, global_yearly_stats_table)
        else:
            self.yearly_perf_text.insert(tk.END,"No yearly data")

        pred_str= global_current_prediction if global_current_prediction else "N/A"
        conf= global_ai_confidence if global_ai_confidence else 0.0
        steps= global_ai_epoch_count
        self.pred_label.config(text=f"AI Prediction: {pred_str}")
        self.conf_label.config(text=f"Confidence: {conf:.2f}")
        self.epoch_label.config(text=f"Training Steps: {steps}")

        current_lr= self.ensemble.optimizers[0].param_groups[0]['lr']
        self.lr_label.config(text=f"LR: {current_lr:.2e}")
        self.atr_label.config(text=f"ATR: {global_ATR_period}")
        self.sl_label.config(text=f"SL: {global_SL_multiplier}")
        self.tp_label.config(text=f"TP: {global_TP_multiplier}")
        self.best_lr_label.config(text=f"Best LR: {global_best_lr if global_best_lr else 'N/A'}")
        self.best_wd_label.config(text=f"Weight Decay: {global_best_wd if global_best_wd else 'N/A'}")

        self.current_sharpe_label.config(text=f"Sharpe: {global_sharpe:.2f}")
        self.current_drawdown_label.config(text=f"Max DD: {global_max_drawdown:.3f}")
        self.current_netprofit_label.config(text=f"Net Pct: {global_net_pct:.2f}")
        self.current_trades_label.config(text=f"Trades: {global_num_trades}")
        if global_inactivity_penalty is not None:
            self.current_inactivity_label.config(text=f"Inact: {global_inactivity_penalty:.2f}")
        else:
            self.current_inactivity_label.config(text="Inactivity Penalty: N/A")
        if global_composite_reward is not None:
            self.current_composite_label.config(text=f"Comp: {global_composite_reward:.2f}")
        else:
            self.current_composite_label.config(text="Current Composite: N/A")
        if global_days_in_profit is not None:
            self.current_days_profit_label.config(text=f"Days in Profit: {global_days_in_profit:.2f}")
        else:
            self.current_days_profit_label.config(text="Current Days in Profit: N/A")

        self.best_sharpe_label.config(text=f"Best Sharpe: {global_best_sharpe:.2f}")
        self.best_drawdown_label.config(text=f"Best Max DD: {global_best_drawdown:.3f}")
        self.best_netprofit_label.config(text=f"Best Net Pct: {global_best_net_pct:.2f}")
        self.best_trades_label.config(text=f"Best Trades: {global_best_num_trades}")
        if global_best_inactivity_penalty is not None:
            self.best_inactivity_label.config(text=f"Best Inact: {global_best_inactivity_penalty:.2f}")
        else:
            self.best_inactivity_label.config(text="Best Inactivity Penalty: N/A")
        if global_best_composite_reward is not None:
            self.best_composite_label.config(text=f"Best Comp: {global_best_composite_reward:.2f}")
        else:
            self.best_composite_label.config(text="Best Composite: N/A")
        if global_best_days_in_profit is not None:
            self.best_days_profit_label.config(text=f"Best Days in Profit: {global_best_days_in_profit:.2f}")
        else:
            self.best_days_profit_label.config(text="Best Days in Profit: N/A")

        self.ai_output_text.delete("1.0", tk.END)
        self.ai_output_text.insert(tk.END, global_ai_adjustments)
        self.ai_log_text.delete("1.0", tk.END)
        self.ai_log_text.insert(tk.END, global_ai_adjustments_log)

        self.root.after(self.update_interval, self.update_dashboard)

###############################################################################
# Threads
###############################################################################
def csv_training_thread(ensemble, data, stop_event, config, use_prev_weights=True):
    from torch.utils.data import random_split, DataLoader
    import traceback
    global global_training_loss, global_validation_loss
    global global_ai_epoch_count

    try:
        ds_full = HourlyDataset(data, seq_len=24, threshold=GLOBAL_THRESHOLD)
        if len(ds_full)<10:
            logging.warning("Not enough data in CSV => exiting.")
            return
        if use_prev_weights:
            ensemble.load_best_weights("best_model_weights.pth", data_full=data)
        n_tot = len(ds_full)
        n_tr  = int(n_tot*0.9)
        n_val = n_tot-n_tr
        ds_train, ds_val = random_split(ds_full,[n_tr, n_val])
        pin = ensemble.device.type == 'cuda'
        workers = 2 if pin else 0
        dl_train = DataLoader(ds_train, batch_size=128, shuffle=True,
                             num_workers=workers, pin_memory=pin)
        dl_val = DataLoader(ds_val, batch_size=128, shuffle=False,
                           num_workers=workers, pin_memory=pin)

        adapt_live = bool(config.get("ADAPT_TO_LIVE",False))
        dummy_input = torch.randn(1,24,8,device=ensemble.device)
        ensemble.optimize_models(dummy_input)

        import talib

        while not stop_event.is_set():
            ensemble.train_steps+=1
            tl, vl = ensemble.train_one_epoch(dl_train, dl_val, data, stop_event)
            global_training_loss.append(tl)
            if vl is not None:
                global_validation_loss.append(vl)
            else:
                global_validation_loss.append(None)

            # quick "latest prediction"
            if len(data)>=24:
                tail = np.array(data[-24:], dtype=np.float64)
                closes= tail[:,4]
                sma= np.convolve(closes, np.ones(10)/10, mode='same')
                rsi= talib.RSI(closes, timeperiod=14)
                macd,_,_ = talib.MACD(closes)
                ext= []
                for i,row in enumerate(tail):
                    ext.append([row[1], row[2], row[3], row[4], row[5],
                                float(sma[i]), float(rsi[i]), float(macd[i])])
                ext= np.array(ext, dtype=np.float32)
                seq_t= torch.tensor(ext).unsqueeze(0).to(ensemble.device)
                idx, conf,_= ensemble.predict(seq_t)
                label_map= {0:"BUY",1:"SELL",2:"HOLD"}
                global global_current_prediction, global_ai_confidence
                global_current_prediction= label_map.get(idx,"N/A")
                global_ai_confidence= conf
                global_ai_epoch_count= ensemble.train_steps
                global_attention_weights_history.append(0)

            if adapt_live:
                changed=False
                while not live_bars_queue.empty():
                    new_b= live_bars_queue.get()
                    for bar in new_b:
                        ts,o_,h_,l_,c_,v_= bar
                        o_/=1e5; h_/=1e5; l_/=1e5; c_/=1e5; v_/=1e4
                        if ts> data[-1][0]:
                            data.append([ts,o_,h_,l_,c_,v_])
                            changed=True
                if changed:
                    ds_updated= HourlyDataset(data, seq_len=24, threshold=GLOBAL_THRESHOLD)
                    if len(ds_updated)>10:
                        nt_= len(ds_updated)
                        ntr_= int(nt_*0.9)
                        nv_= nt_- ntr_
                        ds_tr_, ds_val_= random_split(ds_updated,[ntr_,nv_])
                        pin = ensemble.device.type == 'cuda'
                        workers = 2 if pin else 0
                        dl_tr_ = DataLoader(ds_tr_, batch_size=128, shuffle=True,
                                           num_workers=workers, pin_memory=pin)
                        dl_val_ = DataLoader(ds_val_, batch_size=128, shuffle=False,
                                            num_workers=workers, pin_memory=pin)
                        ensemble.train_one_epoch(dl_tr_, dl_val_, data, stop_event)

            if ensemble.train_steps%5==0 and ensemble.best_state_dicts:
                ensemble.save_best_weights("best_model_weights.pth")

    except:
        traceback.print_exc()
        stop_event.set()

def phemex_live_thread(connector, stop_event, poll_interval=1.0):
    """Continuously fetch recent bars from Phemex at a configurable interval."""
    import traceback
    global global_phemex_data
    while not stop_event.is_set():
        try:
            bars = connector.fetch_latest_bars(limit=100)
            if bars:
                global_phemex_data = bars
                live_bars_queue.put(bars)
        except Exception:
            traceback.print_exc()
            stop_event.set()
        time.sleep(poll_interval)

###############################################################################
# Connector
###############################################################################
class PhemexConnector:
    def __init__(self, config):
        self.symbol = config.get("symbol","BTC/USDT")
        api_conf = config.get("API",{})
        self.api_key= api_conf.get("API_KEY_LIVE","")
        self.api_secret= api_conf.get("API_SECRET_LIVE","")
        default_type= api_conf.get("DEFAULT_TYPE","spot")
        import ccxt
        try:
            self.exchange= ccxt.phemex({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': default_type}
            })
        except Exception as e:
            logging.error(f"Error initializing exchange: {e}")
            sys.exit(1)
        self.exchange.load_markets()
        cands= generate_candidates(self.symbol)
        for c in cands:
            if c in self.exchange.markets:
                self.symbol= c
                break

    def fetch_latest_bars(self, limit=100):
        try:
            bars= self.exchange.fetch_ohlcv(self.symbol, timeframe='1h', limit=limit)
            return bars if bars else []
        except Exception as e:
            logging.error(f"Error fetching bars: {e}")
            return []

def generate_candidates(symbol):
    parts= re.split(r'[/:]', symbol)
    parts= [x for x in parts if x]
    cands= set()
    if len(parts)==2:
        base,quote= parts
        cands.update({
            f"{base}/{quote}",
            f"{base}{quote}",
            f"{base}:{quote}",
            f"{base}/USDT",
            f"{base}USDT"
        })
    else:
        cands.add(symbol)
    return list(cands)

###############################################################################
# Checkpoint
###############################################################################
def save_checkpoint():
    import json
    checkpoint = {
      "global_training_loss": global_training_loss,
      "global_validation_loss": global_validation_loss,
      "global_backtest_profit": global_backtest_profit,
      "global_equity_curve": global_equity_curve,
      "global_ai_adjustments_log": global_ai_adjustments_log,
      "global_hyperparameters": {
          "GLOBAL_THRESHOLD": GLOBAL_THRESHOLD,
          "global_SL_multiplier": global_SL_multiplier,
          "global_TP_multiplier": global_TP_multiplier,
          "global_ATR_period": global_ATR_period
      },
      "global_ai_epoch_count": global_ai_epoch_count,
      "gpt_memory_squirtle": gpt_memory_squirtle,
      "gpt_memory_wartorttle": gpt_memory_wartorttle,
      "gpt_memory_bigmanblastoise": gpt_memory_bigmanblastoise,
      "gpt_memory_moneymaker": gpt_memory_moneymaker,
      "global_attention_weights_history": global_attention_weights_history
    }
    with open("checkpoint.json","w") as f:
        json.dump(checkpoint, f, indent=2)

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
        self.action_space = list(itertools.product(
            self.lr_space, self.wd_space, self.rsi_space,
            self.sma_space, self.macd_fast_space, self.macd_slow_space,
            self.macd_sig_space, self.threshold_space
        ))
        # Randomly sample 1000 possible combinations instead of full cartesian product
        random.shuffle(self.action_space)
        self.action_space = self.action_space[:1000]

        # (4) State includes: [curr_reward, best_reward, sharpe, dd, trades, days_in_profit]
        self.state_dim=6
        self.n_actions= len(self.action_space)

        # (6) bigger or the same
        d_model= 32
        self.embed= nn.Linear(self.state_dim, d_model)
        self.pos_enc= PositionalEncoding(d_model)
        enc_layer= nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=64)
        self.transformer_enc= nn.TransformerEncoder(enc_layer, num_layers=2)
        self.policy_head= nn.Linear(d_model, self.n_actions)
        self.value_head= nn.Linear(d_model,1)

    def forward(self, x):
        x_emb= self.embed(x).unsqueeze(1).transpose(0,1).contiguous()
        x_pe= self.pos_enc(x_emb)
        x_enc= self.transformer_enc(x_pe)
        rep= x_enc.squeeze(0)
        pol= self.policy_head(rep)
        val= self.value_head(rep).squeeze(1)
        return pol, val

class MetaTransformerRL:
    def __init__(self, ensemble, lr=1e-3):
        self.model= TransformerMetaAgent()
        # expose the underlying action space so other methods don't
        # need to reach into the model attribute. Without this the
        # ``pick_action`` method fails with ``AttributeError``.
        self.action_space = self.model.action_space
        self.opt= optim.Adam(self.model.parameters(), lr=lr)
        self.gamma= 0.95
        self.ensemble= ensemble

        # (7) Scheduled exploration
        # Change epsilon parameters for longer exploration
        self.eps_start = 0.5  # Increased from 0.3
        self.eps_end = 0.15   # Increased from 0.05
        self.eps_decay = 0.995  # Slower decay
        # Add exploration reset mechanism
        self.exploration_reset_interval = 100 # not sure if this is used tho
        # Rest of init remains
        self.steps=0
        self.last_improvement=0

    def pick_action(self, state_np):
        # scheduled exploration
        self.steps+=1
        current_eps= max(self.eps_end, self.eps_start*(self.eps_decay**self.steps))
        noise_scale = max(0.1, 1.0 - self.steps/10000) # added random noise

        s= torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pol, val= self.model(s)
            dist= torch.distributions.Categorical(logits= pol)
        if random.random() < current_eps:
            # Add noise to action selection
            a_idx = np.random.choice(len(self.action_space))
            # Apply Gaussian noise to action parameters
            selected_action = list(self.action_space[a_idx])
            selected_action = [x + np.random.normal(0, noise_scale) for x in selected_action]
            a_idx = self._find_nearest_action(selected_action)
            lp = dist.log_prob(torch.tensor([a_idx]))
        else:
            a_idx= dist.sample().item()
            lp= dist.log_prob(torch.tensor([a_idx]))
        return a_idx, lp, val.item()

    def _find_nearest_action(self, candidate):
        """Return the index of the closest action in the action space."""
        if not isinstance(candidate, np.ndarray):
            candidate = np.array(candidate, dtype=float)
        actions = np.array(self.action_space, dtype=float)
        dists = np.linalg.norm(actions - candidate, axis=1)
        return int(np.argmin(dists))

    def update(self, state_np, action_idx, reward, next_state_np, logprob, value):
        s= torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        ns= torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0)
        pol_s, val_s= self.model(s)
        dist= torch.distributions.Categorical(logits= pol_s)
        lp_s= dist.log_prob(torch.tensor([action_idx]))
        with torch.no_grad():
            pol_ns, val_ns= self.model(ns)
        target= reward+ self.gamma* val_ns.item()
        advantage= target- val_s.item()
        loss_p= -lp_s* advantage
        loss_v= 0.5*(val_s- target)**2
        loss= loss_p+ loss_v
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if reward>0:
            self.last_improvement=0
        else:
            self.last_improvement+=1

    def apply_action(self, action_idx):
        # decode
        (lr_adj, wd_adj, rsi_adj, sma_adj, mf_adj, ms_adj, sig_adj, thr_adj)= self.model.action_space[action_idx]
        # 1) LR/WD
        old_lr= self.ensemble.optimizers[0].param_groups[0]['lr']
        new_lr= old_lr*(1+ lr_adj)
        new_lr= max(1e-6, min(new_lr, 1e-1))

        old_wd= self.ensemble.optimizers[0].param_groups[0].get('weight_decay',0)
        new_wd= old_wd*(1+ wd_adj)
        new_wd= max(0, min(new_wd, 1e-2))

        for opt_ in self.ensemble.optimizers:
            for grp in opt_.param_groups:
                grp['lr']= new_lr
                grp['weight_decay']= new_wd

        # 2) indicator hyperparams
        old_hp= self.ensemble.indicator_hparams
        new_rsi= old_hp.rsi_period+ rsi_adj
        new_sma= old_hp.sma_period+ sma_adj
        new_mf = old_hp.macd_fast+ mf_adj
        new_ms = old_hp.macd_slow+ ms_adj
        new_sig= old_hp.macd_signal+ sig_adj
        # 5) also let meta-agent change threshold
        new_threshold= GLOBAL_THRESHOLD + thr_adj

        self.ensemble.indicator_hparams= IndicatorHyperparams(
            rsi_period=new_rsi, sma_period=new_sma,
            macd_fast=new_mf, macd_slow=new_ms, macd_signal=new_sig
        )
        # If you want the dataset to re-init => we would do self.ensemble.dynamic_threshold = new_threshold
        # But for demonstration, we won't forcibly re-load the dataset now.

        return (new_lr, new_wd, new_rsi, new_sma, new_mf, new_ms, new_sig, new_threshold)

###############################################################################
# meta_control_loop
###############################################################################
def meta_control_loop(ensemble, dataset, agent, interval=5.0):
    global global_ai_adjustments, global_ai_adjustments_log
    global global_ai_epoch_count
    global global_composite_reward, global_best_composite_reward
    global global_sharpe, global_max_drawdown, global_num_trades, global_days_in_profit

    time.sleep(2.0)

    # old initial state was just 2 dims. Now we add (sharpe, dd, trades, days_in_profit)
    prev_r= global_composite_reward if global_composite_reward else 0.0
    best_r= global_best_composite_reward if global_best_composite_reward else 0.0
    st_sharpe= global_sharpe
    st_dd= global_max_drawdown
    st_trades= global_num_trades
    st_days= global_days_in_profit if global_days_in_profit else 0.0

    state= np.array([prev_r, best_r, st_sharpe, abs(st_dd), st_trades, st_days], dtype=np.float32)

    while True:
        if global_ai_epoch_count<1:
            time.sleep(1.0)
            continue

        curr_r= global_composite_reward if global_composite_reward else 0.0
        b_r= global_best_composite_reward if global_best_composite_reward else 0.0
        st_sharpe= global_sharpe
        st_dd= global_max_drawdown
        st_trades= global_num_trades
        st_days= global_days_in_profit if global_days_in_profit else 0.0
        new_state= np.array([curr_r, b_r, st_sharpe, abs(st_dd), st_trades, st_days], dtype=np.float32)

        a_idx, logp, val_s= agent.pick_action(state)
        (new_lr, new_wd, nrsi, nsma, nmacdf, nmacds, nmacdsig, nthr)= agent.apply_action(a_idx)

        time.sleep(interval)

        curr2= global_composite_reward if global_composite_reward else 0.0
        rew_delta= curr2- curr_r
        agent.update(state, a_idx, rew_delta, new_state, logp, val_s)

        summary= (f"Meta Update => s:{state} => a_idx={a_idx}, r:{rew_delta:.2f}\n"
                  f" newLR={new_lr:.2e}, newWD={new_wd:.2e}, rsi={nrsi}, sma={nsma}, "
                  f"macdF={nmacdf}, macdS={nmacds}, macdSig={nmacdsig}, threshold={nthr:.5f}")
        global_ai_adjustments_log+= "\n"+ summary
        global_ai_adjustments= summary

        state= new_state
        if agent.last_improvement>20:
            # forced random reinit
            for m in ensemble.models:
                for layer in m.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            agent.last_improvement=0
            msg= f"\n[Stagnation] Forced random reinit of primary model!\n"
            global_ai_adjustments_log+= msg

        time.sleep(0.5)

###############################################################################
# Main
###############################################################################
def main():
    global global_training_loss, global_validation_loss, global_backtest_profit, global_equity_curve
    global global_ai_adjustments_log, global_current_prediction, global_ai_confidence
    global global_ai_epoch_count, global_attention_weights_history, global_ai_adjustments

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    global_training_loss=[]
    global_validation_loss=[]
    global_backtest_profit=[]
    global_equity_curve=[]
    global_ai_adjustments_log="No adjustments yet"
    global_current_prediction=None
    global_ai_confidence=None
    global_ai_epoch_count=0
    global_attention_weights_history=[]
    global_ai_adjustments=""

    config={
        "CSV_PATH":"Gemini_BTCUSD_1h.csv",
        "symbol":"BTC/USDT",
        "ADAPT_TO_LIVE":False,
        "LIVE_POLL_INTERVAL":60.0,
        "API":{"API_KEY_LIVE":"","API_SECRET_LIVE":"","DEFAULT_TYPE":"spot"},
        "CHATGPT":{"API_KEY":""}
    }
    openai.api_key= config["CHATGPT"]["API_KEY"]
    csv_path= config["CSV_PATH"]
    data= load_csv_hourly(csv_path)

    use_prev_weights = False
    if os.path.isfile("best_model_weights.pth"):
        ans = input("Use previous best_model_weights.pth? [y/N]: ").strip().lower()
        if ans.startswith("y"):
            use_prev_weights = True
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = f"best_model_weights_backup_{ts}.pth"
            try:
                os.rename("best_model_weights.pth", backup)
                print(f"Existing weights backed up to {backup}")
            except OSError:
                print("Failed to backup existing weights")

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensemble= EnsembleModel(device=device, n_models=2, lr=3e-4, weight_decay=1e-4)
    connector= PhemexConnector(config)
    stop_event= threading.Event()

    train_th= threading.Thread(target=csv_training_thread,args=(ensemble,data,stop_event,config,use_prev_weights), daemon=True)
    train_th.start()

    poll_interval = config.get("LIVE_POLL_INTERVAL", 60.0)
    phemex_th = threading.Thread(
        target=phemex_live_thread,
        args=(connector, stop_event, poll_interval),
        daemon=True,
    )
    phemex_th.start()

    ds= HourlyDataset(data, seq_len=24, threshold=GLOBAL_THRESHOLD)
    meta_agent= MetaTransformerRL(ensemble=ensemble, lr=1e-3)
    meta_th= threading.Thread(target=lambda: meta_control_loop(ensemble, ds, meta_agent), daemon=True)
    meta_th.start()

    root= tk.Tk()
    gui= TradingGUI(root, ensemble)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

    stop_event.set()
    train_th.join()
    phemex_th.join()

    with open("training_history.json","w") as f:
        json.dump({
            "global_training_loss":global_training_loss,
            "global_validation_loss":global_validation_loss,
            "global_backtest_profit":global_backtest_profit
        }, f)
    ensemble.save_best_weights("best_model_weights.pth")
    save_checkpoint()

if __name__=="__main__":
    main()
