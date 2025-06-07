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

try:
    from torch.amp import autocast, GradScaler
except Exception:  # fallback for torch<2.2
    from torch.cuda.amp import autocast, GradScaler
import openai
from typing import NamedTuple
from sklearn.preprocessing import StandardScaler
import talib  # For RSI and MACD

# Reduce default logging to warnings only
import logging

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s"
)
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
    "learning_rate": 1e-4,
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

# Simple status indicator updated by threads
global_status_message = "Initializing..."

# Protect shared state across threads
state_lock = threading.Lock()


def set_status(msg: str) -> None:
    """Thread-safe update of ``global_status_message``."""
    with state_lock:
        global global_status_message
        global_status_message = msg


def get_status() -> str:
    """Return the current ``global_status_message`` in a thread-safe manner."""
    with state_lock:
        return global_status_message


###############################################################################
# Helper used by worker threads to show countdowns while sleeping
###############################################################################
def status_sleep(message: str, seconds: float):
    """Sleep in 1s increments and update ``global_status_message``."""
    end = time.monotonic() + seconds
    while True:
        remaining = int(end - time.monotonic())
        if remaining <= 0:
            break
        set_status(f"{message} ({remaining}s)")
        time.sleep(1)
