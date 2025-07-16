"""Shared global state for threads and hyperparameters.

``model_lock`` serialises access to the ensemble models during training and
meta-agent updates. Worker threads communicate short messages via
``global_status_primary`` and ``global_status_secondary`` updated using
``set_status`` or ``status_sleep``.
"""

import collections
import json
import os
import threading

# Imports
###############################################################################
import time
from queue import Queue

import matplotlib
import numpy as np
import torch

matplotlib.use("TkAgg")

# Reduce default logging to warnings only
import logging

import openai

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s"
)
client = openai  # alias

# GUI scale factor, updated on startup
UI_SCALE = 1.0

###############################################################################
# Tunable hyperparameters (updated by the RL agent)
###############################################################################

###############################################################################
# Global Hyperparameters and Starting Values
###############################################################################
global_SL_multiplier = 5  # stop-loss multiplier
global_TP_multiplier = 5  # take-profit multiplier
global_ATR_period = 50  # ATR indicator lookback
global_VORTEX_period = 14  # vortex indicator lookback
global_CMF_period = 20  # Chaikin MF lookback
global_use_ATR = True  # legacy toggle, unused
global_use_VORTEX = True  # legacy toggle, unused
global_use_CMF = True  # legacy toggle, unused
global_RSI_period = 9  # RSI lookback
global_SMA_period = 10  # SMA lookback
global_MACD_fast = 12
global_MACD_slow = 26
global_MACD_signal = 9
global_EMA_period = 20
global_DONCHIAN_period = 20
global_KIJUN_period = 26
global_TENKAN_period = 9
global_DISPLACEMENT = 26
global_use_RSI = True  # legacy toggle, unused
global_use_SMA = True  # legacy toggle, unused
global_use_MACD = True  # legacy toggle, unused
global_use_EMA = True  # legacy toggle, unused
global_use_DONCHIAN = False  # legacy toggle, unused
global_use_KIJUN = False  # legacy toggle, unused
global_use_TENKAN = False  # legacy toggle, unused
global_use_DISPLACEMENT = False  # legacy toggle, unused
risk_fraction = 0.03  # fraction of equity to risk per trade
GLOBAL_THRESHOLD = 5e-5  # minimal allowed loss
global_min_hold_seconds = 1800  # enforced trade hold time

# Maximum exposure per side and total gross exposure
MAX_SIDE_EXPOSURE_PCT = 0.10  # max long or short allocation
MAX_GROSS_PCT = 0.12  # total exposure cap

# Number of CPU threads reserved for PyTorch and DataLoader workers
cpu_limit = max(1, os.cpu_count() - 2)  # reserved CPU threads

# Desired long/short fractions controlled by the meta agent
global_long_frac = 0.0  # fraction of equity allocated to longs
global_short_frac = 0.0  # fraction of equity allocated to shorts
# Gross USD value of each leg based on ``live_equity``
gross_long_usd = 0.0  # computed from long_frac * equity
gross_short_usd = 0.0  # computed from short_frac * equity

# Snapshot of the best-performing hyperparameters
global_best_params = {
    "SL_multiplier": global_SL_multiplier,
    "TP_multiplier": global_TP_multiplier,
    "ATR_period": global_ATR_period,
    "risk_fraction": risk_fraction,
    "learning_rate": 1e-4,
}

# Global performance metrics:
global_sharpe = 0.0  # rolling Sharpe ratio
global_max_drawdown = 0.0  # worst equity drop
global_net_pct = 0.0  # cumulative profit %
global_num_trades = 0
global_win_rate = 0.0
global_profit_factor = 0.0
global_avg_trade_duration = 0.0
global_avg_win = 0.0
global_avg_loss = 0.0
global_inactivity_penalty = None  # RL penalty when idle
global_composite_reward = None  # most recent composite reward
global_composite_reward_ema = 0.0
global_days_without_trading = None
global_trade_details = []  # list of trade dicts
global_holdout_sharpe = 0.0  # validation Sharpe
global_holdout_max_drawdown = 0.0  # validation DD

# timeline ring buffer (keep last 300 bars ≈ 12.5 days on 1-h data)
timeline_depth = 300  # approx 12 days on 1h data
timeline_index = 0
timeline_ind_on = np.zeros((timeline_depth, 6), dtype=np.uint8)
timeline_trades = np.zeros(timeline_depth, dtype=np.uint8)  # 1 = in market

# Global best performance stats:
global_best_equity_curve = []  # list of floats
global_best_sharpe = 0.0
global_best_drawdown = 0.0
global_best_net_pct = 0.0
global_best_num_trades = 0
global_best_win_rate = 0.0
global_best_profit_factor = 0.0
global_best_avg_trade_duration = 0.0
global_best_avg_win = 0.0
global_best_avg_loss = 0.0
global_best_inactivity_penalty = None

global_best_composite_reward = float("-inf")
global_best_days_in_profit = None
global_best_trade_details = []  # list of trades from the best run

# Global best hyperparameters:
global_best_lr = None  # best learning rate so far
global_best_wd = None  # best weight decay so far

global_yearly_stats_table = ""
global_best_yearly_stats_table = ""
global_monthly_stats_table = ""
global_best_monthly_stats_table = ""

# Flags controlling components of the composite reward
use_net_term = True  # include net profit in reward
use_sharpe_term = True  # include Sharpe ratio
use_drawdown_term = True  # include drawdown term
use_trade_term = True  # include trade count
use_profit_days_term = True  # include days in profit
risk_filter_enabled = True  # training loss gating

###############################################################################
# GPT Memories (unchanged)
###############################################################################
gpt_memory_squirtle = []  # short-term
gpt_memory_wartorttle = []  # medium-term
gpt_memory_bigmanblastoise = []  # long-term
gpt_memory_moneymaker = []  # profit-focused

###############################################################################
# Additional global variables for meta agent logs
###############################################################################
global_ai_adjustments_log = "No adjustments yet"  # printable log string
global_ai_adjustments = ""  # latest tweak summary
global_ai_confidence = None  # RL confidence
epoch_count = 0  # training epochs completed
global_step = 0  # mini-batches seen
global_current_prediction = None  # latest model output
global_training_loss = []
global_validation_loss = []
global_backtest_profit = []
global_equity_curve = []
global_attention_weights_history = []  # per-step attention snapshots
global_attention_entropy_history = []
# Most recent attention weights averaged across heads
global_last_attention: list[list[float]] | None = None  # averaged heads
global_phemex_data = []
global_days_in_profit = 0.0
global_validation_summary = {}
live_bars_queue = Queue()  # feed of recent bars
# New event to trigger GUI refreshes
gui_event = threading.Event()
live_sharpe_history = collections.deque(maxlen=1000)  # GUI plot cache
live_drawdown_history = collections.deque(maxlen=1000)
trading_paused = False  # hot pause flag

# When ``True`` orders are routed to the Phemex testnet
use_sandbox = True  # True for testnet

# gating flag
nuclear_key_enabled = False  # gating flag
# Set to ``True`` when the GUI bypasses the nuclear key gate
nuke_armed = False  # GUI override

# Simple status indicator updated by threads

global_primary_status = "Initializing..."  # displayed in GUI
global_secondary_status = ""
global_progress_pct = 0.0

# Flag toggled by GUI when user enables live trading
live_trading_enabled = False  # set via GUI toggle

# Live account info updated by the trading loop
global_account_stats: dict = {}  # cached exchange balance
global_position_side: str | None = None
global_position_size: float = 0.0
start_equity: float = 0.0  # baseline when session started
live_equity: float = 0.0
live_trade_count: int = 0

# Production risk flag toggled after sufficient trade history
PROD_RISK = False  # True once sufficient history accumulated

# Flag toggled by training thread when new live weights are ready
live_weights_updated: bool = False  # set when new weights available

# When ``False`` worker threads pause their main loops
bot_running: bool = True  # pause flag for workers

# Singleton hedge book managing open long/short legs
hedge_book = None  # populated with HedgeBook instance


# Protect shared state across threads
state_lock = threading.Lock()  # guards shared state
# New: lock used when mutating model parameters
model_lock = threading.Lock()


def set_cpu_limit(n: int) -> None:
    """Update ``cpu_limit`` and adjust Torch and OpenMP threads."""

    n = max(1, int(n))
    global cpu_limit
    with state_lock:
        cpu_limit = n
    try:
        torch.set_num_threads(n)
    except RuntimeError as exc:  # pragma: no cover - threads already started
        logging.warning("CPU thread update failed: %s", exc)
    os.environ["OMP_NUM_THREADS"] = str(n)


def set_status(msg: str, secondary: str | None = None) -> None:
    """Thread-safe update of status messages."""
    with state_lock:
        global global_primary_status, global_secondary_status
        global_primary_status = msg
        if secondary is not None:
            global_secondary_status = secondary


def get_status() -> str:
    """Return the primary status message in a thread-safe manner."""
    with state_lock:
        return global_primary_status


def get_status_full() -> tuple[str, str]:
    """Return ``(primary, secondary)`` with epoch number fallback."""
    with state_lock:
        secondary = (
            global_secondary_status
            if global_secondary_status
            else f"epoch {epoch_count}"
        )
        return global_primary_status, secondary


def inc_epoch() -> None:
    """Increment the global epoch counter safely."""
    global epoch_count
    with state_lock:
        epoch_count += 1


def inc_step() -> None:
    """Increment the global mini-batch counter safely."""
    global global_step
    with state_lock:
        global_step += 1


def get_warmup_step() -> int:
    """Return the persisted warm-up counter."""
    try:
        with open("warmup.json", "r") as f:
            return int(json.load(f).get("step", 0))
    except Exception:
        return 0


def bump_warmup() -> int:
    """Increment and persist the warm-up counter."""
    val = get_warmup_step() + 1
    try:
        with open("warmup.json", "w") as f:
            json.dump({"step": val}, f)
    except Exception:
        pass
    return val


def get_trade_count() -> int:
    """Return the persisted trade counter from ``warmup.json``."""

    try:
        with open("warmup.json", "r") as f:
            return int(json.load(f).get("trades", 0))
    except Exception:
        return 0


def set_nuclear_key(enabled: bool) -> None:
    """Enable or disable the nuclear key trading gate and log the change."""
    global nuclear_key_enabled
    with state_lock:
        nuclear_key_enabled = enabled
    status = "ENABLED" if enabled else "DISABLED"
    logging.info("NUCLEAR_KEY_%s", status)


def is_nuclear_key_enabled() -> bool:
    """Return ``True`` when the nuclear key gate is active."""
    with state_lock:
        return nuclear_key_enabled


def set_risk_filter_enabled(enabled: bool) -> None:
    """Toggle the training risk filter."""
    global risk_filter_enabled
    with state_lock:
        risk_filter_enabled = enabled
    status = "ENABLED" if enabled else "DISABLED"
    logging.info("RISK_FILTER_%s", status)


def is_risk_filter_enabled() -> bool:
    """Return ``True`` when the risk filter is active."""
    with state_lock:
        return risk_filter_enabled


def set_live_weights_updated(value: bool) -> None:
    """Set the flag signalling updated live weights."""
    global live_weights_updated
    with state_lock:
        live_weights_updated = value


def pop_live_weights_updated() -> bool:
    """Return and reset the live weight update flag."""
    global live_weights_updated
    with state_lock:
        val = live_weights_updated
        live_weights_updated = False
    return val


def set_bot_running(state: bool) -> None:
    """Toggle the global run flag used by worker threads."""
    global bot_running
    with state_lock:
        bot_running = state


def is_bot_running() -> bool:
    """Return ``True`` when the bot is not paused."""
    with state_lock:
        return bot_running


###############################################################################
# Helper used by worker threads to show countdowns while sleeping
###############################################################################


def status_sleep(primary: str, secondary: str, seconds: float) -> None:
    """Sleep in 1s increments and update status fields."""

    end = time.monotonic() + seconds
    total = seconds
    with state_lock:
        global global_progress_pct
        global_progress_pct = 0.0
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break

        pct = int(100 * (total - remaining) / total)
        with state_lock:
            global_progress_pct = pct

        set_status(primary, f"{secondary} ({int(remaining)}s)".strip())

        time.sleep(1)

    with state_lock:
        global_progress_pct = 100


def cancel_open_orders() -> None:
    """Placeholder to cancel all outstanding orders."""
    logging.info("CANCEL_OPEN_ORDERS")
    global global_position_side, global_position_size
    global_position_side = None
    global_position_size = 0.0


def close_position() -> None:
    """Placeholder to close any open position."""
    logging.info("CLOSE_POSITION")
    global global_position_side, global_position_size
    global_position_side = None
    global_position_size = 0.0


def update_trade_params(sl_mult: float, tp_mult: float) -> None:
    """Update stop-loss and take-profit multipliers."""
    global global_SL_multiplier, global_TP_multiplier
    with state_lock:
        global_SL_multiplier = sl_mult
        global_TP_multiplier = tp_mult


def sync_globals(hp, ind_hp) -> None:
    """Synchronise hyperparams with module-level globals."""

    global global_SL_multiplier, global_TP_multiplier
    global global_ATR_period, global_VORTEX_period, global_CMF_period
    global global_RSI_period, global_SMA_period
    global global_MACD_fast, global_MACD_slow, global_MACD_signal
    global global_EMA_period, global_DONCHIAN_period
    global global_KIJUN_period, global_TENKAN_period, global_DISPLACEMENT
    global global_use_ATR, global_use_VORTEX, global_use_CMF
    global global_use_RSI, global_use_SMA, global_use_MACD
    global global_use_EMA, global_use_DONCHIAN
    global global_use_KIJUN, global_use_TENKAN, global_use_DISPLACEMENT
    global global_long_frac, global_short_frac
    global gross_long_usd, gross_short_usd
    with state_lock:
        global_SL_multiplier = hp.sl
        global_TP_multiplier = hp.tp
        global_ATR_period = ind_hp.atr_period
        global_VORTEX_period = ind_hp.vortex_period
        global_CMF_period = ind_hp.cmf_period
        global_EMA_period = ind_hp.ema_period
        global_DONCHIAN_period = ind_hp.donchian_period
        global_KIJUN_period = ind_hp.kijun_period
        global_TENKAN_period = ind_hp.tenkan_period
        global_DISPLACEMENT = ind_hp.displacement
        global_RSI_period = ind_hp.rsi_period
        global_SMA_period = ind_hp.sma_period
        global_MACD_fast = ind_hp.macd_fast
        global_MACD_slow = ind_hp.macd_slow
        global_MACD_signal = ind_hp.macd_signal
        global_use_ATR = ind_hp.use_atr
        global_use_VORTEX = ind_hp.use_vortex
        global_use_CMF = ind_hp.use_cmf
        global_use_RSI = ind_hp.use_rsi
        global_use_SMA = ind_hp.use_sma
        global_use_MACD = ind_hp.use_macd
        global_use_EMA = ind_hp.use_ema
        global_use_DONCHIAN = ind_hp.use_donchian
        global_use_KIJUN = ind_hp.use_kijun
        global_use_TENKAN = ind_hp.use_tenkan
        global_use_DISPLACEMENT = ind_hp.use_displacement
        global_long_frac = hp.long_frac
        global_short_frac = hp.short_frac
        gross_long_usd = global_long_frac * live_equity
        gross_short_usd = global_short_frac * live_equity
