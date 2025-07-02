"""Shared global state for threads and hyperparameters.

``model_lock`` serialises access to the ensemble models during training and
meta-agent updates. Worker threads communicate short messages via
``global_status_primary`` and ``global_status_secondary`` updated using
``set_status`` or ``status_sleep``.
"""

# Imports
###############################################################################
import time
import threading
from queue import Queue
import collections
import numpy as np
import matplotlib
import os
import torch
import json

matplotlib.use("TkAgg")

try:
    pass
except Exception:  # fallback for torch<2.2
    pass
import openai

# Reduce default logging to warnings only
import logging

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s"
)
client = openai  # alias

# GUI scale factor, updated on startup
UI_SCALE = 1.0

###############################################################################
# Global Hyperparameters and Starting Values
###############################################################################
global_SL_multiplier = 5
global_TP_multiplier = 5
global_ATR_period = 50
global_VORTEX_period = 14
global_CMF_period = 20
global_use_ATR = True
global_use_VORTEX = True
global_use_CMF = True
global_RSI_period = 9
global_SMA_period = 10
global_MACD_fast = 12
global_MACD_slow = 26
global_MACD_signal = 9
global_EMA_period = 20
global_DONCHIAN_period = 20
global_KIJUN_period = 26
global_TENKAN_period = 9
global_DISPLACEMENT = 26
global_use_RSI = True
global_use_SMA = True
global_use_MACD = True
global_use_EMA = True
global_use_DONCHIAN = False
global_use_KIJUN = False
global_use_TENKAN = False
global_use_DISPLACEMENT = False
risk_fraction = 0.03
GLOBAL_THRESHOLD = 5e-5
global_min_hold_seconds = 1800

# Maximum exposure per side and total gross exposure
MAX_SIDE_EXPOSURE_PCT = 0.10
MAX_GROSS_PCT = 0.12

# Number of CPU threads reserved for PyTorch and DataLoader workers
cpu_limit = max(1, os.cpu_count() - 2)

# Desired long/short fractions controlled by the meta agent
global_long_frac = 0.0
global_short_frac = 0.0
# Gross USD value of each leg based on ``live_equity``
gross_long_usd = 0.0
gross_short_usd = 0.0

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
global_win_rate = 0.0
global_profit_factor = 0.0
global_avg_trade_duration = 0.0
global_avg_win = 0.0
global_avg_loss = 0.0
global_inactivity_penalty = None
global_composite_reward = None
global_composite_reward_ema = 0.0
global_days_without_trading = None
global_trade_details = []
global_holdout_sharpe = 0.0
global_holdout_max_drawdown = 0.0

# timeline ring buffer (keep last 300 bars ≈ 12.5 days on 1-h data)
timeline_depth = 300
timeline_index = 0
timeline_ind_on = np.zeros((timeline_depth, 6), dtype=np.uint8)
timeline_trades = np.zeros(timeline_depth, dtype=np.uint8)  # 1 = in market

# Global best performance stats:
global_best_equity_curve = []
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
global_best_composite_reward = None
global_best_days_in_profit = None

# Global best hyperparameters:
global_best_lr = None
global_best_wd = None

global_yearly_stats_table = ""
global_best_yearly_stats_table = ""
global_monthly_stats_table = ""
global_best_monthly_stats_table = ""

# Flags controlling components of the composite reward
use_net_term = True
use_sharpe_term = True
use_drawdown_term = True
use_trade_term = True
use_profit_days_term = True

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
epoch_count = 0
global_step = 0
global_current_prediction = None
global_training_loss = []
global_validation_loss = []
global_backtest_profit = []
global_equity_curve = []
global_attention_weights_history = []
global_attention_entropy_history = []
# Most recent attention weights averaged across heads
global_last_attention: list[list[float]] | None = None
global_phemex_data = []
global_days_in_profit = 0.0
global_validation_summary = {}
live_bars_queue = Queue()
live_sharpe_history = collections.deque(maxlen=1000)
live_drawdown_history = collections.deque(maxlen=1000)
trading_paused = False

# When ``True`` orders are routed to the Phemex testnet
use_sandbox = True

# gating flag
nuclear_key_enabled = False
# Set to ``True`` when the GUI bypasses the nuclear key gate
nuke_armed = False

# Simple status indicator updated by threads

global_primary_status = "Initializing..."
global_secondary_status = ""
global_progress_pct = 0.0

# Flag toggled by GUI when user enables live trading
live_trading_enabled = False

# Live account info updated by the trading loop
global_account_stats: dict = {}
global_position_side: str | None = None
global_position_size: float = 0.0
start_equity: float = 0.0
live_equity: float = 0.0
live_trade_count: int = 0

# Production risk flag toggled after sufficient trade history
PROD_RISK = False

# Flag toggled by training thread when new live weights are ready
live_weights_updated: bool = False

# When ``False`` worker threads pause their main loops
bot_running: bool = True

# Singleton hedge book managing open long/short legs
hedge_book = None


# Protect shared state across threads
state_lock = threading.Lock()
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
