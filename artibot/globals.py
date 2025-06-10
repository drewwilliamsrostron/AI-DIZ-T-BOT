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
import queue
import collections
import matplotlib

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

###############################################################################
# Global Hyperparameters and Starting Values
###############################################################################
global_SL_multiplier = 5
global_TP_multiplier = 5
global_ATR_period = 50
risk_fraction = 0.03
GLOBAL_THRESHOLD = 5e-5
global_min_hold_seconds = 1800

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
global_days_without_trading = None
global_trade_details = []
global_holdout_sharpe = 0.0
global_holdout_max_drawdown = 0.0

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
global_current_prediction = None
global_training_loss = []
global_validation_loss = []
global_backtest_profit = []
global_equity_curve = []
global_attention_weights_history = []
global_attention_entropy_history = []
global_phemex_data = []
global_days_in_profit = 0.0
global_validation_summary = {}
live_bars_queue = queue.Queue()
live_sharpe_history = collections.deque(maxlen=1000)
live_drawdown_history = collections.deque(maxlen=1000)
trading_paused = False

# gating flag
nuclear_key_enabled = False

# Simple status indicator updated by threads

global_primary_status = "Initializing..."
global_secondary_status = ""
global_progress_pct = 0.0

# Flag toggled by GUI when user enables live trading
live_trading_enabled = False


# Protect shared state across threads
state_lock = threading.Lock()
# New: lock used when mutating model parameters
model_lock = threading.Lock()


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


def set_nuclear_key(enabled: bool) -> None:
    """Enable or disable the nuclear key trading gate."""
    global nuclear_key_enabled
    with state_lock:
        nuclear_key_enabled = enabled


def is_nuclear_key_enabled() -> bool:
    """Return ``True`` when the nuclear key gate is active."""
    with state_lock:
        return nuclear_key_enabled


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


def close_position() -> None:
    """Placeholder to close any open position."""
    logging.info("CLOSE_POSITION")


def update_trade_params(sl_mult: float, tp_mult: float) -> None:
    """Update stop-loss and take-profit multipliers."""
    global global_SL_multiplier, global_TP_multiplier
    with state_lock:
        global_SL_multiplier = sl_mult
        global_TP_multiplier = tp_mult
