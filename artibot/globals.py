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
global_inactivity_penalty = None
global_composite_reward = None
global_days_without_trading = None
global_trade_details = []
global_days_in_profit = None
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
live_bars_queue = queue.Queue()

# Simple status indicator updated by threads
global_status_primary = "Initializing"
global_status_secondary = ""

# Protect shared state across threads
state_lock = threading.Lock()
# New: lock used when mutating model parameters
model_lock = threading.Lock()


def set_status(primary: str, secondary: str) -> None:
    """Thread-safe update of status fields."""
    with state_lock:
        global global_status_primary, global_status_secondary
        global_status_primary = primary
        global_status_secondary = secondary


def get_status() -> str:
    """Return the combined status string in a thread-safe manner."""
    with state_lock:
        return f"{global_status_primary} {global_status_secondary}".strip()


def get_status_full() -> tuple[str, str]:
    """Return both status fields."""
    with state_lock:
        return global_status_primary, global_status_secondary


def inc_epoch() -> None:
    """Increment the global epoch counter safely."""
    global epoch_count
    with state_lock:
        epoch_count += 1


###############################################################################
# Helper used by worker threads to show countdowns while sleeping
###############################################################################
def status_sleep(primary: str, secondary: str, seconds: float) -> None:
    """Sleep in 1s increments and update status fields."""
    end = time.monotonic() + seconds
    while True:
        remaining = int(end - time.monotonic())
        if remaining <= 0:
            break
        set_status(primary, f"{secondary} ({remaining}s)".strip())
        time.sleep(1)
