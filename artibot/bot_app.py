"""Entry point for running the trading bot.

This module starts the training loop, live market polling, the
meta–reinforcement learning agent and the Tkinter GUI.

Configuration is loaded from ``master_config.json`` in the repository root
and may be overridden by environment variables from a ``.env`` file.
"""

from .globals import *  # noqa: F401,F403,F405
from .dataset import load_csv_hourly, HourlyDataset
from .ensemble import EnsembleModel
from .training import (
    csv_training_thread,
    phemex_live_thread,
    PhemexConnector,
    save_checkpoint,
)
from .rl import MetaTransformerRL, meta_control_loop
from .gui import TradingGUI

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - installed via environment.py
    load_dotenv = None

# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "master_config.json")


def load_config(path: str = CONFIG_PATH) -> dict:
    """Return configuration merged with environment variables."""

    cfg = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    env_path = os.path.join(os.path.dirname(path), ".env")
    if load_dotenv and os.path.exists(env_path):
        load_dotenv(env_path)

    api = cfg.setdefault("API", {})
    chat = cfg.setdefault("CHATGPT", {})

    def _get(name: str, default: str | bool | None = None):
        return os.environ.get(name, default)

    api["API_KEY_LIVE"] = _get("API_KEY_LIVE", api.get("API_KEY_LIVE", ""))
    api["API_SECRET_LIVE"] = _get("API_SECRET_LIVE", api.get("API_SECRET_LIVE", ""))
    api["API_KEY_TEST"] = _get("API_KEY_TEST", api.get("API_KEY_TEST", ""))
    api["API_SECRET_TEST"] = _get("API_SECRET_TEST", api.get("API_SECRET_TEST", ""))
    api["DEFAULT_TYPE"] = _get("DEFAULT_TYPE", api.get("DEFAULT_TYPE", "spot"))
    api["LIVE_TRADING"] = (
        str(_get("LIVE_TRADING", api.get("LIVE_TRADING", True))).lower() == "true"
    )

    chat["API_KEY"] = _get("CHATGPT_API_KEY", chat.get("API_KEY", ""))

    cfg["symbol"] = _get("SYMBOL", cfg.get("symbol", "BTCUSD"))
    cfg["CSV_PATH"] = _get("CSV_PATH", cfg.get("CSV_PATH", "Gemini_BTCUSD_1h.csv"))
    cfg["ADAPT_TO_LIVE"] = (
        str(_get("ADAPT_TO_LIVE", cfg.get("ADAPT_TO_LIVE", True))).lower() == "true"
    )

    return cfg


def run_bot():
    global global_training_loss, global_validation_loss, global_backtest_profit, global_equity_curve
    global global_ai_adjustments_log, global_current_prediction, global_ai_confidence
    global global_ai_epoch_count, global_attention_weights_history, global_ai_adjustments
    global global_status_message

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global_training_loss = []
    global_validation_loss = []
    global_backtest_profit = []
    global_equity_curve = []
    global_ai_adjustments_log = "No adjustments yet"
    global_current_prediction = None
    global_ai_confidence = None
    global_ai_epoch_count = 0
    global_attention_weights_history = []
    global_ai_adjustments = ""

    config = load_config()
    openai.api_key = config["CHATGPT"]["API_KEY"]
    csv_path = config["CSV_PATH"]
    if not os.path.isabs(csv_path):
        here = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(here, "..", csv_path)
    csv_path = os.path.abspath(os.path.expanduser(csv_path))
    print(f"Loading CSV data from: {csv_path}")
    data = load_csv_hourly(csv_path)

    if len(data) < 10:
        print("Error: no usable CSV data found")
        global_status_message = "CSV load failed"
        return

    use_prev_weights = bool(config.get("USE_PREV_WEIGHTS", True))
    if os.path.isfile("best_model_weights.pth"):
        if not use_prev_weights:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = f"best_model_weights_backup_{ts}.pth"
            try:
                os.rename("best_model_weights.pth", backup)
                print(f"Existing weights backed up to {backup}")
            except OSError:
                print("Failed to backup existing weights")
        else:
            use_prev_weights = True
    else:
        use_prev_weights = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensemble = EnsembleModel(device=device, n_models=2, lr=3e-4, weight_decay=1e-4)
    connector = PhemexConnector(config)
    stop_event = threading.Event()

    train_th = threading.Thread(
        target=csv_training_thread,
        args=(ensemble, data, stop_event, config, use_prev_weights),
        daemon=True,
    )
    train_th.start()

    poll_interval = config.get("LIVE_POLL_INTERVAL", 60.0)
    phemex_th = threading.Thread(
        target=phemex_live_thread,
        args=(connector, stop_event, poll_interval),
        daemon=True,
    )
    phemex_th.start()

    ds = HourlyDataset(data, seq_len=24, threshold=GLOBAL_THRESHOLD)
    meta_agent = MetaTransformerRL(ensemble=ensemble, lr=1e-3)
    meta_th = threading.Thread(
        target=lambda: meta_control_loop(ensemble, ds, meta_agent), daemon=True
    )
    meta_th.start()

    root = tk.Tk()
    TradingGUI(root, ensemble)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

    stop_event.set()
    train_th.join()
    phemex_th.join()

    with open("training_history.json", "w") as f:
        json.dump(
            {
                "global_training_loss": global_training_loss,
                "global_validation_loss": global_validation_loss,
                "global_backtest_profit": global_backtest_profit,
            },
            f,
        )
    ensemble.save_best_weights("best_model_weights.pth")
    save_checkpoint()


if __name__ == "__main__":
    run_bot()
