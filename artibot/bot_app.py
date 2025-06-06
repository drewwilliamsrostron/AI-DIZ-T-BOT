"""Entry point for running the trading bot.

This module starts the training loop, live market polling, the
meta–reinforcement learning agent and the Tkinter GUI. Before running the
bot you must edit :data:`CONFIG` below with your own API credentials.
"""

from .globals import *

# ---------------------------------------------------------------------------
# Configuration – **fill in your API keys here**
# ---------------------------------------------------------------------------
CONFIG = {
    "CSV_PATH": "Gemini_BTCUSD_1h.csv",  # historical data for initial training
    "symbol": "BTC/USDT",
    "ADAPT_TO_LIVE": False,
    "LIVE_POLL_INTERVAL": 60.0,
    "API": {"API_KEY_LIVE": "", "API_SECRET_LIVE": "", "DEFAULT_TYPE": "spot"},
    "CHATGPT": {"API_KEY": ""},
}


def run_bot():
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

    config = CONFIG
    openai.api_key = config["CHATGPT"]["API_KEY"]
    csv_path = config["CSV_PATH"]
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

if __name__ == "__main__":
    run_bot()
