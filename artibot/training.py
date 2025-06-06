"""Background CSV training thread and exchange connector."""

from .globals import *
from .dataset import HourlyDataset

###############################################################################
def csv_training_thread(ensemble, data, stop_event, config, use_prev_weights=True):
    from torch.utils.data import random_split, DataLoader
    import traceback
    global global_training_loss, global_validation_loss
    global global_ai_epoch_count
    global global_status_message

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
            global global_status_message
            global_status_message = f"Training step {ensemble.train_steps}"
            print(global_status_message, flush=True)
            tl, vl = ensemble.train_one_epoch(dl_train, dl_val, data, stop_event)
            global_training_loss.append(tl)
            if vl is not None:
                global_validation_loss.append(vl)
            else:
                global_validation_loss.append(None)
            if global_backtest_profit:
                last_pf = global_backtest_profit[-1]
            else:
                last_pf = 0.0
            val_str = f"{vl:.4f}" if vl is not None else "N/A"
            print(
                f"Epoch {ensemble.train_steps}: loss={tl:.4f} val={val_str} "
                f"net_pct={last_pf:.2f}",
                flush=True,
            )

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
                    global_status_message = "Adapting to live data"
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

    except Exception as e:
        traceback.print_exc()
        global_status_message = f"Training error: {e}"
        stop_event.set()

def phemex_live_thread(connector, stop_event, poll_interval=1.0):
    """Continuously fetch recent bars from Phemex at a configurable interval."""
    import traceback
    global global_phemex_data, global_status_message
    while not stop_event.is_set():
        try:
            global_status_message = "Fetching live data"
            bars = connector.fetch_latest_bars(limit=100)
            if bars:
                global_phemex_data = bars
                live_bars_queue.put(bars)
        except Exception as e:
            traceback.print_exc()
            global_status_message = f"Fetch error: {e}"
            stop_event.set()
        status_sleep("Waiting before next fetch", poll_interval)

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
