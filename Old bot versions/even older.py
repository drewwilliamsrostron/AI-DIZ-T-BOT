#!/usr/bin/env python3
"""
Complex AI Trading + Continuous Training + Robust Backtest Each Epoch + Live Phemex + Tkinter GUI + Meta–Control
===============================================================================================
Highlights:
 1) Robust backtest (with partial fills, commission, slippage, and ATR–based stop–loss/take–profit) after each epoch.
 2) Simulated live trading (via backtesting) and continuous training.
 3) Tkinter GUI displaying live AI details plus two panels:
    - Latest AI Adjustments (showing meta–control prompts/responses)
    - AI Adjustments Log (complete history)
    Also, the Backtest Results tab shows the net profit history.
    And on the Training tab, there is an additional separate chart for the equity curve.
 4) Meta–control layer that uses ChatGPT for dynamic hyperparameter adjustment.
    The state summary now includes all metric points since the last meta–control call and reports whether the metrics have improved or worsened.
 5) A fourth AI (“Moneymaker”) whose sole goal is to maximize profit. It can modify the prompt templates of the other three AIs.
 6) The first interaction (when no training epochs have occurred) is skipped.
 7) Checkpoint saving and loading now include the full ChatGPT conversation history and graph data.
DISCLAIMER:
  This is a demonstration. Real trading requires robust risk management and deeper testing.
"""

###############################################################################
# Installer Block
###############################################################################
import subprocess, sys

def install_dependencies():
    dependencies = ["openai", "ccxt", "pandas", "numpy", "torch", "matplotlib"]
    # Tkinter is included in the standard library.
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            print(f"Dependency '{dep}' not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

install_dependencies()

###############################################################################
# Imports
###############################################################################
import os, random, json, time, datetime, threading, queue, logging, re
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
from torch.utils.data import Dataset, DataLoader, random_split
from torch import amp
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

###############################################################################
# Function: confidence_to_scale
###############################################################################
def confidence_to_scale(conf):
    if conf < 0.1:
        return "Extremely Unlikely"
    elif conf < 0.3:
        return "Unlikely"
    elif conf < 0.5:
        return "Possible"
    elif conf < 0.7:
        return "Likely"
    elif conf < 0.9:
        return "Very Likely"
    else:
        return "Extremely Likely"

###############################################################################
# Helper Function: generate_candidates
###############################################################################
def generate_candidates(symbol):
    parts = re.split(r'[/:]', symbol)
    parts = [p for p in parts if p]
    candidates = set()
    if len(parts) == 2:
        base, quote = parts
        candidates.update({
            f"{base}/{quote}",
            f"{base}{quote}",
            f"{base}:{quote}",
            f"{base}/USDT",
            f"{base}USDT"
        })
    else:
        candidates.add(symbol)
    return list(candidates)

###############################################################################
# Helper Function: extract_json
###############################################################################
def extract_json(text):
    """Extracts the first JSON object found in text."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group()
        try:
            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Error parsing extracted JSON: {e}")
    else:
        logging.error("No JSON object found in the text.")
    return None

###############################################################################
# Globals
###############################################################################
global_training_loss = []
global_validation_loss = []
global_current_prediction = None
global_ai_confidence = None
global_ai_confidence_scale = None
global_ai_epoch_count = 0
global_ai_adjustments = "No adjustments yet"      # Latest meta–control info
global_ai_adjustments_log = "No adjustments yet"    # Complete meta–control log

global_phemex_data = []
live_bars_queue = queue.Queue()

global_backtest_profit = []  # Net profit history
global_equity_curve = []     # Equity curve: list of (timestamp, balance)
global_attention_weights = []  # Most recent attention weights (for last prediction)
global_attention_weights_history = []  # History of average attention weights

BEST_MODEL_PATH = "best_model_weights.pth"
GLOBAL_THRESHOLD = 0.001  # For dataset labeling

# Additional trade-management hyperparameters
global_SL_multiplier = 1.0
global_TP_multiplier = 1.0
global_ATR_period = 14

# Global OpenAI client (to be set later)
client = None

# Separate GPT conversation memories for each persona.
gpt_memory_squirtle = []
gpt_memory_wartortle = []
gpt_memory_bigmanblastoise = []
gpt_memory_moneymaker = []  # For the fourth AI

# Global marker for the last meta–control call (by epoch)
global_last_meta_epoch = 0

###############################################################################
# Global Prompt Templates (for modularity)
###############################################################################
squirtle_prompt_template = (
    "Squirtle: Based on the following performance metrics (which include training and validation losses, loss gap, "
    "net profit history since the last meta–control call, and current trading indicator parameters), "
    "suggest minimal adjustments to the hyperparameters so that both the training and validation losses continue to decrease over time and the simulated trading performance improves. "
    "Also consider modifying the stop-loss multiplier, take-profit multiplier, and ATR period. "
    "Return a JSON object with keys 'adjust_attention_focus' (a float multiplier), 'adjust_prediction_threshold' (a float), "
    "'adjust_learning_rate' (a float), 'adjust_weight_decay' (a float), 'adjust_stop_loss_multiplier' (a float), "
    "'adjust_take_profit_multiplier' (a float), and 'adjust_ATR_period' (an integer). "
    "Metrics:\n{summary}\nOnly suggest adjustments if necessary; otherwise, return an empty JSON object."
)

wartortle_prompt_template = (
    "Wartortle: Given the following metrics and the preliminary adjustments from Squirtle: {squirtle_response}, "
    "refine these adjustments if needed and output final adjustments as a JSON object with keys "
    "'adjust_attention_focus', 'adjust_prediction_threshold', 'adjust_learning_rate', 'adjust_weight_decay', "
    "'adjust_stop_loss_multiplier', 'adjust_take_profit_multiplier', and 'adjust_ATR_period'.\n"
    "Metrics:\n{summary}\nReturn only the JSON."
)

bigmanblastoise_prompt_template = (
    "BigManBlastoise: Review the following adjustments provided by Wartortle: {wartortle_response}. "
    "Ensure these adjustments are safe and sensible. The safe ranges are: "
    "attention multiplier between 1.0 and 1.5, prediction threshold between 0.05 and 0.15, "
    "learning rate between 1e-4 and 1e-2, weight decay between 0.0 and 0.01, stop-loss multiplier >= 0.5, "
    "take-profit multiplier >= 0.5, and ATR period between 5 and 30. "
    "If any values are outside these ranges, modify them accordingly. Return the final adjustments as a JSON object."
)

moneymaker_prompt_template = (
    "Moneymaker: Your sole objective is to maximize profit. You have control over the other three AIs and may modify their prompt templates if doing so would increase profit. "
    "The current prompt templates are:\n"
    "Squirtle: {squirtle_prompt}\n"
    "Wartortle: {wartortle_prompt}\n"
    "BigManBlastoise: {bigmanblastoise_prompt}\n"
    "Based on the performance metrics:\n{summary}\n"
    "and the current responses:\n"
    "Squirtle: {squirtle_response}\nWartortle: {wartortle_response}\nBigManBlastoise: {bigmanblastoise_response}\n"
    "Return a JSON object with keys 'squirtle_prompt_mod', 'wartortle_prompt_mod', and 'bigmanblastoise_prompt_mod'. "
    "If no modifications are needed, return an empty JSON object."
)

###############################################################################
# Meta–Control / ChatGPT Helpers
###############################################################################
def query_chatgpt(prompt, memory, model="gpt-4o", temperature=0.7, max_tokens=150, retries=3, backoff_factor=1.0):
    memory_text = "\n".join(memory[-5:])
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Remember previous adjustments: " + memory_text},
        {"role": "user", "content": prompt}
    ]
    for i in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response = completion.choices[0].message.content.strip()
            memory.append("User: " + prompt)
            memory.append("Assistant: " + response)
            return response
        except Exception as e:
            logging.error(f"ChatGPT API error: {e}")
            time.sleep(backoff_factor * (2 ** i))
    logging.error("Max retries exceeded for ChatGPT API request.")
    return "{}"

def robust_parse(text):
    extracted = extract_json(text)
    return extracted if extracted is not None else text

def build_ai_state_summary(ensemble, dataset, start_epoch=0):
    try:
        current_lr = ensemble.optimizers[0].param_groups[0]['lr']
    except Exception as e:
        current_lr = "N/A"
        logging.error(f"Error extracting learning rate: {e}")
    if len(global_training_loss) > start_epoch:
        loss_history = global_training_loss[start_epoch:]
        initial_train_loss = global_training_loss[start_epoch]
        current_train_loss = global_training_loss[-1]
        diff_train = current_train_loss - initial_train_loss
    else:
        loss_history = "N/A"
        diff_train = "N/A"
    if len(global_validation_loss) > start_epoch and global_validation_loss[start_epoch] is not None:
        initial_val_loss = global_validation_loss[start_epoch]
        current_val_loss = global_validation_loss[-1]
        diff_val = current_val_loss - initial_val_loss
    else:
        current_val_loss = "N/A"
        diff_val = "N/A"
    if len(global_backtest_profit) > start_epoch:
        initial_net_profit = global_backtest_profit[start_epoch]
        current_net_profit = global_backtest_profit[-1]
        diff_net_profit = current_net_profit - initial_net_profit
    else:
        diff_net_profit = "N/A"
    try:
        dataset_size = len(dataset.samples)
    except Exception as e:
        dataset_size = "N/A"
        logging.error(f"Error extracting dataset size: {e}")
    summary = (
        f"Current Prediction: {global_current_prediction}\n"
        f"Confidence: {global_ai_confidence}\n"
        f"Epoch Count: {global_ai_epoch_count}\n"
        f"Training Loss History (from epoch {start_epoch}): {loss_history}\n"
        f"Latest Validation Loss: {current_val_loss}\n"
        f"Training Loss Change: {diff_train}\n"
        f"Validation Loss Change: {diff_val}\n"
        f"Net Profit History (from epoch {start_epoch}): {global_backtest_profit[start_epoch:] if len(global_backtest_profit) > start_epoch else 'N/A'}\n"
        f"Net Profit Change: {diff_net_profit}\n"
        f"Attention Weights (latest): {global_attention_weights}\n"
        f"Learning Rate: {current_lr}\n"
        f"Dataset Size: {dataset_size}\n"
        f"Stop Loss Multiplier: {global_SL_multiplier}\n"
        f"Take Profit Multiplier: {global_TP_multiplier}\n"
        f"ATR Period: {global_ATR_period}\n"
    )
    return summary

# --- Controllers using global prompt templates.
def squirtle_controller(ensemble, summary):
    global squirtle_prompt_template
    prompt = squirtle_prompt_template.format(summary=summary)
    response = query_chatgpt(prompt, memory=gpt_memory_squirtle)
    parsed_response = robust_parse(response)
    return {"prompt": prompt, "response": parsed_response}

def wartortle_controller(ensemble, summary, squirtle_response):
    global wartortle_prompt_template
    prompt = wartortle_prompt_template.format(summary=summary, squirtle_response=squirtle_response)
    response = query_chatgpt(prompt, memory=gpt_memory_wartortle)
    parsed_response = robust_parse(response)
    return {"prompt": prompt, "response": parsed_response}

def bigmanblastoise_overseer(ensemble, wartortle_response):
    global bigmanblastoise_prompt_template
    prompt = bigmanblastoise_prompt_template.format(wartortle_response=wartortle_response)
    response = query_chatgpt(prompt, memory=gpt_memory_bigmanblastoise)
    parsed_response = robust_parse(response)
    return {"prompt": prompt, "response": parsed_response}

# --- New: Fourth AI controller (Moneymaker)
def moneymaker_controller(ensemble, summary, squirtle_res, wartortle_res, bigmanblastoise_res):
    global moneymaker_prompt_template, squirtle_prompt_template, wartortle_prompt_template, bigmanblastoise_prompt_template
    prompt = moneymaker_prompt_template.format(
        squirtle_prompt=squirtle_prompt_template,
        wartortle_prompt=wartortle_prompt_template,
        bigmanblastoise_prompt=bigmanblastoise_prompt_template,
        summary=summary,
        squirtle_response=squirtle_res["response"],
        wartortle_response=wartortle_res["response"],
        bigmanblastoise_response=bigmanblastoise_res["response"]
    )
    response = query_chatgpt(prompt, memory=gpt_memory_moneymaker)
    parsed_response = robust_parse(response)
    return {"prompt": prompt, "response": parsed_response}

def adjust_attention_focus(ensemble, new_multiplier):
    for model in ensemble.models:
        model.attention_focus_multiplier = new_multiplier
    logging.info(f"Attention focus multiplier adjusted to: {new_multiplier}")

def adjust_prediction_threshold(new_threshold):
    global GLOBAL_THRESHOLD
    GLOBAL_THRESHOLD = new_threshold
    logging.info(f"Prediction threshold adjusted to: {new_threshold}")

def adjust_learning_rate(ensemble, new_lr):
    for optimizer in ensemble.optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    logging.info(f"Learning rate adjusted to: {new_lr}")

def adjust_weight_decay(ensemble, new_wd):
    for optimizer in ensemble.optimizers:
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = new_wd
    logging.info(f"Weight decay adjusted to: {new_wd}")

def adjust_stop_loss_multiplier(new_sl):
    global global_SL_multiplier
    global_SL_multiplier = new_sl
    logging.info(f"Stop Loss multiplier adjusted to: {new_sl}")

def adjust_take_profit_multiplier(new_tp):
    global global_TP_multiplier
    global_TP_multiplier = new_tp
    logging.info(f"Take Profit multiplier adjusted to: {new_tp}")

def adjust_ATR_period(new_atr):
    global global_ATR_period
    global_ATR_period = int(new_atr)
    logging.info(f"ATR period adjusted to: {new_atr}")

def apply_adjustments(ensemble, adjustments):
    if not isinstance(adjustments, dict):
        logging.info(f"Adjustments not in JSON format; full response: {adjustments}")
        return
    if "adjust_attention_focus" in adjustments:
        new_multiplier = adjustments["adjust_attention_focus"]
        adjust_attention_focus(ensemble, new_multiplier)
    if "adjust_prediction_threshold" in adjustments:
        new_threshold = adjustments["adjust_prediction_threshold"]
        adjust_prediction_threshold(new_threshold)
    if "adjust_learning_rate" in adjustments:
        new_lr = adjustments["adjust_learning_rate"]
        adjust_learning_rate(ensemble, new_lr)
    if "adjust_weight_decay" in adjustments:
        new_wd = adjustments["adjust_weight_decay"]
        adjust_weight_decay(ensemble, new_wd)
    if "adjust_stop_loss_multiplier" in adjustments:
        new_sl = adjustments["adjust_stop_loss_multiplier"]
        adjust_stop_loss_multiplier(new_sl)
    if "adjust_take_profit_multiplier" in adjustments:
        new_tp = adjustments["adjust_take_profit_multiplier"]
        adjust_take_profit_multiplier(new_tp)
    if "adjust_ATR_period" in adjustments:
        new_atr = adjustments["adjust_ATR_period"]
        adjust_ATR_period(new_atr)

###############################################################################
# Meta–Control Loop (with fourth AI and skipping first interaction)
###############################################################################
def meta_control_loop(ensemble, dataset, interval=600):
    global global_ai_adjustments, global_ai_adjustments_log, global_last_meta_epoch, global_ai_epoch_count
    while True:
        # Skip the meta–control interaction on the very first run.
        if global_ai_epoch_count == 0:
            time.sleep(interval)
            continue

        summary = build_ai_state_summary(ensemble, dataset, start_epoch=global_last_meta_epoch)
        squirtle_res = squirtle_controller(ensemble, summary)
        wartortle_res = wartortle_controller(ensemble, summary, squirtle_res["response"])
        bigmanblastoise_res = bigmanblastoise_overseer(ensemble, wartortle_res["response"])
        moneymaker_res = moneymaker_controller(ensemble, summary, squirtle_res, wartortle_res, bigmanblastoise_res)
        # If Moneymaker returns modifications, update the global prompt templates.
        if isinstance(moneymaker_res["response"], dict) and moneymaker_res["response"]:
            mods = moneymaker_res["response"]
            if "squirtle_prompt_mod" in mods:
                global squirtle_prompt_template
                squirtle_prompt_template = mods["squirtle_prompt_mod"]
                logging.info("Squirtle prompt template modified by Moneymaker.")
            if "wartortle_prompt_mod" in mods:
                global wartortle_prompt_template
                wartortle_prompt_template = mods["wartortle_prompt_mod"]
                logging.info("Wartortle prompt template modified by Moneymaker.")
            if "bigmanblastoise_prompt_mod" in mods:
                global bigmanblastoise_prompt_template
                bigmanblastoise_prompt_template = mods["bigmanblastoise_prompt_mod"]
                logging.info("BigManBlastoise prompt template modified by Moneymaker.")
        latest_info = (
            "Squirtle:\n"
            f"Prompt: {squirtle_res['prompt']}\n"
            f"Response: {squirtle_res['response']}\n\n"
            "Wartortle:\n"
            f"Prompt: {wartortle_res['prompt']}\n"
            f"Response: {wartortle_res['response']}\n\n"
            "BigManBlastoise:\n"
            f"Prompt: {bigmanblastoise_res['prompt']}\n"
            f"Response: {bigmanblastoise_res['response']}\n\n"
            "Moneymaker:\n"
            f"Prompt: {moneymaker_res['prompt']}\n"
            f"Response: {moneymaker_res['response']}\n"
        )
        global_ai_adjustments = latest_info
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        global_ai_adjustments_log += f"\n[{timestamp}] {latest_info}\n"
        global_last_meta_epoch = global_ai_epoch_count
        time.sleep(interval)

###############################################################################
# 1) Phemex Connector
###############################################################################
class PhemexConnector:
    def __init__(self, config):
        self.symbol = config.get("symbol", "BTC/USDT")
        api_conf = config.get("API", {})
        self.api_key = api_conf.get("API_KEY_LIVE", "")
        self.api_secret = api_conf.get("API_SECRET_LIVE", "")
        default_type = api_conf.get("DEFAULT_TYPE", "spot")
        try:
            self.exchange = ccxt.phemex({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': default_type}
            })
            logging.info("Connected to Phemex via CCXT.")
        except Exception as e:
            logging.error(f"Error init exchange: {e}")
            sys.exit(1)
        self.exchange.load_markets()
        candidates = generate_candidates(self.symbol)
        for c in candidates:
            if c in self.exchange.markets:
                self.symbol = c
                break
        logging.info(f"Using symbol: {self.symbol}")

    def fetch_latest_bars(self, limit=100):
        try:
            bars = self.exchange.fetch_ohlcv(self.symbol, timeframe='1h', limit=limit)
            if not bars:
                logging.warning("No bars returned from Phemex. Possibly API or symbol issue.")
            return bars
        except Exception as e:
            logging.error(f"Error fetching bars from Phemex: {e}")
            return []

###############################################################################
# 2) CSV Loader
###############################################################################
def load_csv_hourly(csv_path="Gemini_BTCUSD_1h.csv"):
    if not os.path.isfile(csv_path):
        logging.error(f"CSV file '{csv_path}' not found.")
        return []
    df = pd.read_csv(csv_path, comment='#', header=0, skiprows=1, low_memory=False)
    logging.info(f"Columns found in CSV: {df.columns.tolist()}")
    df.rename(columns={
        "unix": "unix",
        "date": "date",
        "symbol": "symbol",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "Volume BTC": "volume_btc",
        "Volume USDT": "volume_usdt",
        "Volume USD": "volume_usd"
    }, inplace=True)
    if 'volume_btc' not in df.columns and 'volume_usdt' in df.columns:
        df['volume_btc'] = df['volume_usdt']
    elif 'volume_btc' not in df.columns and 'volume_usd' in df.columns:
        df['volume_btc'] = df['volume_usd']
    data = []
    for row in df.itertuples(index=False):
        ts = getattr(row, "unix", None)
        if ts is None:
            continue
        o = float(getattr(row, "open", 1.0))
        h = float(getattr(row, "high", 1.0))
        l = float(getattr(row, "low", 1.0))
        c = float(getattr(row, "close", 1.0))
        v = float(getattr(row, "volume_btc", 1.0))
        ts = int(ts)
        o /= 100000.0; h /= 100000.0; l /= 100000.0; c /= 100000.0; v /= 10000.0
        data.append([ts, o, h, l, c, v])
    data.sort(key=lambda x: x[0])
    return data

###############################################################################
# 3) Dataset + Model
###############################################################################
class HourlyDataset(Dataset):
    def __init__(self, data, seq_len=24, threshold=0.001):
        self.data = data
        self.seq_len = seq_len
        self.threshold = threshold
        self.samples = []
        self.labels = []
        self.preprocess()

    def preprocess(self):
        for i in range(self.seq_len, len(self.data)-1):
            window = self.data[i-self.seq_len:i]
            seq_np = np.array(window)[:, 1:6].astype(np.float32)
            curr_close = window[-1][4]
            next_close = self.data[i][4]
            ret = (next_close - curr_close) / (curr_close + 1e-8)
            if ret > self.threshold:
                lbl = 0
            elif ret < -self.threshold:
                lbl = 1
            else:
                lbl = 2
            self.samples.append(seq_np)
            self.labels.append(lbl)
        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels  = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx])
        y = torch.tensor(self.labels[idx])
        return x, y

class TradingModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.attn = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        bs = x.size(0)
        dev = x.device
        h0 = torch.zeros(self.num_layers, bs, self.hidden_size, device=dev)
        c0 = torch.zeros(self.num_layers, bs, self.hidden_size, device=dev)
        out, _ = self.lstm(x, (h0, c0))
        raw_attn = self.attn(out)
        if hasattr(self, 'attention_focus_multiplier'):
            time_weights = torch.linspace(1.0, self.attention_focus_multiplier, out.size(1), device=dev)
            raw_attn = raw_attn * time_weights.unsqueeze(0).unsqueeze(-1)
        w = torch.softmax(raw_attn, dim=1)
        context = torch.sum(w * out, dim=1)
        context = self.dropout(context)
        logits = self.fc(context)
        return logits, w

###############################################################################
# 4) Robust Backtest
###############################################################################
def robust_backtest(ensemble, data):
    if len(data) < 24:
        return {"net_pct": 0.0, "trades": 0, "effective_net_pct": 0.0}
    device = next(ensemble.models[0].parameters()).device
    preds = [2] * 24
    n_windows = len(data) - 24
    windows = np.empty((n_windows, 24, 5), dtype=np.float32)
    for i in range(n_windows):
        window = data[i:i+24]
        windows[i] = np.array(window)[:, 1:6]
    windows_tensor = torch.tensor(windows, device=device)
    pred_indices, _ = ensemble.vectorized_predict(windows_tensor)
    preds.extend(pred_indices.tolist())
    tss, ops, hps, lps, cps = [], [], [], [], []
    for row in data:
        tss.append(row[0])
        ops.append(row[1] * 100000.0)
        hps.append(row[2] * 100000.0)
        lps.append(row[3] * 100000.0)
        cps.append(row[4] * 100000.0)
    df = pd.DataFrame({
        'timestamp': tss,
        'open': ops,
        'high': hps,
        'low': lps,
        'close': cps,
        'prediction': preds
    })
    df['previous_close'] = df['close'].shift(1)
    df['tr'] = df.apply(lambda row: max(
        row['high'] - row['low'],
        abs(row['high'] - row['previous_close']) if pd.notna(row['previous_close']) else 0,
        abs(row['low'] - row['previous_close']) if pd.notna(row['previous_close']) else 0
    ), axis=1)
    df['ATR'] = df['tr'].rolling(window=global_ATR_period, min_periods=1).mean()

    initial_balance = 10000.0
    commission_rate = 0.0012
    slippage = 0.0002
    balance = initial_balance
    equity_curve = []
    position_size = 0.0  
    in_position = False
    position_side = None
    risk_fraction = 0.20
    trade_count = 0
    entry_price = None
    stop_loss = None
    take_profit = None
    entry_capital = 0.0  

    for i, row in df.iterrows():
        c = float(row['close'])
        pred = int(row['prediction'])
        if in_position:
            exit_trade = False
            if position_side == 'long':
                if row['low'] <= stop_loss or pred == 1 or row['high'] >= take_profit:
                    exit_trade = True
            elif position_side == 'short':
                if row['high'] >= stop_loss or pred == 0 or row['low'] <= take_profit:
                    exit_trade = True
            if exit_trade:
                if position_side == 'long':
                    exit_price = stop_loss if row['low'] <= stop_loss else (take_profit if row['high'] >= take_profit else c)
                    commission_exit = position_size * exit_price * commission_rate
                    proceeds = position_size * exit_price - commission_exit
                    balance += proceeds
                elif position_side == 'short':
                    exit_price = stop_loss if row['high'] >= stop_loss else (take_profit if row['low'] <= take_profit else c)
                    commission_exit = abs(position_size) * exit_price * commission_rate
                    cost_to_cover = abs(position_size) * exit_price + commission_exit
                    balance += (entry_capital - cost_to_cover)
                in_position = False
                position_size = 0.0
                position_side = None
                entry_capital = 0.0
                trade_count += 1
        if not in_position:
            if pred == 0:
                capital_to_risk = balance * risk_fraction
                fill_price = c * (1.0 + slippage)
                coins = capital_to_risk / fill_price
                commission_entry = coins * fill_price * commission_rate
                if balance >= (capital_to_risk + commission_entry):
                    balance -= (capital_to_risk + commission_entry)
                    position_size = coins
                    entry_capital = capital_to_risk
                    in_position = True
                    position_side = 'long'
                    entry_price = c
                    atr_value = row['ATR'] if not np.isnan(row['ATR']) else 1.0
                    stop_loss = entry_price - (global_SL_multiplier * atr_value)
                    take_profit = entry_price + (global_TP_multiplier * atr_value)
            elif pred == 1:
                capital_to_risk = balance * risk_fraction
                fill_price = c * (1.0 - slippage)
                coins = capital_to_risk / fill_price
                commission_entry = coins * fill_price * commission_rate
                if balance >= (capital_to_risk + commission_entry):
                    balance -= (capital_to_risk + commission_entry)
                    position_size = -coins
                    entry_capital = capital_to_risk
                    in_position = True
                    position_side = 'short'
                    entry_price = c
                    atr_value = row['ATR'] if not np.isnan(row['ATR']) else 1.0
                    stop_loss = entry_price + (global_SL_multiplier * atr_value)
                    take_profit = entry_price - (global_TP_multiplier * atr_value)
        if in_position:
            if position_side == 'long':
                pos_valuation = position_size * c
                equity = balance + pos_valuation
            elif position_side == 'short':
                equity = balance + (entry_capital - (abs(position_size) * c))
        else:
            equity = balance
        equity_curve.append((row['timestamp'], equity))
    if in_position:
        final_price = df.iloc[-1]['close']
        if position_side == 'long':
            commission_exit = position_size * final_price * commission_rate
            proceeds = position_size * final_price - commission_exit
            balance += proceeds
        elif position_side == 'short':
            commission_exit = abs(position_size) * final_price * commission_rate
            cost_to_cover = abs(position_size) * final_price + commission_exit
            balance += (entry_capital - cost_to_cover)
        trade_count += 1
        equity = balance
        equity_curve[-1] = (df.iloc[-1]['timestamp'], equity)
    final_profit = balance - initial_balance
    net_pct = (final_profit / initial_balance) * 100.0
    bonus_factor = 0.5
    effective_net_pct = net_pct if net_pct <= 0 else net_pct + bonus_factor * trade_count
    global global_equity_curve
    global_equity_curve = equity_curve
    return {"net_pct": net_pct, "trades": trade_count, "effective_net_pct": effective_net_pct}

###############################################################################
# 5) Ensemble with Vectorized Prediction and TorchScript Optimization
###############################################################################
class EnsembleModel:
    def __init__(self, device, n_models=2, lr=1e-3, weight_decay=0.0):
        self.device = device
        self.models = [TradingModel(dropout=0.2).to(device) for _ in range(n_models)]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay) for m in self.models]
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = amp.GradScaler(enabled=(device.type=='cuda'))
        self.best_val_loss = float('inf')
        self.best_state_dicts = None
        self.train_steps = 0
        self.optimized_models = None

    def train_one_epoch(self, dl_train, dl_val, data_full):
        for m in self.models:
            m.train()
        losses = []
        for batch_x, batch_y in dl_train:
            bx = batch_x.to(self.device, non_blocking=True)
            by = batch_y.to(self.device, non_blocking=True)
            mini_losses = []
            for model, opt in zip(self.models, self.optimizers):
                opt.zero_grad()
                with amp.autocast(device_type=self.device.type,
                                  dtype=torch.float16 if self.device.type=='cuda' else torch.float32):
                    logits, _ = model(bx)
                    loss = self.criterion(logits, by)
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
                mini_losses.append(loss.item())
            losses.append(np.mean(mini_losses))
        train_loss = float(np.mean(losses))
        val_loss = None
        if dl_val is not None:
            val_loss = self.evaluate_val_loss(dl_val)
        result = robust_backtest(self, data_full)
        net_pct = result["net_pct"]
        trades = result["trades"]
        eff_net_pct = result["effective_net_pct"]
        global_backtest_profit.append(eff_net_pct)
        if val_loss is not None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state_dicts = [m.state_dict() for m in self.models]
                logging.info(f"New best val_loss: {val_loss:.4f} | Net Profit: {eff_net_pct:.2f}% (raw: {net_pct:.2f}%, trades: {trades})")
            logging.info(f"Epoch {self.train_steps+1} => Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Net Profit: {eff_net_pct:.2f}% (raw: {net_pct:.2f}%, trades: {trades})")
        else:
            logging.info(f"Epoch {self.train_steps+1} => Train Loss={train_loss:.4f}, No Val set => Net Profit: {eff_net_pct:.2f}% (raw: {net_pct:.2f}%, trades: {trades})")
        return train_loss, val_loss

    def evaluate_val_loss(self, dl_val):
        for m in self.models:
            m.eval()
        losses = []
        with torch.no_grad():
            for batch_x, batch_y in dl_val:
                bx = batch_x.to(self.device, non_blocking=True)
                by = batch_y.to(self.device, non_blocking=True)
                model_losses = []
                for model in self.models:
                    logits, _ = model(bx)
                    loss = self.criterion(logits, by)
                    model_losses.append(loss.item())
                losses.append(np.mean(model_losses))
        return float(np.mean(losses))

    def predict(self, x):
        for m in self.models:
            m.eval()
        with torch.no_grad():
            outs, weights = [], []
            for model in self.models:
                logits, w = model(x.to(self.device))
                p = torch.softmax(logits, dim=1)
                outs.append(p.cpu().numpy())
                weights.append(w.cpu().numpy())
            avgp = np.mean(outs, axis=0)
            pred_idx = int(np.argmax(avgp[0]))
            conf = float(avgp[0][pred_idx])
            attn = np.mean(np.concatenate(weights, axis=1), axis=1)
        return pred_idx, conf, attn.tolist()

    def vectorized_predict(self, windows_tensor):
        models = self.optimized_models if self.optimized_models is not None else self.models
        for m in models:
            m.eval()
        outs = []
        with torch.no_grad():
            for model in models:
                logits, _ = model(windows_tensor.to(self.device))
                probs = torch.softmax(logits, dim=1)
                outs.append(probs)
        avg_probs = sum(outs) / len(outs)
        pred_indices = avg_probs.argmax(dim=1)
        confs = avg_probs.max(dim=1)[0]
        return pred_indices.cpu(), confs.cpu()

    def optimize_models(self, dummy_input):
        self.optimized_models = []
        for model in self.models:
            model.eval()
            traced_model = torch.jit.trace(model, dummy_input)
            self.optimized_models.append(traced_model)
        logging.info("Models optimized with TorchScript.")

    def save_best_weights(self, path=BEST_MODEL_PATH):
        if self.best_state_dicts is None:
            logging.warning("No best_state_dicts to save!")
            return
        torch.save({
            'best_val_loss': self.best_val_loss,
            'state_dicts': self.best_state_dicts
        }, path)
        logging.info(f"Best model weights saved to {path}")

    def load_best_weights(self, path=BEST_MODEL_PATH):
        if os.path.isfile(path):
            ckpt = torch.load(path, map_location=self.device)
            self.best_val_loss = ckpt['best_val_loss']
            self.best_state_dicts = ckpt['state_dicts']
            for m, sd in zip(self.models, self.best_state_dicts):
                m.load_state_dict(sd)
            logging.info(f"Loaded best weights from {path} (val_loss={self.best_val_loss:.4f})")
        else:
            logging.warning(f"No checkpoint found at {path}")

###############################################################################
# 6) Background Threads
###############################################################################
def csv_training_thread(ensemble, data, stop_event, config):
    try:
        ds_full = HourlyDataset(data, seq_len=24, threshold=GLOBAL_THRESHOLD)
        if len(ds_full) < 10:
            logging.warning("Not enough data in CSV. Exiting training thread.")
            return
        ensemble.load_best_weights(BEST_MODEL_PATH)
        n_total = len(ds_full)
        n_train = int(n_total * 0.9)
        n_val = n_total - n_train
        ds_train, ds_val = random_split(ds_full, [n_train, n_val])
        dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
        dl_val = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
        adapt_to_live = bool(config.get("ADAPT_TO_LIVE", False))
        dummy_input = torch.randn(1, 24, 5, device=ensemble.device)
        ensemble.optimize_models(dummy_input)
        while not stop_event.is_set():
            ensemble.train_steps += 1
            train_loss, val_loss = ensemble.train_one_epoch(dl_train, dl_val, data)
            global_training_loss.append(train_loss)
            global_validation_loss.append(val_loss if val_loss is not None else None)
            if len(data) >= 24:
                tail24 = data[-24:]
                seq_np = np.array(tail24)[:, 1:6].astype(np.float32)
                seq_t = torch.tensor(seq_np).unsqueeze(0).to(next(ensemble.models[0].parameters()).device)
                pred_idx, conf, atn = ensemble.predict(seq_t)
                label_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
                global global_current_prediction, global_ai_confidence, global_ai_confidence_scale, global_ai_epoch_count, global_attention_weights
                global_current_prediction = label_map.get(pred_idx, "N/A")
                global_ai_confidence = conf
                global_ai_confidence_scale = confidence_to_scale(conf)
                global_ai_epoch_count = ensemble.train_steps
                global_attention_weights = atn
                global_attention_weights_history.append(np.mean(atn))
            if adapt_to_live:
                data_changed = False
                while not live_bars_queue.empty():
                    new_bars = live_bars_queue.get()
                    for bar in new_bars:
                        ts, o, h, l, c, v = bar
                        o /= 100000.0; h /= 100000.0; l /= 100000.0; c /= 100000.0; v /= 10000.0
                        if ts > data[-1][0]:
                            data.append([ts, o, h, l, c, v])
                            data_changed = True
                if data_changed:
                    ds_updated = HourlyDataset(data, seq_len=24, threshold=GLOBAL_THRESHOLD)
                    if len(ds_updated) > 10:
                        nt = len(ds_updated)
                        ntr = int(nt * 0.9)
                        nv = nt - ntr
                        ds_train_upd, ds_val_upd = random_split(ds_updated, [ntr, nv])
                        dl_train_upd = DataLoader(ds_train_upd, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
                        dl_val_upd = DataLoader(ds_val_upd, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
                        ensemble.train_one_epoch(dl_train_upd, dl_val_upd, data)
            if ensemble.train_steps % 5 == 0 and ensemble.best_state_dicts is not None:
                ensemble.save_best_weights(BEST_MODEL_PATH)
    except Exception as e:
        logging.exception(f"Error in training_thread: {e}")
        stop_event.set()

def phemex_live_thread(connector, stop_event):
    try:
        while not stop_event.is_set():
            bars = connector.fetch_latest_bars(limit=100)
            global global_phemex_data
            if bars:
                global_phemex_data = bars
                live_bars_queue.put(bars)
            time.sleep(15)
    except Exception as e:
        logging.exception(f"Error in phemex_live_thread: {e}")
        stop_event.set()

###############################################################################
# 7) Tkinter GUI with AI Adjustments, Log Panels, and Chart Displays
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
        
        self.info_frame = ttk.Frame(root)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        disclaimer_text = ("NOT INVESTMENT ADVICE! This is a demonstration.\nUse caution and do your own research.")
        self.disclaimer_label = ttk.Label(self.info_frame, text=disclaimer_text,
                                          font=("Helvetica", 9, "italic"), foreground="darkred")
        self.disclaimer_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.pred_label = ttk.Label(self.info_frame, text="AI Prediction: N/A", font=("Helvetica", 12))
        self.pred_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.conf_label = ttk.Label(self.info_frame, text="Confidence: N/A", font=("Helvetica", 12))
        self.conf_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.epoch_label = ttk.Label(self.info_frame, text="Training Steps: 0", font=("Helvetica", 12))
        self.epoch_label.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.frame_ai = ttk.Frame(root)
        self.frame_ai.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.ai_output_label = ttk.Label(self.frame_ai, text="Latest AI Adjustments:", font=("Helvetica", 12, "bold"))
        self.ai_output_label.pack(anchor="n")
        self.ai_output_text = tk.Text(self.frame_ai, width=40, height=10, wrap="word")
        self.ai_output_text.pack(fill=tk.BOTH, expand=True)
        self.frame_ai_log = ttk.Frame(root)
        self.frame_ai_log.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.ai_log_label = ttk.Label(self.frame_ai_log, text="AI Adjustments Log:", font=("Helvetica", 12, "bold"))
        self.ai_log_label.pack(anchor="n")
        self.ai_log_text = tk.Text(self.frame_ai_log, width=40, height=10, wrap="word")
        self.ai_log_text.pack(fill=tk.BOTH, expand=True)
        
        self.update_interval = 2000
        self.root.after(self.update_interval, self.update_dashboard)

    def update_dashboard(self):
        self.ax_loss.clear()
        self.ax_loss.set_title("Training vs. Validation Loss")
        x1 = range(1, len(global_training_loss)+1)
        self.ax_loss.plot(x1, global_training_loss, color='blue', marker='o', label='Training Loss')
        val_filtered = [(i+1, v) for i, v in enumerate(global_validation_loss) if v is not None]
        if val_filtered:
            xv, yv = zip(*val_filtered)
            self.ax_loss.plot(xv, yv, color='orange', marker='x', label='Validation Loss')
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.legend()
        
        self.ax_equity_train.clear()
        self.ax_equity_train.set_title("Equity Curve")
        if global_equity_curve:
            try:
                times_eq, balances = zip(*global_equity_curve)
                times_dt = []
                for t in times_eq:
                    if t < 10**10:
                        times_dt.append(datetime.datetime.fromtimestamp(t))
                    else:
                        times_dt.append(datetime.datetime.fromtimestamp(t/1000.0))
                self.ax_equity_train.plot(times_dt, balances, color='red', marker='.')
                self.ax_equity_train.set_xlabel("Time")
                self.ax_equity_train.set_ylabel("Equity ($)")
            except Exception as e:
                logging.error(f"Error plotting equity curve on training page: {e}")
                self.ax_equity_train.text(0.5, 0.5, "Error in Equity Curve", ha='center', va='center')
        else:
            self.ax_equity_train.text(0.5, 0.5, "No Equity Data", ha='center', va='center')
        self.canvas_train.draw()
        
        self.ax_live.clear()
        self.ax_live.set_title("Phemex Live Price (1h Bars)")
        if global_phemex_data:
            times, closes = [], []
            for bar in global_phemex_data:
                t = bar[0]
                c = bar[4]
                times.append(datetime.datetime.fromtimestamp(t/1000.0))
                closes.append(c)
            self.ax_live.plot(times, closes, marker='o')
        else:
            self.ax_live.text(0.5, 0.5, "No Live Data", ha='center', va='center')
        self.canvas_live.draw()
        
        self.ax_net_profit.clear()
        self.ax_net_profit.set_title("Net Profit (%)")
        if global_backtest_profit:
            x2 = range(1, len(global_backtest_profit)+1)
            self.ax_net_profit.plot(x2, global_backtest_profit, marker='o', color='green')
        else:
            self.ax_net_profit.text(0.5, 0.5, "No Backtest Data", ha='center', va='center')
        self.ax_net_profit.set_xlabel("Epoch")
        self.ax_net_profit.set_ylabel("Effective Net Profit (%)")
        self.canvas_backtest.draw()
        
        self.ax_details.clear()
        self.ax_details.set_title("Average Attention Weight History")
        if global_attention_weights_history:
            x_vals = list(range(1, len(global_attention_weights_history)+1))
            self.ax_details.plot(x_vals, global_attention_weights_history, marker='o', color='purple')
            self.ax_details.set_xlabel("Epoch")
            self.ax_details.set_ylabel("Avg Attention Weight")
        else:
            self.ax_details.text(0.5, 0.5, "No Attention Data", ha='center', va='center')
        self.canvas_details.draw()
        
        pred_str = global_current_prediction if global_current_prediction else "N/A"
        conf = global_ai_confidence if global_ai_confidence else 0.0
        scale = global_ai_confidence_scale if global_ai_confidence_scale else "N/A"
        steps = global_ai_epoch_count
        self.pred_label.config(text=f"AI Prediction: {pred_str} ({scale})")
        self.conf_label.config(text=f"Confidence: {conf:.2f}")
        self.epoch_label.config(text=f"Training Steps: {steps}")
        
        self.ai_output_text.delete("1.0", tk.END)
        self.ai_output_text.insert(tk.END, global_ai_adjustments)
        self.ai_log_text.delete("1.0", tk.END)
        self.ai_log_text.insert(tk.END, global_ai_adjustments_log)
        
        self.root.after(self.update_interval, self.update_dashboard)

###############################################################################
# Checkpoint Saving Function
###############################################################################
def save_checkpoint():
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
        "gpt_memory_wartortle": gpt_memory_wartortle,
        "gpt_memory_bigmanblastoise": gpt_memory_bigmanblastoise,
        "gpt_memory_moneymaker": gpt_memory_moneymaker,
        "global_attention_weights_history": global_attention_weights_history
    }
    with open("checkpoint.json", "w") as f:
        json.dump(checkpoint, f, indent=2)
    logging.info("Checkpoint saved to 'checkpoint.json'.")

###############################################################################
# 8) Main
###############################################################################
def main():
    global client, gpt_memory_squirtle, gpt_memory_wartortle, gpt_memory_bigmanblastoise, gpt_memory_moneymaker, global_equity_curve, global_attention_weights_history, global_ai_adjustments_log
    # Load training history if available.
    if os.path.isfile("training_history.json"):
        with open("training_history.json", "r") as f:
            history = json.load(f)
            global_training_loss.extend(history.get("global_training_loss", []))
            global_validation_loss.extend(history.get("global_validation_loss", []))
            global_backtest_profit.extend(history.get("global_backtest_profit", []))
    # Load checkpoint (includes GPT memories and full graph data) if available.
    if os.path.isfile("checkpoint.json"):
        with open("checkpoint.json", "r") as f:
            checkpoint = json.load(f)
        global_equity_curve = checkpoint.get("global_equity_curve", [])
        gpt_memory_squirtle = checkpoint.get("gpt_memory_squirtle", [])
        gpt_memory_wartortle = checkpoint.get("gpt_memory_wartortle", [])
        gpt_memory_bigmanblastoise = checkpoint.get("gpt_memory_bigmanblastoise", [])
        gpt_memory_moneymaker = checkpoint.get("gpt_memory_moneymaker", [])
        global_attention_weights_history = checkpoint.get("global_attention_weights_history", [])
        global_ai_adjustments_log = checkpoint.get("global_ai_adjustments_log", "No adjustments yet")
    if not os.path.isfile("master_config.json"):
        logging.warning("master_config.json not found. Using defaults.")
        config = {
            "CSV_PATH": "Gemini_BTCUSD_1h.csv",
            "symbol": "BTC/USDT",
            "ADAPT_TO_LIVE": True,
            "API": {"API_KEY_LIVE": "", "API_SECRET_LIVE": "", "DEFAULT_TYPE": "spot"},
            "CHATGPT": {"API_KEY": "YOUR_CHATGPT_API_KEY_HERE"}
        }
    else:
        with open("master_config.json", "r") as f:
            config = json.load(f)
    client = OpenAI(api_key=config["CHATGPT"]["API_KEY"])
    
    csv_path = config.get("CSV_PATH", "Gemini_BTCUSD_1h.csv")
    data = load_csv_hourly(csv_path)
    logging.info(f"Loaded {len(data)} bars from CSV.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logging.info("Running on GPU (CUDA).")
    else:
        logging.warning("CUDA not found. Running on CPU.")
    ensemble = EnsembleModel(device=device, n_models=2, lr=1e-3, weight_decay=0.0)
    connector = PhemexConnector(config)
    stop_event = threading.Event()
    train_th = threading.Thread(target=csv_training_thread, args=(ensemble, data, stop_event, config), daemon=True)
    train_th.start()
    phemex_th = threading.Thread(target=phemex_live_thread, args=(connector, stop_event), daemon=True)
    phemex_th.start()
    
    dataset = HourlyDataset(data, seq_len=24, threshold=GLOBAL_THRESHOLD)
    meta_thread = threading.Thread(target=meta_control_loop, args=(ensemble, dataset,), daemon=True)
    meta_thread.start()
    
    root = tk.Tk()
    gui = TradingGUI(root, ensemble)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt => Exiting GUI.")
    stop_event.set()
    train_th.join()
    phemex_th.join()
    
    with open("training_history.json", "w") as f:
        json.dump({
            "global_training_loss": global_training_loss,
            "global_validation_loss": global_validation_loss,
            "global_backtest_profit": global_backtest_profit
        }, f)
    
    ensemble.save_best_weights(BEST_MODEL_PATH)
    save_checkpoint()
    logging.info("All done. The AI keeps running until you kill the script.")

if __name__ == "__main__":
    main()
