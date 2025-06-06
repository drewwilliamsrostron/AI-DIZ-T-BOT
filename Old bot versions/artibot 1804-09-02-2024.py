#!/usr/bin/env python3
"""
Complex AI Trading + Continuous Training + Robust Backtest Each Epoch + Live Phemex + Tkinter GUI + Meta–Control
===============================================================================================
DISCLAIMER:
  This is a demonstration. Real trading requires robust risk management and thorough testing.
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
import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

###############################################################################
# Global Hyperparameters for Trade Simulation (defaults adjusted)
###############################################################################
global_SL_multiplier = 0.9        # default stop-loss multiplier (tighter stop)
global_TP_multiplier = 1.1        # default take-profit multiplier (wider target)
global_ATR_period = 14            # ATR period

###############################################################################
# Helper Functions for Meta–Control and JSON Parsing
###############################################################################
def extract_json(text):
    """Extract the first JSON object found in text."""
    try:
        # Using a regex to capture the first { ... } block.
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            return json.loads(json_str)
    except Exception as e:
        logging.error(f"Error in extract_json: {e}")
    return None

def robust_parse(text):
    """Attempt to extract JSON; if unsuccessful, return an empty dict."""
    extracted = extract_json(text)
    return extracted if extracted is not None else {}

def query_chatgpt(prompt, memory, model="gpt-4o", temperature=0.7, max_tokens=150, retries=3, backoff_factor=1.0):
    """Send a prompt to ChatGPT and return the parsed JSON response (or {} on failure)."""
    memory_text = "\n".join(memory[-5:])
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Remember previous adjustments: " + memory_text},
        {"role": "user", "content": prompt}
    ]
    for i in range(retries):
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response = completion.choices[0].message.content.strip()
            memory.append("User: " + prompt)
            memory.append("Assistant: " + response)
            try:
                parsed = json.loads(response)
                return parsed
            except Exception as e:
                logging.error(f"Error parsing JSON from ChatGPT response: {e}")
                return {}
        except Exception as e:
            logging.error(f"ChatGPT API error: {e}")
            time.sleep(backoff_factor * (2 ** i))
    logging.error("Max retries exceeded for ChatGPT API request.")
    return {}

###############################################################################
# Meta–Control Prompt Templates and Controllers
###############################################################################
squirtle_prompt_template = (
    "Squirtle: Based on the following performance metrics (which include training and validation losses, loss gap, net profit history since the last meta–control call, and current trading indicator parameters), "
    "suggest minimal adjustments to the hyperparameters so that both the training and validation losses continue to decrease and the simulated trading performance improves. "
    "Also consider modifying the stop-loss multiplier, take-profit multiplier, and ATR period. "
    "Return a JSON object with keys 'adjust_attention_focus' (a float multiplier), 'adjust_prediction_threshold' (a float), "
    "'adjust_learning_rate' (a float), 'adjust_weight_decay' (a float), 'adjust_stop_loss_multiplier' (a float), "
    "'adjust_take_profit_multiplier' (a float), and 'adjust_ATR_period' (an integer). "
    "Metrics:\n{summary}"
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
    "Ensure these adjustments are safe and sensible. The safe ranges are: attention multiplier between 1.0 and 1.5, "
    "prediction threshold between 0.05 and 0.15, learning rate between 1e-4 and 1e-2, weight decay between 0.0 and 0.01, "
    "stop-loss multiplier >= 0.5, take-profit multiplier >= 0.5, and ATR period between 5 and 30. "
    "If any values are outside these ranges, modify them accordingly. Return the final adjustments as a JSON object."
)

moneymaker_prompt_template = (
    "Moneymaker: Your sole objective is to maximize profit. You have control over the other three AIs and may modify their prompt templates if doing so increases profit. "
    "The current prompt templates are:\nSquirtle: {squirtle_prompt}\nWartortle: {wartortle_prompt}\nBigManBlastoise: {bigmanblastoise_prompt}\n"
    "Based on the performance metrics:\n{summary}\n"
    "and the current responses:\nSquirtle: {squirtle_response}\nWartortle: {wartortle_response}\nBigManBlastoise: {bigmanblastoise_response}\n"
    "Return a JSON object with keys 'squirtle_prompt_mod', 'wartortle_prompt_mod', and 'bigmanblastoise_prompt_mod'. If no modifications are needed, return an empty JSON object."
)

def squirtle_controller(ensemble, summary):
    prompt = squirtle_prompt_template.format(summary=summary)
    response = query_chatgpt(prompt, memory=gpt_memory_squirtle)
    parsed_response = robust_parse(json.dumps(response))
    return {"prompt": prompt, "response": parsed_response}

def wartortle_controller(ensemble, summary, squirtle_response):
    prompt = wartortle_prompt_template.format(summary=summary, squirtle_response=squirtle_response)
    response = query_chatgpt(prompt, memory=gpt_memory_wartorttle)
    parsed_response = robust_parse(json.dumps(response))
    return {"prompt": prompt, "response": parsed_response}

def bigmanblastoise_overseer(ensemble, wartortle_response):
    prompt = bigmanblastoise_prompt_template.format(wartortle_response=wartortle_response)
    response = query_chatgpt(prompt, memory=gpt_memory_bigmanblastoise)
    parsed_response = robust_parse(json.dumps(response))
    return {"prompt": prompt, "response": parsed_response}

def moneymaker_controller(ensemble, summary, squirtle_res, wartortle_res, bigmanblastoise_res):
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
    parsed_response = robust_parse(json.dumps(response))
    return {"prompt": prompt, "response": parsed_response}

def apply_adjustments(adjustments):
    global global_SL_multiplier, global_TP_multiplier, global_ATR_period
    if not isinstance(adjustments, dict):
        logging.info(f"Adjustments not in JSON format; full response: {adjustments}")
        return
    if "adjust_stop_loss_multiplier" in adjustments:
        global_SL_multiplier = float(adjustments["adjust_stop_loss_multiplier"])
        logging.info(f"Stop Loss multiplier adjusted to: {global_SL_multiplier}")
    if "adjust_take_profit_multiplier" in adjustments:
        global_TP_multiplier = float(adjustments["adjust_take_profit_multiplier"])
        logging.info(f"Take Profit multiplier adjusted to: {global_TP_multiplier}")
    if "adjust_ATR_period" in adjustments:
        global_ATR_period = int(adjustments["adjust_ATR_period"])
        logging.info(f"ATR period adjusted to: {global_ATR_period}")
    # (Other hyperparameters such as attention focus, learning rate, etc. may be applied to the model/optimizer as needed.)

###############################################################################
# Global GPT Memories
###############################################################################
gpt_memory_squirtle = []
gpt_memory_wartorttle = []
gpt_memory_bigmanblastoise = []
gpt_memory_moneymaker = []

###############################################################################
# Robust Backtest Function (with unchanged core simulation logic)
###############################################################################
def robust_backtest(ensemble, data):
    if len(data) < 24:
        return {"net_pct": 0.0, "trades": 0, "effective_net_pct": 0.0, "equity_curve": []}

    device = next(ensemble.models[0].parameters()).device
    windows = np.lib.stride_tricks.sliding_window_view(np.array(data)[:, 1:6], (24, 5)).squeeze()
    windows_tensor = torch.tensor(windows, dtype=torch.float32, device=device)
    pred_indices, _ = ensemble.vectorized_predict(windows_tensor)
    preds = [2] * 23 + pred_indices.tolist()

    df = pd.DataFrame({
        'timestamp': [row[0] for row in data],
        'open': [row[1] * 100000 for row in data],
        'high': [row[2] * 100000 for row in data],
        'low': [row[3] * 100000 for row in data],
        'close': [row[4] * 100000 for row in data],
        'prediction': preds
    })

    df['previous_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['previous_close']),
                                     abs(df['low'] - df['previous_close'])))
    df['ATR'] = df['tr'].rolling(global_ATR_period, min_periods=1).mean()

    initial_balance = 10000.0
    balance = initial_balance
    commission_rate = 0.0012
    slippage = 0.0002
    risk_fraction = 0.20
    equity_curve = []
    trades = []

    position = {'size': 0.0, 'side': None, 'entry_price': 0.0, 'stop_loss': 0.0,
                'take_profit': 0.0, 'entry_time': None}

    for i, row in df.iterrows():
        current_price = row['close']
        current_time = row['timestamp']
        prediction = row['prediction']

        if position['size'] != 0:
            exit_condition = False
            exit_reason = ""
            exit_price = current_price

            if position['side'] == 'long':
                if row['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = "SL"
                    exit_condition = True
                elif row['high'] >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = "TP"
                    exit_condition = True
                elif prediction == 1:
                    exit_reason = "Signal"
                    exit_condition = True
            elif position['side'] == 'short':
                if row['high'] >= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = "SL"
                    exit_condition = True
                elif row['low'] <= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = "TP"
                    exit_condition = True
                elif prediction == 0:
                    exit_reason = "Signal"
                    exit_condition = True

            if exit_condition:
                if position['side'] == 'long':
                    exit_price *= (1 - slippage)
                    proceeds = position['size'] * exit_price
                    commission = proceeds * commission_rate
                    new_balance = balance + (proceeds - commission)
                else:
                    exit_price *= (1 + slippage)
                    cost = abs(position['size']) * exit_price
                    commission = cost * commission_rate
                    new_balance = balance + abs(position['size']) * (position['entry_price'] - exit_price) - commission

                if not np.isfinite(new_balance):
                    logging.error("Non-finite new_balance encountered during exit; trade skipped.")
                else:
                    balance = new_balance
                    trades.append({
                        'entry': position['entry_time'],
                        'exit': current_time,
                        'side': position['side'],
                        'return': (exit_price / position['entry_price'] - 1) * (-1 if position['side'] == 'short' else 1),
                        'duration': current_time - position['entry_time'],
                        'exit_reason': exit_reason
                    })
                position = {k: 0.0 if not isinstance(v, (type(None), str)) else None for k, v in position.items()}

        if position['size'] == 0 and prediction in (0, 1):
            atr = row['ATR'] if not np.isnan(row['ATR']) else 1.0
            MIN_ATR = 1.0
            atr = max(atr, MIN_ATR)
            fill_price = current_price * ((1 + slippage) if prediction == 0 else (1 - slippage))
            stop_distance = global_SL_multiplier * atr
            risk_capital = balance * risk_fraction
            epsilon = 1e-8
            position_size = risk_capital / (stop_distance + epsilon)
            max_size = balance / fill_price
            position_size = min(position_size, max_size)
            if position_size <= 0:
                continue

            if prediction == 0:  # Long
                position.update({
                    'size': position_size,
                    'side': 'long',
                    'entry_price': fill_price,
                    'stop_loss': fill_price - stop_distance,
                    'take_profit': fill_price + global_TP_multiplier * atr,
                    'entry_time': current_time
                })
            else:  # Short
                position.update({
                    'size': -position_size,
                    'side': 'short',
                    'entry_price': fill_price,
                    'stop_loss': fill_price + stop_distance,
                    'take_profit': fill_price - global_TP_multiplier * atr,
                    'entry_time': current_time
                })

        current_equity = balance
        if position['size'] != 0:
            if position['side'] == 'long':
                current_equity += position['size'] * (current_price - position['entry_price'])
            else:
                current_equity += abs(position['size']) * (position['entry_price'] - current_price)
        equity_curve.append((current_time, current_equity))

    if position['size'] != 0:
        final_price = df.iloc[-1]['close']
        if position['side'] == 'long':
            commission_exit = position['size'] * final_price * commission_rate
            profit = position['size'] * (final_price - position['entry_price']) - commission_exit
            balance += profit
        else:
            commission_exit = abs(position['size']) * final_price * commission_rate
            profit = abs(position['size']) * (position['entry_price'] - final_price) - commission_exit
            balance += profit
        trades.append({
            'entry': position['entry_time'],
            'exit': df.iloc[-1]['timestamp'],
            'side': position['side'],
            'return': (final_price / position['entry_price'] - 1) * (-1 if position['side'] == 'short' else 1),
            'duration': df.iloc[-1]['timestamp'] - position['entry_time'],
            'exit_reason': "Final"
        })
        equity_curve[-1] = (df.iloc[-1]['timestamp'], balance)

    final_profit = balance - initial_balance
    net_pct = (final_profit / initial_balance) * 100.0
    bonus_factor = 0.5
    effective_net_pct = net_pct if net_pct <= 0 else net_pct + bonus_factor * len(trades)

    returns = pd.Series([e[1] for e in equity_curve],
                        index=pd.to_datetime([e[0] for e in equity_curve], unit='ms'))
    sharpe_ratio = returns.pct_change(fill_method=None).mean() / returns.pct_change(fill_method=None).std() * np.sqrt(365*24)
    max_drawdown = (returns.expanding().max() - returns).max()

    return {
        "net_pct": net_pct,
        "trades": len(trades),
        "effective_net_pct": effective_net_pct,
        "sharpe": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "equity_curve": equity_curve,
        "trade_details": trades
    }

###############################################################################
# Ensemble Model with Vectorized Prediction and TorchScript Optimization
###############################################################################
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
        # Apply an optional attention focus multiplier if set as an attribute.
        if hasattr(self, 'attention_focus_multiplier'):
            time_weights = torch.linspace(1.0, self.attention_focus_multiplier, out.size(1), device=dev)
            raw_attn = raw_attn * time_weights.unsqueeze(0).unsqueeze(-1)
        w = torch.softmax(raw_attn, dim=1)
        context = torch.sum(w * out, dim=1)
        context = self.dropout(context)
        logits = self.fc(context)
        return logits, w

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

    def save_best_weights(self, path="best_model_weights.pth"):
        if self.best_state_dicts is None:
            logging.warning("No best_state_dicts to save!")
            return
        torch.save({
            'best_val_loss': self.best_val_loss,
            'state_dicts': self.best_state_dicts
        }, path)
        logging.info(f"Best model weights saved to {path}")

    def load_best_weights(self, path="best_model_weights.pth"):
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
# CSV Loader (Fixed to skip first row)
###############################################################################
def load_csv_hourly(csv_path="Gemini_BTCUSD_1h.csv"):
    if not os.path.isfile(csv_path):
        logging.error(f"CSV file '{csv_path}' not found.")
        return []
    df = pd.read_csv(csv_path, sep=None, engine='python', skiprows=1)
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
        "Volume USD": "volume_usd"
    }, inplace=True)
    if 'volume_btc' not in df.columns and 'volume_usd' in df.columns:
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
# Dataset Definition
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
            lbl = 0 if ret > self.threshold else (1 if ret < -self.threshold else 2)
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

###############################################################################
# Tkinter GUI Definition
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
                logging.error(f"Error plotting equity curve: {e}")
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
# Background Threads for Training and Live Data
###############################################################################
def csv_training_thread(ensemble, data, stop_event, config):
    try:
        ds_full = HourlyDataset(data, seq_len=24, threshold=GLOBAL_THRESHOLD)
        if len(ds_full) < 10:
            logging.warning("Not enough data in CSV. Exiting training thread.")
            return
        ensemble.load_best_weights("best_model_weights.pth")
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
                # (You may want to adjust how confidence is scaled.)
                global_ai_confidence_scale = "N/A"
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
                ensemble.save_best_weights("best_model_weights.pth")
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
# Phemex Connector
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
            logging.error(f"Error initializing exchange: {e}")
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
                logging.warning("No bars returned from Phemex.")
            return bars
        except Exception as e:
            logging.error(f"Error fetching bars: {e}")
            return []

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
# Checkpoint and Training History Saving
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
        "gpt_memory_wartorttle": gpt_memory_wartorttle,
        "gpt_memory_bigmanblastoise": gpt_memory_bigmanblastoise,
        "gpt_memory_moneymaker": gpt_memory_moneymaker,
        "global_attention_weights_history": global_attention_weights_history
    }
    with open("checkpoint.json", "w") as f:
        json.dump(checkpoint, f, indent=2)
    logging.info("Checkpoint saved to 'checkpoint.json'.")

###############################################################################
# Main Function
###############################################################################
def main():
    global client, global_training_loss, global_validation_loss, global_backtest_profit
    global global_equity_curve, global_ai_adjustments_log, global_current_prediction, global_ai_confidence, global_ai_confidence_scale, global_ai_epoch_count, global_attention_weights, global_attention_weights_history, GLOBAL_THRESHOLD

    # Initialize global variables for training history.
    global_training_loss = []
    global_validation_loss = []
    global_backtest_profit = []
    global_equity_curve = []
    global_ai_adjustments_log = "No adjustments yet"
    global_current_prediction = None
    global_ai_confidence = None
    global_ai_confidence_scale = None
    global_ai_epoch_count = 0
    global_attention_weights = []
    global_attention_weights_history = []

    # Load configuration.
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

    openai.api_key = config["CHATGPT"]["API_KEY"]

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
    
    dataset = HourlyDataset(data, seq_len=24, threshold=0.001)
    meta_thread = threading.Thread(target=lambda: meta_control_loop(ensemble, dataset), daemon=True)
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
    
    ensemble.save_best_weights("best_model_weights.pth")
    save_checkpoint()
    logging.info("All done. The AI keeps running until you kill the script.")

def meta_control_loop(ensemble, dataset, interval=600):
    global global_ai_adjustments, global_ai_adjustments_log, global_last_meta_epoch, global_ai_epoch_count
    global_last_meta_epoch = 0
    while True:
        if global_ai_epoch_count == 0:
            time.sleep(interval)
            continue

        summary = (
            f"Current Prediction: {global_current_prediction}\n"
            f"Confidence: {global_ai_confidence}\n"
            f"Epoch Count: {global_ai_epoch_count}\n"
            f"Training Loss History (from epoch {global_last_meta_epoch}): {global_training_loss[global_last_meta_epoch:]}\n"
            f"Latest Validation Loss: {global_validation_loss[-1] if global_validation_loss else 'N/A'}\n"
            f"Net Profit History (from epoch {global_last_meta_epoch}): {global_backtest_profit[global_last_meta_epoch:]}\n"
            f"Attention Weights (latest): {global_attention_weights}\n"
            f"Learning Rate: 0.001\n"
            f"Dataset Size: {len(dataset.samples)}\n"
            f"Stop Loss Multiplier: {global_SL_multiplier}\n"
            f"Take Profit Multiplier: {global_TP_multiplier}\n"
            f"ATR Period: {global_ATR_period}\n"
        )
        squirtle_res = squirtle_controller(ensemble, summary)
        wartortle_res = wartortle_controller(ensemble, summary, squirtle_res["response"])
        bigmanblastoise_res = bigmanblastoise_overseer(ensemble, wartortle_res["response"])
        moneymaker_res = moneymaker_controller(ensemble, summary, squirtle_res, wartortle_res, bigmanblastoise_res)
        # For example, if Moneymaker returns modifications, update our hyperparameters:
        if isinstance(moneymaker_res["response"], dict) and moneymaker_res["response"]:
            mods = moneymaker_res["response"]
            logging.info(f"Moneymaker modifications: {mods}")
            apply_adjustments(mods)
        latest_info = (
            "Squirtle:\n" + f"Prompt: {squirtle_res['prompt']}\nResponse: {squirtle_res['response']}\n\n" +
            "Wartortle:\n" + f"Prompt: {wartortle_res['prompt']}\nResponse: {wartortle_res['response']}\n\n" +
            "BigManBlastoise:\n" + f"Prompt: {bigmanblastoise_res['prompt']}\nResponse: {bigmanblastoise_res['response']}\n\n" +
            "Moneymaker:\n" + f"Prompt: {moneymaker_res['prompt']}\nResponse: {moneymaker_res['response']}\n"
        )
        global_ai_adjustments = latest_info
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        global_ai_adjustments_log += f"\n[{timestamp}] {latest_info}\n"
        global_last_meta_epoch = global_ai_epoch_count
        time.sleep(interval)

###############################################################################
# Global Variables for Dashboard (initialized here)
###############################################################################
global_training_loss = []
global_validation_loss = []
global_backtest_profit = []
global_equity_curve = []
global_ai_adjustments = ""
global_ai_adjustments_log = ""
global_current_prediction = None
global_ai_confidence = None
global_ai_confidence_scale = None
global_ai_epoch_count = 0
global_attention_weights = []
global_attention_weights_history = []
GLOBAL_THRESHOLD = 0.001

global_phemex_data = []
live_bars_queue = queue.Queue()

###############################################################################
# Run Main
###############################################################################
if __name__ == "__main__":
    main()
