# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Complex AI Trading with Offline Reinforcement Learning
===============================================================================================
DISCLAIMER:
  This is a demonstration. Real trading requires robust risk management and thorough testing.
"""

###############################################################################
# Installer Block (run once at startup)
###########################################b####################################
import subprocess, sys

def install_dependencies():
    dependencies = {
        "openai": "openai",
        "ccxt": "ccxt",
        "pandas": "pandas",
        "numpy": "numpy",
        "torch": "torch",
        "matplotlib": "matplotlib",
        "scikit-learn": "sklearn"
    }
    for dep_name, mod_name in dependencies.items():
        try:
            __import__(mod_name)
        except ImportError:
            print(f"Dependency '{dep_name}' not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep_name])

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
from typing import NamedTuple
from contextlib import nullcontext
from sklearn.preprocessing import StandardScaler
from torch.distributions import Categorical

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
client = openai  # alias

###############################################################################
# Global Hyperparameters and Variables
###############################################################################
global_SL_multiplier = 0.9        
global_TP_multiplier = 1.1        
global_ATR_period = 14            
risk_fraction = 0.01              
MAX_LEVERAGE = 10                
MAX_RISK_PCT = 0.20              
GLOBAL_THRESHOLD = 0.001          
MAX_DRAWDOWN_PCT = 20             

global_best_params = {
    "SL_multiplier": global_SL_multiplier,
    "TP_multiplier": global_TP_multiplier,
    "ATR_period": global_ATR_period,
    "risk_fraction": risk_fraction,
    "learning_rate": 3e-5
}

global_sharpe = None
global_inactivity_penalty = None
global_composite_reward = None
global_days_without_trading = None
global_trade_details = []  

###############################################################################
# Helper Functions for JSON Parsing
###############################################################################
def extract_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logging.error(f"Error in extract_json: {e}")
    return None

def robust_parse(text):
    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = extract_json(text)
    return parsed if parsed is not None else {}

###############################################################################
# Optimized query_chatgpt Function
###############################################################################
def query_chatgpt(prompt, memory, model="gpt-4o", temperature=0.7, max_tokens=150, retries=3, backoff_factor=0.01):
    system_message = (
        "MAKE SURE TO ADD JSON. YOUR OUTPUT MUST BE A VALID JSON OBJECT AND NOTHING ELSE. "
        "DO NOT INCLUDE ANY COMMENTARY OR ADDITIONAL TEXT."
    )
    messages = [
        {"role": "system", "content": system_message},
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
            if response.startswith("```"):
                response = "\n".join(response.splitlines()[1:])
                response = response.rstrip("`").strip()
            if "{" not in response or "}" not in response:
                logging.error(f"Response did not contain JSON. Response was: {response}")
                time.sleep(backoff_factor)
                continue
            try:
                parsed = json.loads(response)
                return parsed
            except Exception as e:
                logging.error(f"Error parsing JSON: {e}. Response was: {response}")
                time.sleep(backoff_factor)
                continue
        except Exception as e:
            logging.error(f"ChatGPT API error: {e}")
            time.sleep(backoff_factor)
    logging.error("Max retries exceeded for ChatGPT API request.")
    return {}

###############################################################################
# Define a NamedTuple for Trade Parameters
###############################################################################
class TradeParams(NamedTuple):
    risk_fraction: torch.Tensor
    sl_multiplier: torch.Tensor
    tp_multiplier: torch.Tensor
    attention: torch.Tensor

###############################################################################
# Meta–Control Prompt Templates and Controllers
###############################################################################
squirtle_prompt_template = (
    "Squirtle: Based on the following performance metrics (including Sharpe ratio, inactivity penalty, and days without trading), "
    "suggest minimal adjustments to hyperparameters so that training and validation losses decrease and simulated trading performance improves. "
    "Return a JSON object with keys 'adjust_learning_rate', 'adjust_weight_decay', 'adjust_stop_loss_multiplier', 'adjust_take_profit_multiplier', "
    "'adjust_ATR_period', and 'adjust_risk_fraction'.\nMetrics:\n{summary}"
)

wartortle_prompt_template = (
    "Wartortle: Given these metrics and Squirtle's adjustments, refine the adjustments if needed and output a JSON object with final changes."
)

bigmanblastoise_prompt_template = (
    "BigManBlastoise: Review the adjustments from Wartortle and enforce safety ranges: stop-loss multiplier between 0.5 and 2.5, "
    "take-profit multiplier between 1.0 and 3.0, risk fraction between 0.005 and 0.20, ATR period between 5 and 30. Return a JSON object with the final adjustments."
)

moneymaker_prompt_template = (
    "Moneymaker: Your objective is to maximize profit. Based on the metrics and responses from Squirtle, Wartortle, and BigManBlastoise, "
    "return a JSON object with modifications to improve performance. If no changes are needed, return an empty JSON object.\n"
    "Metrics:\n{summary}\nSquirtle: {squirtle_response}\nWartortle: {wartortle_response}\nBigManBlastoise: {bigmanblastoise_response}"
)

def squirtle_controller(ensemble, summary):
    prompt = squirtle_prompt_template.format(summary=summary)
    response = query_chatgpt(prompt, memory=gpt_memory_squirtle)
    return {"prompt": prompt, "response": robust_parse(json.dumps(response))}

def wartortle_controller(ensemble, summary, squirtle_response):
    prompt = wartortle_prompt_template.format(squirtle_response=squirtle_response, summary=summary)
    response = query_chatgpt(prompt, memory=gpt_memory_wartorttle)
    return {"prompt": prompt, "response": robust_parse(json.dumps(response))}

def bigmanblastoise_overseer(ensemble, wartortle_response):
    prompt = bigmanblastoise_prompt_template.format(wartortle_response=wartortle_response)
    response = query_chatgpt(prompt, memory=gpt_memory_bigmanblastoise)
    return {"prompt": prompt, "response": robust_parse(json.dumps(response))}

def moneymaker_controller(ensemble, summary, squirtle_res, wartortle_res, bigmanblastoise_res):
    prompt = moneymaker_prompt_template.format(
        squirtle_response=squirtle_res["response"],
        wartortle_response=wartortle_res["response"],
        bigmanblastoise_response=bigmanblastoise_res["response"],
        summary=summary
    )
    response = query_chatgpt(prompt, memory=gpt_memory_moneymaker)
    return {"prompt": prompt, "response": robust_parse(json.dumps(response))}

def apply_adjustments(adjustments):
    global global_SL_multiplier, global_TP_multiplier, global_ATR_period, risk_fraction
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
    if "adjust_risk_fraction" in adjustments:
        risk_fraction = float(adjustments["adjust_risk_fraction"])
        logging.info(f"Risk fraction adjusted to: {risk_fraction}")

###############################################################################
# Global GPT Memories Initialization
###############################################################################
gpt_memory_squirtle = []
gpt_memory_wartorttle = []
gpt_memory_bigmanblastoise = []
gpt_memory_moneymaker = []

###############################################################################
# CSV Loader (Simple Scaling by dividing prices by 100000)
###############################################################################
def load_csv_hourly(csv_path):
    if not os.path.isfile(csv_path):
        logging.error(f"CSV file '{csv_path}' not found.")
        return []
    try:
        df = pd.read_csv(csv_path, sep=r'[,\t]+', engine='python', skiprows=1, header=0)
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        return []
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    data = []
    for i, row in df.iterrows():
        try:
            ts = int(row['unix'])
            o = float(row['open']) / 100000.0
            h = float(row['high']) / 100000.0
            l = float(row['low']) / 100000.0
            c = float(row['close']) / 100000.0
            v = float(row['volume_btc']) if 'volume_btc' in df.columns else 0.0
        except Exception as e:
            logging.error(f"Error processing row {i}: {e}")
            continue
        data.append([ts, o, h, l, c, v])
    return sorted(data, key=lambda x: x[0])

###############################################################################
# Dataset Definition (Modified for Offline RL)
###############################################################################
class HourlyDataset(Dataset):
    def __init__(self, data, seq_len=24):
        self.data = data
        self.seq_len = seq_len
        self.samples, self.next_closes = self.preprocess()
        
    def preprocess(self):
        samples, next_closes = [], []
        for i in range(self.seq_len, len(self.data) - 1):
            window = self.data[i - self.seq_len:i]
            if np.isnan(window).any() or np.isinf(window).any():
                continue
            next_close = self.data[i][4]  # Next time step's close price
            samples.append(np.array([row[1:6] for row in window], dtype=np.float32))
            next_closes.append(next_close)
        return np.array(samples, dtype=np.float32), np.array(next_closes, dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.next_closes[idx])

###############################################################################
# Trading Model (Modified for Policy Gradient)
###############################################################################
class TradingModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, num_actions=3, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.attn = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_actions + 3)  # 3 actions + trade params
        
        # Initialize weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.attn.weight)
        nn.init.xavier_normal_(self.fc.weight)
        
    def forward(self, x):
        bs = x.size(0)
        dev = x.device
        h0 = torch.zeros(self.num_layers, bs, self.hidden_size, device=dev)
        c0 = torch.zeros(self.num_layers, bs, self.hidden_size, device=dev)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.layernorm(out)
        raw_attn = self.attn(out)
        w = torch.softmax(raw_attn, dim=1)
        context = torch.sum(w * out, dim=1)
        context = self.dropout(context)
        
        out_all = self.fc(context)
        action_logits = out_all[:, :3]
        action_probs = torch.softmax(action_logits, dim=1)
        
        # Trade parameters
        risk_frac = 0.005 + 0.195 * torch.sigmoid(out_all[:, 3])
        sl_mult = 0.1 + 4.9 * torch.sigmoid(out_all[:, 4])
        tp_mult = 0.1 + 4.9 * torch.sigmoid(out_all[:, 5])
        
        return action_probs, TradeParams(risk_frac, sl_mult, tp_mult, w)

###############################################################################
# Ensemble Model (Modified for Offline RL Training)
###############################################################################
class EnsembleModel:
    def __init__(self, device, n_models=2, lr=3e-5, weight_decay=0.0, entropy_coef=0.01):
        self.device = device
        self.models = [TradingModel().to(device) for _ in range(n_models)]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay) 
                             for m in self.models]
        self.entropy_coef = entropy_coef
        self.scaler = amp.GradScaler(enabled=(device.type == 'cuda'))
        self.best_val_loss = float('inf')
        self.best_state_dicts = None
        self.train_steps = 0
        self.optimized_models = None

    def train_one_epoch(self, dl_train, dl_val, data_full):
        for m in self.models:
            m.train()
        losses = []
        
        for batch_x, batch_next_close in dl_train:
            bx = batch_x.to(self.device, non_blocking=True)
            b_next_close = batch_next_close.to(self.device, non_blocking=True)
            
            mini_losses = []
            for model, opt in zip(self.models, self.optimizers):
                opt.zero_grad()
                
                with amp.autocast(device_type=self.device.type, enabled=(self.device.type=='cuda')):
                    # Get current close prices (index 3 in each [open, high, low, close, volume] row)
                    current_closes = bx[:, -1, 3]
                    price_changes = (b_next_close - current_closes) / (current_closes + 1e-8)
                    
                    # Get action probabilities and sample actions
                    action_probs, trade_params = model(bx)
                    m_dist = Categorical(action_probs)
                    actions = m_dist.sample()
                    log_probs = m_dist.log_prob(actions)
                    
                    # Calculate rewards: reward = price change (for buy) or negative price change (for sell)
                    rewards = torch.zeros_like(price_changes)
                    rewards[actions == 0] = price_changes[actions == 0]  # Buy
                    rewards[actions == 1] = -price_changes[actions == 1]  # Sell
                    # Hold (action 2) remains 0
                    
                    # Calculate entropy for exploration
                    entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=1).mean()
                    
                    # Policy gradient loss with entropy regularization
                    policy_loss = -(log_probs * rewards).mean()
                    loss = policy_loss - self.entropy_coef * entropy
                
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                self.scaler.step(opt)
                self.scaler.update()
                
                mini_losses.append(loss.item())
            
            losses.append(np.mean(mini_losses))
        
        # Backtesting & logging (unchanged from previous version)
        tail24 = data_full[-24:]
        seq_np = np.array(tail24)[:, 1:6].astype(np.float32)
        seq_t = torch.tensor(seq_np).unsqueeze(0).to(next(self.models[0].parameters()).device)
        pred_idx, conf, trade_params = self.predict(seq_t)
        result = robust_backtest(self, data_full)
        net_pct = result["net_pct"]
        trades = result["trades"]
        eff_net_pct = result["effective_net_pct"]

        global global_training_loss, global_validation_loss, global_backtest_profit, global_equity_curve
        global global_sharpe, global_inactivity_penalty, global_composite_reward, global_days_without_trading, global_trade_details
        global_backtest_profit.append(eff_net_pct)
        global_equity_curve = result["equity_curve"]
        global_sharpe = result["sharpe"]
        global_inactivity_penalty = result["inactivity_penalty"]
        global_composite_reward = result["composite_reward"]
        global_days_without_trading = result["days_without_trading"]
        global_trade_details = result["trade_details"]

        if dl_val is not None:
            val_loss = self.evaluate_val_loss(dl_val)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state_dicts = [m.state_dict() for m in self.models]
                logging.info(
                    f"New best val_loss: {val_loss:.4f} | Net Profit: {eff_net_pct:.2f}% (raw: {net_pct:.2f}%, trades: {trades}). "
                    f"Sharpe: {result['sharpe']:.4f}, Inactivity Penalty: {result['inactivity_penalty']:.2f}, "
                    f"Days without Trading: {result['days_without_trading']:.2f}, Composite Reward: {result['composite_reward']:.2f}"
                )
                logging.info("Trade Details:\n" + json.dumps(result["trade_details"], indent=2))
            logging.info(
                f"Epoch {self.train_steps+1} => Train Loss={np.mean(losses):.4f}, Val Loss={val_loss:.4f}, "
                f"Net Profit: {eff_net_pct:.2f}% (raw: {net_pct:.2f}%, trades: {trades})"
            )
        else:
            logging.info(
                f"Epoch {self.train_steps+1} => Train Loss={np.mean(losses):.4f}, No Val set => "
                f"Net Profit: {eff_net_pct:.2f}% (raw: {net_pct:.2f}%, trades: {trades})"
            )

        global_training_loss.append(np.mean(losses))
        global_validation_loss.append(val_loss if dl_val is not None else None)
        self.train_steps += 1
        return np.mean(losses), val_loss

    def evaluate_val_loss(self, dl_val):
        for m in self.models:
            m.eval()
        losses = []
        with torch.no_grad():
            for batch_x, batch_next_close in dl_val:
                bx = batch_x.to(self.device, non_blocking=True)
                b_next_close = batch_next_close.to(self.device, non_blocking=True)
                model_losses = []
                for m in self.models:
                    action_probs, _ = m(bx)
                    current_closes = bx[:, -1, 3]
                    price_changes = (b_next_close - current_closes) / (current_closes + 1e-8)
                    m_dist = Categorical(action_probs)
                    actions = m_dist.sample()
                    log_probs = m_dist.log_prob(actions)
                    loss = -(log_probs * ((actions==0)*price_changes + (actions==1)*(-price_changes))).mean()
                    model_losses.append(loss.item())
                losses.append(np.mean(model_losses))
        return float(np.mean(losses))

    def predict(self, x):
        for m in self.models:
            m.eval()
        with torch.no_grad():
            outs, trade_params_list = [], []
            for m in self.models:
                action_probs, trade_params = m(x.to(self.device))
                outs.append(action_probs.cpu().numpy())
                trade_params_list.append({
                    "risk_fraction": trade_params.risk_fraction.cpu().numpy(),
                    "sl_multiplier": trade_params.sl_multiplier.cpu().numpy(),
                    "tp_multiplier": trade_params.tp_multiplier.cpu().numpy(),
                    "attention": trade_params.attention.cpu().numpy()
                })
            avgp = np.mean(outs, axis=0)
            pred_idx = int(np.argmax(avgp[0]))
            conf = float(avgp[0][pred_idx])
            avg_trade_params = {}
            for key in trade_params_list[0]:
                avg_trade_params[key] = np.mean([tp[key][0] for tp in trade_params_list])
        return pred_idx, conf, avg_trade_params

    def vectorized_predict(self, windows_tensor):
        models = self.optimized_models if self.optimized_models is not None else self.models
        for m in models:
            m.eval()
        outs = []
        with torch.no_grad():
            for m in models:
                action_probs, _ = m(windows_tensor.to(self.device))
                outs.append(action_probs)
        avg_probs = sum(outs) / len(outs)
        pred_indices = avg_probs.argmax(dim=1)
        confs = avg_probs.max(dim=1)[0]
        return pred_indices.cpu(), confs.cpu()

    def optimize_models(self, dummy_input):
        self.optimized_models = []
        for m in self.models:
            m.eval()
            traced_model = torch.jit.trace(m, dummy_input, strict=False)
            self.optimized_models.append(traced_model)
        logging.info("Models optimized with TorchScript.")

    def save_best_weights(self, path="best_test_model_weights.pth"):
        if self.best_state_dicts is None:
            logging.warning("No best_state_dicts to save!")
            return
        torch.save({
            'best_val_loss': self.best_val_loss,
            'state_dicts': self.best_state_dicts
        }, path)
        logging.info(f"Best model weights saved to {path}")

    def load_best_weights(self, path="best_test_model_weights.pth"):
        if os.path.isfile(path):
            try:
                with torch.serialization.safe_globals(["numpy._core.multiarray.scalar"]):
                    ckpt = torch.load(path, map_location=self.device, weights_only=False)
                self.best_val_loss = ckpt['best_val_loss']
                self.best_state_dicts = ckpt['state_dicts']
                for m, sd in zip(self.models, self.best_state_dicts):
                    m.load_state_dict(sd, strict=False)
                logging.info(f"Loaded best weights from {path} (val_loss={self.best_val_loss:.4f})")
            except Exception as e:
                logging.error(f"Error loading weights: {e}")
        else:
            logging.warning(f"No checkpoint found at {path}")

###############################################################################
# Robust Backtest Function (Uses Action Probabilities)
###############################################################################
def robust_backtest(ensemble, data_full):
    if len(data_full) < 24:
        return {"net_pct": 0.0, "trades": 0, "effective_net_pct": 0.0, "equity_curve": []}
    device = ensemble.device
    windows = np.lib.stride_tricks.sliding_window_view(np.array(data_full)[:, 1:6], (24, 5)).squeeze()
    windows_tensor = torch.tensor(windows, dtype=torch.float32, device=device)
    pred_indices, _ = ensemble.vectorized_predict(windows_tensor)
    preds = [2] * 23 + pred_indices.tolist()
    df = pd.DataFrame({
        'timestamp': [row[0] for row in data_full],
        'open': [row[1] * 100000 for row in data_full],
        'high': [row[2] * 100000 for row in data_full],
        'low': [row[3] * 100000 for row in data_full],
        'close': [row[4] * 100000 for row in data_full],
        'prediction': preds
    })
    df['previous_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['previous_close']),
                                     abs(df['low'] - df['previous_close'])))
    df['ATR'] = df['tr'].rolling(global_ATR_period, min_periods=1).mean()
    initial_balance = 10000.0
    balance = initial_balance
    commission_rate = 0.0001
    slippage = 0.0002
    LEVERAGE = 10  
    FUNDING_RATE = 0.0001  
    equity_curve = []
    trades = []
    position = {'size': 0.0, 'side': None, 'entry_price': 0.0,
                'stop_loss': 0.0, 'take_profit': 0.0, 'entry_time': None}
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
                    exit_reason = "SL hit"
                    exit_condition = True
                elif row['high'] >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = "TP hit"
                    exit_condition = True
                elif row['prediction'] == 1:
                    exit_reason = "Signal reversal"
                    exit_condition = True
            elif position['side'] == 'short':
                if row['high'] >= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = "SL hit"
                    exit_condition = True
                elif row['low'] <= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = "TP hit"
                    exit_condition = True
                elif row['prediction'] == 0:
                    exit_reason = "Signal reversal"
                    exit_condition = True
            if exit_condition:
                if position['side'] == 'long':
                    exit_price *= (1 - slippage)
                    proceeds = position['size'] * exit_price
                    commission_exit = proceeds * commission_rate
                    profit = (position['size'] * exit_price) - (position['size'] * position['entry_price']) - commission_exit
                    balance += profit
                else:
                    exit_price *= (1 + slippage)
                    commission_exit = abs(position['size']) * exit_price * commission_rate
                    profit = abs(position['size']) * (position['entry_price'] - exit_price) - commission_exit
                    duration_hours = (current_time - position['entry_time']) / (3600 * 1000)
                    funding_cost = abs(position['size']) * position['entry_price'] * FUNDING_RATE * duration_hours
                    profit -= funding_cost
                    balance += profit
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'stop_loss': position['stop_loss'],
                    'take_profit': position['take_profit'],
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'return': (exit_price / position['entry_price'] - 1) * (1 if position['side']=='long' else -1),
                    'duration': current_time - position['entry_time']
                })
                position = {'size': 0.0, 'side': None, 'entry_price': 0.0,
                            'stop_loss': 0.0, 'take_profit': 0.0, 'entry_time': None}
        if position['size'] == 0 and row['prediction'] in (0, 1):
            atr = row['ATR'] if not np.isnan(row['ATR']) else 1.0
            atr = max(atr, 1.0)
            fill_price = current_price * ((1 + slippage) if row['prediction'] == 0 else (1 - slippage))
            stop_distance = global_SL_multiplier * atr
            risk_capital = balance * risk_fraction  
            position_size_risk = risk_capital / (stop_distance + 1e-8)
            max_size = balance * LEVERAGE / fill_price
            position_size = min(position_size_risk, max_size)
            if position_size <= 0:
                continue
            commission_entry = position_size * fill_price * commission_rate
            balance -= commission_entry
            if row['prediction'] == 0:
                position.update({
                    'size': position_size,
                    'side': 'long',
                    'entry_price': fill_price,
                    'stop_loss': fill_price - stop_distance,
                    'take_profit': fill_price + global_TP_multiplier * atr,
                    'entry_time': current_time
                })
            else:
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
            exit_price = final_price * (1 - slippage)
            commission_exit = position['size'] * exit_price * commission_rate
            profit = position['size'] * (exit_price - position['entry_price']) - commission_exit
            balance += profit
        else:
            exit_price = final_price * (1 + slippage)
            commission_exit = abs(position['size']) * exit_price * commission_rate
            profit = abs(position['size']) * (position['entry_price'] - exit_price) - commission_exit
            duration_hours = (df.iloc[-1]['timestamp'] - position['entry_time']) / (3600 * 1000)
            funding_cost = abs(position['size']) * position['entry_price'] * FUNDING_RATE * duration_hours
            profit -= funding_cost
            balance += profit
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.iloc[-1]['timestamp'],
            'side': position['side'],
            'entry_price': position['entry_price'],
            'stop_loss': position['stop_loss'],
            'take_profit': position['take_profit'],
            'exit_price': exit_price,
            'exit_reason': "Final",
            'return': (exit_price / position['entry_price'] - 1) * (1 if position['side']=='long' else -1),
            'duration': df.iloc[-1]['timestamp'] - position['entry_time']
        })
        equity_curve[-1] = (df.iloc[-1]['timestamp'], balance)
    
    final_profit = balance - initial_balance
    net_pct = (final_profit / initial_balance) * 100.0
    bonus_factor = 0.5
    effective_net_pct = net_pct if net_pct <= 0 else net_pct + bonus_factor * len(trades)
    returns = pd.Series([e[1] for e in equity_curve],
                        index=pd.to_datetime([e[0] for e in equity_curve], unit='ms'))
    sharpe_std = returns.pct_change(fill_method=None).std()
    sharpe_ratio = (returns.pct_change(fill_method=None).mean() / sharpe_std * np.sqrt(365*24)) if sharpe_std else 0.0
    max_drawdown = (returns.expanding().max() - returns).max()
    
    # --- Inactivity Penalty & Composite Reward Calculation ---
    inactivity_threshold = 2 * 24 * 3600 * 1000  # 2 days in ms
    inactivity_penalty_weight = 2.0
    inactivity_penalty = 0.0
    days_without_trading = 0.0
    if trades:
        trade_entries = sorted([trade['entry_time'] for trade in trades])
        gaps = []
        start_gap = trade_entries[0] - df.iloc[0]['timestamp']
        if start_gap > inactivity_threshold:
            gaps.append(start_gap)
        for i in range(1, len(trade_entries)):
            gap = trade_entries[i] - trade_entries[i-1]
            if gap > inactivity_threshold:
                gaps.append(gap)
        end_gap = df.iloc[-1]['timestamp'] - trade_entries[-1]
        if end_gap > inactivity_threshold:
            gaps.append(end_gap)
        days_without_trading = sum(((gap - inactivity_threshold) / 86400000.0) for gap in gaps)
        inactivity_penalty = days_without_trading * inactivity_penalty_weight
    else:
        total_duration = df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']
        if total_duration > inactivity_threshold:
            days_without_trading = (total_duration - inactivity_threshold) / 86400000.0
            inactivity_penalty = days_without_trading * inactivity_penalty_weight
    sharpe_weight = 10.0
    composite_reward = net_pct + (sharpe_ratio * sharpe_weight) - inactivity_penalty
    # ---------------------------------------------------------------
    
    return {
        "net_pct": net_pct,
        "trades": len(trades),
        "effective_net_pct": effective_net_pct,
        "sharpe": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "equity_curve": equity_curve,
        "trade_details": trades,
        "composite_reward": composite_reward,
        "inactivity_penalty": inactivity_penalty,
        "days_without_trading": days_without_trading
    }

###############################################################################
# Tkinter GUI Definition (with additional metrics and trade details)
###############################################################################
class TradingGUI:
    def __init__(self, root, ensemble):
        self.root = root
        self.ensemble = ensemble
        self.root.title("Complex AI Trading w/ Robust Backtest + Live Phemex REINFORCEMENT LEARNING")
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
        
        self.frame_trades = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_trades, text="Trade Details")
        self.trade_text = tk.Text(self.frame_trades, width=50, height=20)
        self.trade_text.pack(fill=tk.BOTH, expand=True)
        
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
        self.sharpe_label = ttk.Label(self.info_frame, text="Sharpe Ratio: N/A", font=("Helvetica", 12))
        self.sharpe_label.grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.inactivity_label = ttk.Label(self.info_frame, text="Inactivity Penalty: N/A", font=("Helvetica", 12))
        self.inactivity_label.grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.composite_label = ttk.Label(self.info_frame, text="Composite Reward: N/A", font=("Helvetica", 12))
        self.composite_label.grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        
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
        
        self.update_interval = 2000  # milliseconds
        self.root.after(self.update_interval, self.update_dashboard)

    def update_dashboard(self):
        self.ax_loss.clear()
        self.ax_loss.set_title("Training vs. Validation Loss")
        x1 = range(1, len(global_training_loss) + 1)
        self.ax_loss.plot(x1, global_training_loss, color='blue', marker='o', label='Training Loss')
        val_filtered = [(i + 1, v) for i, v in enumerate(global_validation_loss) if v is not None]
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
                times_dt = [datetime.datetime.fromtimestamp(t/1000.0) for t in times_eq]
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
                t, c = bar[0], bar[4]
                times.append(datetime.datetime.fromtimestamp(t/1000.0))
                closes.append(c)
            self.ax_live.plot(times, closes, marker='o')
        else:
            self.ax_live.text(0.5, 0.5, "No Live Data", ha='center', va='center')
        self.canvas_live.draw()
        
        self.ax_net_profit.clear()
        self.ax_net_profit.set_title("Net Profit (%)")
        if global_backtest_profit:
            x2 = range(1, len(global_backtest_profit) + 1)
            self.ax_net_profit.plot(x2, global_backtest_profit, marker='o', color='green')
        else:
            self.ax_net_profit.text(0.5, 0.5, "No Backtest Data", ha='center', va='center')
        self.ax_net_profit.set_xlabel("Epoch")
        self.ax_net_profit.set_ylabel("Effective Net Profit (%)")
        self.canvas_backtest.draw()
        
        self.ax_details.clear()
        self.ax_details.set_title("Average Attention Weight History")
        if global_attention_weights_history:
            x_vals = list(range(1, len(global_attention_weights_history) + 1))
            self.ax_details.plot(x_vals, global_attention_weights_history, marker='o', color='purple')
            self.ax_details.set_xlabel("Epoch")
            self.ax_details.set_ylabel("Avg Attention Weight")
        else:
            self.ax_details.text(0.5, 0.5, "No Attention Data", ha='center', va='center')
        self.canvas_details.draw()
        
        self.trade_text.delete("1.0", tk.END)
        if global_trade_details:
            self.trade_text.insert(tk.END, json.dumps(global_trade_details, indent=2))
        else:
            self.trade_text.insert(tk.END, "No Trade Details Available")
        
        pred_str = global_current_prediction if global_current_prediction else "N/A"
        conf = global_ai_confidence if global_ai_confidence else 0.0
        steps = global_ai_epoch_count
        self.pred_label.config(text=f"AI Prediction: {pred_str}")
        self.conf_label.config(text=f"Confidence: {conf:.2f}")
        self.epoch_label.config(text=f"Training Steps: {steps}")
        self.sharpe_label.config(text=f"Sharpe Ratio: {global_sharpe:.4f}" if global_sharpe is not None else "Sharpe Ratio: N/A")
        self.inactivity_label.config(text=f"Inactivity Penalty: {global_inactivity_penalty:.2f}" if global_inactivity_penalty is not None else "Inactivity Penalty: N/A")
        self.composite_label.config(text=f"Composite Reward: {global_composite_reward:.2f}" if global_composite_reward is not None else "Composite Reward: N/A")
        
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
        # Use the new RL dataset (no threshold parameter)
        ds_full = HourlyDataset(data, seq_len=24)
        if len(ds_full) < 10:
            logging.warning("Not enough data in CSV. Exiting training thread.")
            return
        ensemble.load_best_weights("best_test_model_weights.pth")
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
                pred_idx, conf, trade_params = ensemble.predict(seq_t)
                label_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
                global global_current_prediction, global_ai_confidence, global_ai_epoch_count, global_attention_weights, global_attention_weights_history
                global_current_prediction = label_map.get(pred_idx, "N/A")
                global_ai_confidence = conf
                global_ai_epoch_count = ensemble.train_steps
                global_attention_weights = trade_params.get("attention", None)
                global_attention_weights_history.append(np.mean(global_attention_weights) if global_attention_weights is not None else 0)
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
                    ds_updated = HourlyDataset(data, seq_len=24)
                    if len(ds_updated) > 10:
                        nt = len(ds_updated)
                        ntr = int(nt * 0.9)
                        nv = nt - ntr
                        ds_train_upd, ds_val_upd = random_split(ds_updated, [ntr, nv])
                        dl_train_upd = DataLoader(ds_train_upd, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
                        dl_val_upd = DataLoader(ds_val_upd, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
                        ensemble.train_one_epoch(dl_train_upd, dl_val_upd, data)
            if ensemble.train_steps % 5 == 0 and ensemble.best_state_dicts is not None:
                ensemble.save_best_weights("best_test_model_weights.pth")
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
            time.sleep(5)
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
# Meta–Control Loop
###############################################################################
def meta_control_loop(ensemble, dataset, interval=200):
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
            f"Learning Rate: {ensemble.optimizers[0].param_groups[0]['lr']}\n"
            f"Dataset Size: {len(dataset.samples)}\n"
            f"Stop Loss Multiplier: {global_SL_multiplier}\n"
            f"Take Profit Multiplier: {global_TP_multiplier}\n"
            f"ATR Period: {global_ATR_period}\n"
        )
        squirtle_res = squirtle_controller(ensemble, summary)
        wartortle_res = wartortle_controller(ensemble, summary, squirtle_res["response"])
        bigmanblastoise_res = bigmanblastoise_overseer(ensemble, wartortle_res["response"])
        moneymaker_res = moneymaker_controller(ensemble, summary, squirtle_res, wartortle_res, bigmanblastoise_res)
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

def main():
    global global_training_loss, global_validation_loss, global_backtest_profit, global_equity_curve, global_ai_adjustments_log, global_current_prediction, global_ai_confidence, global_ai_epoch_count, global_attention_weights, global_attention_weights_history, GLOBAL_THRESHOLD, global_ai_adjustments

    global_training_loss = []
    global_validation_loss = []
    global_backtest_profit = []
    global_equity_curve = []
    global_ai_adjustments_log = "No adjustments yet"
    global_current_prediction = None
    global_ai_confidence = None
    global_ai_epoch_count = 0
    global_attention_weights = []
    global_attention_weights_history = []
    global_ai_adjustments = ""

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
    ensemble = EnsembleModel(device=device, n_models=2, lr=3e-5, weight_decay=0.0)
    connector = PhemexConnector(config)
    stop_event = threading.Event()
    train_th = threading.Thread(target=csv_training_thread, args=(ensemble, data, stop_event, config), daemon=True)
    train_th.start()
    phemex_th = threading.Thread(target=phemex_live_thread, args=(connector, stop_event), daemon=True)
    phemex_th.start()
    
    # Use new RL dataset here as well (no threshold parameter)
    dataset = HourlyDataset(data, seq_len=24)
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
    
    ensemble.save_best_weights("best_test_model_weights.pth")
    save_checkpoint()
    logging.info("All done. The AI keeps running until you kill the script.")

###############################################################################
# Global Variables for Dashboard and Live Data
###############################################################################
global_training_loss = []
global_validation_loss = []
global_backtest_profit = []
global_equity_curve = []
global_ai_adjustments = ""
global_ai_adjustments_log = ""
global_current_prediction = None
global_ai_confidence = None
global_ai_epoch_count = 0
global_attention_weights = []
global_attention_weights_history = []
GLOBAL_THRESHOLD = 0.001
global_phemex_data = []
live_bars_queue = queue.Queue()
gpt_memory_squirtle = []
gpt_memory_wartorttle = []
gpt_memory_bigmanblastoise = []
gpt_memory_moneymaker = []
global_ai_adjustments = ""

if __name__ == "__main__":
    main()
