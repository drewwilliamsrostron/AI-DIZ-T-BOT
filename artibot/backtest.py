"""Backtesting utilities for evaluating strategies."""

# ruff: noqa: F403, F405
import numpy as np
import pandas as pd
import talib
import torch

import artibot.globals as G
from .metrics import inactivity_exponential_penalty, compute_days_in_profit


###############################################################################
# robust_backtest
###############################################################################
def robust_backtest(ensemble, data_full):
    """Run a simplified backtest and return key metrics."""
    if len(data_full) < 24:
        return {
            "net_pct": 0.0,
            "trades": 0,
            "effective_net_pct": 0.0,
            "equity_curve": [],
        }

    LEVERAGE = 10
    min_hold_seconds = G.global_min_hold_seconds
    commission_rate = 0.0001
    slippage = 0.0002
    FUNDING_RATE = 0.0001
    device = ensemble.device

    hp = ensemble.indicator_hparams

    # (5) If meta-agent is adjusting threshold, store it in ensemble or define a separate variable.
    # For a simpler demonstration, we keep using GLOBAL_THRESHOLD, but you could do:
    # threshold = ensemble.dynamic_threshold if ensemble.dynamic_threshold is not None else GLOBAL_THRESHOLD
    # or pass it in the function signature.

    # Clamps for RSI, SMA, MACD
    rsi_period = max(2, min(hp.rsi_period, 50))
    sma_period = max(2, min(hp.sma_period, 100))
    fast_macd = max(2, min(hp.macd_fast, hp.macd_slow - 1))
    slow_macd = max(fast_macd + 1, min(hp.macd_slow, 200))
    sig_macd = max(1, min(hp.macd_signal, 50))

    # (9) Composite Reward: alpha=3.0, beta=0.5, gamma=0.8, delta=0.1
    alpha = 3.0
    beta = 0.5
    gamma = 0.8
    delta = 0.1

    raw_data = np.array(data_full, dtype=np.float64)
    if raw_data[:, 0].max() > 1_000_000_000_000:
        raw_data[:, 0] //= 1000
    closes = raw_data[:, 4]

    sma = np.convolve(closes, np.ones(sma_period) / sma_period, mode="same")
    rsi = talib.RSI(closes, timeperiod=rsi_period)
    macd_, macdsig_, macdhist_ = talib.MACD(
        closes, fastperiod=fast_macd, slowperiod=slow_macd, signalperiod=sig_macd
    )

    extd = np.column_stack(
        [
            raw_data[:, 1:6],
            sma.astype(np.float32),
            rsi.astype(np.float32),
            macd_.astype(np.float32),
        ]
    ).astype(np.float32)

    df_extd = pd.DataFrame(extd)
    roll_mean = df_extd.rolling(window=50, min_periods=1).mean()
    roll_std = df_extd.rolling(window=50, min_periods=1).std().replace(0, 1e-8)
    extd = ((df_extd - roll_mean) / roll_std).to_numpy()

    extd = np.clip(extd, -50.0, 50.0)

    extd = np.nan_to_num(extd)
    timestamps = raw_data[:, 0]

    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(extd, (24, 8)).squeeze()

    windows = np.clip(windows, -50.0, 50.0)

    windows = np.nan_to_num(windows)
    windows_t = torch.tensor(windows, dtype=torch.float32, device=device)
    pred_indices, _, avg_params = ensemble.vectorized_predict(windows_t, batch_size=512)
    preds = [2] * 23 + pred_indices.tolist()

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": raw_data[:, 1],
            "high": raw_data[:, 2],
            "low": raw_data[:, 3],
            "close": raw_data[:, 4],
            "prediction": preds,
        }
    )
    df["previous_close"] = df["close"].shift(1)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            np.abs(df["high"] - df["previous_close"]),
            np.abs(df["low"] - df["previous_close"]),
        ),
    )
    df["ATR"] = df["tr"].rolling(G.global_ATR_period, min_periods=1).mean()

    init_bal = 100.0
    bal = init_bal
    eq_curve = []
    trades = []
    pos = {
        "size": 0.0,
        "side": None,
        "entry_price": 0.0,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "entry_time": None,
    }
    last_exit_t = None

    sl_m = avg_params["sl_multiplier"].item()
    tp_m = avg_params["tp_multiplier"].item()
    rf_raw = avg_params["risk_fraction"].item()
    rf = 0.2 * rf_raw + 0.01

    for i, row in df.iterrows():
        cur_t = row["timestamp"]
        cur_p = row["close"]
        if pos["size"] != 0:
            exit_condition = False
            exit_price = cur_p
            exit_reason = ""
            if pos["side"] == "long":
                if row["low"] <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "SL hit"
                    exit_condition = True
                elif row["high"] >= pos["take_profit"]:
                    exit_price = pos["take_profit"]
                    exit_reason = "TP hit"
                    exit_condition = True
                elif row["prediction"] == 1:
                    exit_reason = "Signal reversal"
                    exit_condition = True
            else:
                if row["high"] >= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "SL hit"
                    exit_condition = True
                elif row["low"] <= pos["take_profit"]:
                    exit_price = pos["take_profit"]
                    exit_reason = "TP hit"
                    exit_condition = True
                elif row["prediction"] == 0:
                    exit_reason = "Signal reversal"
                    exit_condition = True
            if exit_condition:
                last_exit_t = cur_t
                if pos["side"] == "long":
                    exit_price *= 1 - slippage
                    proceeds = pos["size"] * exit_price
                    comm_exit = proceeds * commission_rate
                    profit = proceeds - (pos["size"] * pos["entry_price"]) - comm_exit
                    bal += profit
                else:
                    exit_price *= 1 + slippage
                    comm_exit = abs(pos["size"]) * exit_price * commission_rate
                    profit = (
                        abs(pos["size"]) * (pos["entry_price"] - exit_price) - comm_exit
                    )
                    hrs = (cur_t - pos["entry_time"]) / 3600.0
                    funding = abs(pos["size"]) * pos["entry_price"] * FUNDING_RATE * hrs
                    profit -= funding
                    bal += profit
                trades.append(
                    {
                        "entry_time": pos["entry_time"],
                        "exit_time": cur_t,
                        "side": pos["side"],
                        "entry_price": pos["entry_price"],
                        "stop_loss": pos["stop_loss"],
                        "take_profit": pos["take_profit"],
                        "exit_price": exit_price,
                        "exit_reason": exit_reason,
                        "return": (exit_price / pos["entry_price"] - 1)
                        * (1 if pos["side"] == "long" else -1),
                        "duration": cur_t - pos["entry_time"],
                    }
                )
                pos = {
                    "size": 0.0,
                    "side": None,
                    "entry_price": 0.0,
                    "stop_loss": 0.0,
                    "take_profit": 0.0,
                    "entry_time": None,
                }
        if pos["size"] == 0 and row["prediction"] in (0, 1):
            if last_exit_t is not None and (cur_t - last_exit_t) < min_hold_seconds:
                pass
            else:
                atr = row["ATR"] if not np.isnan(row["ATR"]) else 1.0
                atr = max(1.0, atr)
                fill_p = cur_p * (
                    1 + slippage if row["prediction"] == 0 else 1 - slippage
                )
                st_dist = sl_m * atr
                tp_val = (
                    fill_p + tp_m * atr
                    if row["prediction"] == 0
                    else fill_p - tp_m * atr
                )
                risk_cap = bal * rf
                pos_size_risk = risk_cap / (st_dist + 1e-8)
                max_sz = (bal * LEVERAGE) / fill_p
                pos_sz = min(pos_size_risk, max_sz)
                if pos_sz > 0:
                    comm_entry = pos_sz * fill_p * commission_rate
                    bal -= comm_entry
                    if row["prediction"] == 0:
                        pos.update(
                            {
                                "size": pos_sz,
                                "side": "long",
                                "entry_price": fill_p,
                                "stop_loss": fill_p - st_dist,
                                "take_profit": tp_val,
                                "entry_time": cur_t,
                            }
                        )
                    else:
                        pos.update(
                            {
                                "size": -pos_sz,
                                "side": "short",
                                "entry_price": fill_p,
                                "stop_loss": fill_p + st_dist,
                                "take_profit": tp_val,
                                "entry_time": cur_t,
                            }
                        )
        curr_eq = bal
        if pos["size"] != 0:
            if pos["side"] == "long":
                curr_eq += pos["size"] * (cur_p - pos["entry_price"])
            else:
                curr_eq += abs(pos["size"]) * (pos["entry_price"] - cur_p)
        eq_curve.append((cur_t, curr_eq))

    if pos["size"] != 0:
        final_price = df.iloc[-1]["close"]
        if pos["side"] == "long":
            exit_price = final_price * (1 - slippage)
            comm_exit = pos["size"] * exit_price * commission_rate
            pf = pos["size"] * (exit_price - pos["entry_price"]) - comm_exit
            bal += pf
        else:
            exit_price = final_price * (1 + slippage)
            comm_exit = abs(pos["size"]) * exit_price * commission_rate
            pf = abs(pos["size"]) * (pos["entry_price"] - exit_price) - comm_exit
            hrs = (df.iloc[-1]["timestamp"] - pos["entry_time"]) / 3600.0
            fund = abs(pos["size"]) * pos["entry_price"] * FUNDING_RATE * hrs
            pf -= fund
            bal += pf
        trades.append(
            {
                "entry_time": pos["entry_time"],
                "exit_time": df.iloc[-1]["timestamp"],
                "side": pos["side"],
                "entry_price": pos["entry_price"],
                "stop_loss": pos["stop_loss"],
                "take_profit": pos["take_profit"],
                "exit_price": exit_price,
                "exit_reason": "Final",
                "return": (exit_price / pos["entry_price"] - 1)
                * (1 if pos["side"] == "long" else -1),
                "duration": df.iloc[-1]["timestamp"] - pos["entry_time"],
            }
        )
        eq_curve[-1] = (df.iloc[-1]["timestamp"], bal)

    final_profit = bal - init_bal
    net_pct = (final_profit / init_bal) * 100.0
    if not np.isfinite(net_pct):
        net_pct = 0.0
    # Clamp extreme values so a runaway trade does not dominate training
    eff_net_pct = float(np.clip(net_pct, -1000.0, 1000.0))

    # inactivity penalty
    tot_inact_pen = 0.0
    if trades:
        te_sorted = sorted([t["entry_time"] for t in trades])
        gaps = []
        start_gap = te_sorted[0] - df.iloc[0]["timestamp"]
        if start_gap > 0:
            gaps.append(start_gap)
        for i in range(1, len(te_sorted)):
            gp = te_sorted[i] - te_sorted[i - 1]
            if gp > 0:
                gaps.append(gp)
        end_gap = df.iloc[-1]["timestamp"] - te_sorted[-1]
        if end_gap > 0:
            gaps.append(end_gap)
        for g in gaps:
            tot_inact_pen += inactivity_exponential_penalty(g)
    else:
        if len(df) > 1:
            total_s = df.iloc[-1]["timestamp"] - df.iloc[0]["timestamp"]
            tot_inact_pen += inactivity_exponential_penalty(total_s)

    days_in_pf = compute_days_in_profit(eq_curve, init_bal)

    eqdf = pd.DataFrame(eq_curve, columns=["timestamp", "balance"])
    if eqdf["timestamp"].max() > 1_000_000_000_000:
        eqdf["timestamp"] //= 1000
    eqdf["dt"] = pd.to_datetime(eqdf["timestamp"], unit="s")
    eqdf.set_index("dt", inplace=True)
    eqdf = eqdf.resample("1D").last().dropna()
    if len(eqdf) < 2:
        sharpe = 0.0
        mdd = 0.0
    else:
        dret = eqdf["balance"].pct_change().dropna()
        mu = dret.mean()
        sigma = dret.std()
        sharpe = (mu * 252) / (sigma * np.sqrt(252)) if sigma > 1e-12 else 0.0
        rollmax = eqdf["balance"].cummax()
        dd = (eqdf["balance"] - rollmax) / rollmax
        mdd = dd.min()

    net_score = net_pct / 100.0
    shr_score = sharpe
    trade_count = len(trades)
    trade_term = trade_count * delta
    # final composite
    # composite_reward= (alpha* net_score + beta* shr_score - gamma* dd_pen + trade_term)
    # composite_reward-= tot_inact_pen
    composite_reward = (  # new stuff
        alpha * net_score
        + beta * shr_score * 2  # Sharper reward for risk-adjusted returns
        + gamma * (1 - abs(mdd))  # Better drawdown handling
        + trade_term * 3  # Stronger incentive for reasonable trade frequency
        + (days_in_pf / 365) * 1000  # Strong bonus for consistent profitability
    )
    composite_reward = float(np.clip(composite_reward, -100.0, 100.0))
    return {
        "net_pct": net_pct,
        "trades": trade_count,
        "effective_net_pct": eff_net_pct,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "equity_curve": eq_curve,
        "trade_details": trades,
        "composite_reward": composite_reward,
        "inactivity_penalty": tot_inact_pen,
        "days_without_trading": 0.0,
        "days_in_profit": days_in_pf,
    }
