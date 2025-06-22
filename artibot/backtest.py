"""Backtesting utilities for evaluating strategies."""

# ruff: noqa: F403, F405, E402
import numpy as np
import pandas as pd
import os
from .environment import ensure_dependencies
import talib
import torch

from .utils import rolling_zscore
from .dataset import trailing_sma, HourlyDataset
from . import indicators

import artibot.globals as G
from .execution import submit_order
from . import risk
from .metrics import (
    inactivity_exponential_penalty,
    compute_days_in_profit,
    compute_trade_metrics,
)

FIXED_FEATURES = None


def compute_indicators(
    data_full,
    indicator_hp,
    *,
    with_scaled: bool = False,
    use_ichimoku: bool = False,
    enable_all: bool = False,
):
    """Return indicator arrays for ``data_full``.

    When ``with_scaled`` is ``True`` the returned dictionary also contains a
    ``scaled`` key with normalised feature vectors matching the dataset
    preprocessing.
    """

    from artibot.hyperparams import WARMUP_STEPS

    global FIXED_FEATURES
    if G.global_step < WARMUP_STEPS and FIXED_FEATURES is not None and not enable_all:
        return FIXED_FEATURES

    raw = np.array(data_full, dtype=np.float64)
    closes = raw[:, 4]
    highs = raw[:, 2]
    lows = raw[:, 3]
    volume = raw[:, 5] if raw.shape[1] > 5 else np.zeros_like(closes)

    atr_period = indicator_hp.atr_period

    rsi_period = max(2, min(getattr(indicator_hp, "rsi_period", 14), 50))
    sma_period = max(2, min(getattr(indicator_hp, "sma_period", 10), 100))
    fast_macd = max(
        2,
        min(
            getattr(indicator_hp, "macd_fast", 12),
            getattr(indicator_hp, "macd_slow", 26) - 1,
        ),
    )
    slow_macd = max(fast_macd + 1, min(getattr(indicator_hp, "macd_slow", 26), 200))
    sig_macd = max(1, min(getattr(indicator_hp, "macd_signal", 9), 50))

    out = {}
    cols = [raw[:, 1:6]]

    # === NEW FEATURE BLOCK (sentiment / macro / rvol) =======================
    import artibot.feature_store as _fs

    sent_vec = np.array(
        [_fs.news_sentiment(int(t)) for t in raw[:, 0]], dtype=np.float32
    )
    macro_vec = np.array(
        [_fs.macro_surprise(int(t)) for t in raw[:, 0]], dtype=np.float32
    )
    rvol_vec = np.array([_fs.realised_vol(int(t)) for t in raw[:, 0]], dtype=np.float32)
    out["sent_24h"] = sent_vec
    out["macro_z"] = macro_vec
    out["rvol_7d"] = rvol_vec
    cols.extend([sent_vec, macro_vec, rvol_vec])

    if getattr(indicator_hp, "use_sma", True):
        sma = trailing_sma(closes, sma_period)
        out["sma"] = sma.astype(np.float32)
        cols.append(out["sma"])

    if getattr(indicator_hp, "use_rsi", True):
        rsi = talib.RSI(closes, timeperiod=rsi_period)
        out["rsi"] = rsi.astype(np.float32)
        cols.append(out["rsi"])

    if getattr(indicator_hp, "use_macd", True):
        macd_, _sig, _hist = talib.MACD(
            closes,
            fastperiod=fast_macd,
            slowperiod=slow_macd,
            signalperiod=sig_macd,
        )
        out["macd"] = macd_.astype(np.float32)
        cols.append(out["macd"])

    if getattr(indicator_hp, "use_ema", False):
        ema_v = indicators.ema(closes, period=getattr(indicator_hp, "ema_period", 20))
        out["ema"] = ema_v.astype(np.float32)
        cols.append(out["ema"])

    atr_vals = indicators.atr(highs, lows, closes, period=atr_period)
    out["atr"] = atr_vals.astype(np.float32)
    if getattr(indicator_hp, "use_atr", True):
        cols.append(out["atr"])

    if getattr(indicator_hp, "use_vortex", True):
        vp, vn = indicators.vortex(
            highs, lows, closes, period=getattr(indicator_hp, "vortex_period", 14)
        )
        out["vortex_pos"] = vp.astype(np.float32)
        out["vortex_neg"] = vn.astype(np.float32)
        cols.extend([out["vortex_pos"], out["vortex_neg"]])

    if getattr(indicator_hp, "use_cmf", True):
        cmf_v = indicators.cmf(
            highs, lows, closes, volume, period=getattr(indicator_hp, "cmf_period", 20)
        )
        out["cmf"] = cmf_v.astype(np.float32)
        cols.append(out["cmf"])

    if getattr(indicator_hp, "use_donchian", False):
        up, lo, mid = indicators.donchian(
            highs, lows, period=getattr(indicator_hp, "donchian_period", 20)
        )
        out["don_up"] = up.astype(np.float32)
        out["don_lo"] = lo.astype(np.float32)
        out["don_mid"] = mid.astype(np.float32)
        cols.extend([out["don_up"], out["don_lo"], out["don_mid"]])

    if getattr(indicator_hp, "use_kijun", False):
        kij = indicators.kijun(
            highs, lows, period=getattr(indicator_hp, "kijun_period", 26)
        )
        out["kijun"] = kij.astype(np.float32)
        cols.append(out["kijun"])

    if getattr(indicator_hp, "use_tenkan", False):
        ten = indicators.tenkan(
            highs, lows, period=getattr(indicator_hp, "tenkan_period", 9)
        )
        out["tenkan"] = ten.astype(np.float32)
        cols.append(out["tenkan"])

    if getattr(indicator_hp, "use_displacement", False):
        disp = np.roll(closes, getattr(indicator_hp, "displacement", 26))
        disp[: getattr(indicator_hp, "displacement", 26)] = np.nan
        out["disp"] = disp.astype(np.float32)
        cols.append(out["disp"])

    if use_ichimoku:
        tenkan, kijun, span_a, span_b = indicators.ichimoku(highs, lows)
        out["tenkan"] = tenkan.astype(np.float32)
        out["kijun"] = kijun.astype(np.float32)
        out["span_a"] = span_a.astype(np.float32)
        out["span_b"] = span_b.astype(np.float32)
        cols.extend([out["tenkan"], out["kijun"], out["span_a"], out["span_b"]])

    if with_scaled:
        feats = np.column_stack(cols)
        feats = np.nan_to_num(feats)
        out["scaled"] = rolling_zscore(feats, window=50)

    if G.global_step == 0 and enable_all:
        FIXED_FEATURES = out
        return out

    if G.global_step < WARMUP_STEPS and not enable_all and FIXED_FEATURES is not None:
        return FIXED_FEATURES

    return out


###############################################################################
# robust_backtest
###############################################################################
def robust_backtest(ensemble, data_full, indicators=None):
    """Run a simplified backtest and return key metrics.

    Parameters
    ----------
    ensemble:
        Ensemble providing ``vectorized_predict`` and indicator settings.
    data_full:
        Full OHLCV history used for backtesting.
    indicators:
        Optional dictionary with precomputed ``sma``, ``rsi`` and ``macd``
        arrays. When ``None`` (default), they are derived on the fly.
    """
    if len(data_full) < 24:
        return {
            "net_pct": 0.0,
            "trades": 0,
            "effective_net_pct": 0.0,
            "equity_curve": [],
        }

    LEVERAGE = 2  # reduce draw-down pressure
    min_hold_seconds = G.global_min_hold_seconds
    commission_rate = 0.0006  # 0.06 % taker fee (Phemex) ✓
    FUNDING_RATE = 0.0001
    device = ensemble.device

    hp = ensemble.indicator_hparams

    # (5) If meta-agent is adjusting threshold, store it in ensemble or define a separate variable.
    # For a simpler demonstration, we keep using GLOBAL_THRESHOLD, but you could do:
    # threshold = ensemble.dynamic_threshold if ensemble.dynamic_threshold is not None else GLOBAL_THRESHOLD
    # or pass it in the function signature.

    # (9) Composite Reward: alpha=3.0, beta=0.5, gamma=0.8, delta=0.1
    alpha = 3.0
    beta = 0.5
    gamma = 0.8
    delta = 0.1

    def _price_with_noise(side: str, price: float) -> float:
        return submit_order(lambda **kw: kw["price"], side, 0.0, price, delay=0.0)

    raw_data = np.array(data_full, dtype=np.float64)
    if raw_data[:, 0].max() > 1_000_000_000_000:
        raw_data[:, 0] //= 1000

    if indicators is None:
        indic = compute_indicators(data_full, hp, with_scaled=True)
    else:
        indic = indicators

    sma = indic.get("sma")
    rsi = indic.get("rsi")
    macd_ = indic.get("macd")

    if "scaled" in indic:
        extd = indic["scaled"].astype(np.float32)
    else:
        cols = [raw_data[:, 1:6]]
        if sma is not None:
            cols.append(sma.astype(np.float32))
        if rsi is not None:
            cols.append(rsi.astype(np.float32))
        if macd_ is not None:
            cols.append(macd_.astype(np.float32))
        feats = np.column_stack(cols)
        feats = np.nan_to_num(feats)
        extd = rolling_zscore(feats, window=50)

    timestamps = raw_data[:, 0]

    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(extd, (24, extd.shape[1])).squeeze()

    windows = np.clip(windows, -50.0, 50.0)

    windows = np.nan_to_num(windows)
    windows_t = torch.tensor(windows, dtype=torch.float32, device=device)
    pred_indices, _, avg_params = ensemble.vectorized_predict(windows_t, batch_size=512)
    preds = [2] * 23 + pred_indices.tolist()

    timestamps = timestamps.astype(np.int64)
    highs = raw_data[:, 2]
    lows = raw_data[:, 3]
    closes = raw_data[:, 4]
    preds = np.array(preds, dtype=np.int64)

    n = min(len(preds), len(closes))
    timestamps = timestamps[:n]
    highs = highs[:n]
    lows = lows[:n]
    closes = closes[:n]
    preds = preds[:n]

    prev_close = np.concatenate(([np.nan], closes[:-1]))
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)),
    )
    atr = pd.Series(tr).rolling(hp.atr_period, min_periods=1).mean().to_numpy()

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

    for i in range(len(closes)):
        cur_t = int(timestamps[i])
        cur_p = closes[i]
        high = highs[i]
        low = lows[i]
        pred = preds[i]
        atr_i = atr[i]
        if pos["size"] != 0:
            exit_condition = False
            exit_price = cur_p
            exit_reason = ""
            if pos["side"] == "long":
                if low <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "SL hit"
                    exit_condition = True
                elif high >= pos["take_profit"]:
                    exit_price = pos["take_profit"]
                    exit_reason = "TP hit"
                    exit_condition = True
                elif pred == 1:
                    exit_reason = "Signal reversal"
                    exit_condition = True
            else:
                if high >= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "SL hit"
                    exit_condition = True
                elif low <= pos["take_profit"]:
                    exit_price = pos["take_profit"]
                    exit_reason = "TP hit"
                    exit_condition = True
                elif pred == 0:
                    exit_reason = "Signal reversal"
                    exit_condition = True
            if exit_condition:
                last_exit_t = cur_t
                if pos["side"] == "long":
                    exit_price = _price_with_noise("sell", exit_price)
                    proceeds = pos["size"] * exit_price
                    comm_exit = proceeds * commission_rate
                    profit = proceeds - (pos["size"] * pos["entry_price"]) - comm_exit
                    bal += profit
                else:
                    exit_price = _price_with_noise("buy", exit_price)
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
                        "size": pos["size"],
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
        if pos["size"] == 0 and pred in (0, 1):
            if last_exit_t is not None and (cur_t - last_exit_t) < min_hold_seconds:
                pass
            else:
                atr_val = atr_i if not np.isnan(atr_i) else 1.0
                atr_val = max(1.0, atr_val)
                fill_p = _price_with_noise("buy" if pred == 0 else "sell", cur_p)
                st_dist = sl_m * atr_val
                tp_val = (
                    fill_p + tp_m * atr_val if pred == 0 else fill_p - tp_m * atr_val
                )
                pos_sz = risk.position_size(bal, rf, st_dist, fill_p, LEVERAGE)
                if pos_sz > 0:
                    comm_entry = pos_sz * fill_p * commission_rate
                    bal -= comm_entry
                    if pred == 0:
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
        final_price = closes[-1]
        if pos["side"] == "long":
            exit_price = _price_with_noise("sell", final_price)
            comm_exit = pos["size"] * exit_price * commission_rate
            pf = pos["size"] * (exit_price - pos["entry_price"]) - comm_exit
            bal += pf
        else:
            exit_price = _price_with_noise("buy", final_price)
            comm_exit = abs(pos["size"]) * exit_price * commission_rate
            pf = abs(pos["size"]) * (pos["entry_price"] - exit_price) - comm_exit
            hrs = (timestamps[-1] - pos["entry_time"]) / 3600.0
            fund = abs(pos["size"]) * pos["entry_price"] * FUNDING_RATE * hrs
            pf -= fund
            bal += pf
        trades.append(
            {
                "entry_time": pos["entry_time"],
                "exit_time": int(timestamps[-1]),
                "side": pos["side"],
                "size": pos["size"],
                "entry_price": pos["entry_price"],
                "stop_loss": pos["stop_loss"],
                "take_profit": pos["take_profit"],
                "exit_price": exit_price,
                "exit_reason": "Final",
                "return": (exit_price / pos["entry_price"] - 1)
                * (1 if pos["side"] == "long" else -1),
                "duration": int(timestamps[-1]) - pos["entry_time"],
            }
        )
        eq_curve[-1] = (int(timestamps[-1]), bal)

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
        start_gap = te_sorted[0] - int(timestamps[0])
        if start_gap > 0:
            gaps.append(start_gap)
        for i in range(1, len(te_sorted)):
            gp = te_sorted[i] - te_sorted[i - 1]
            if gp > 0:
                gaps.append(gp)
        end_gap = int(timestamps[-1]) - te_sorted[-1]
        if end_gap > 0:
            gaps.append(end_gap)
        for g in gaps:
            tot_inact_pen += inactivity_exponential_penalty(g)
    else:
        if len(timestamps) > 1:
            total_s = int(timestamps[-1]) - int(timestamps[0])
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

    metrics = compute_trade_metrics(trades)
    win_rate = metrics["win_rate"]
    profit_factor = metrics["profit_factor"]
    avg_duration = metrics["avg_duration"]
    avg_win = metrics["avg_win"]
    avg_loss = metrics["avg_loss"]

    net_score = net_pct / 100.0
    shr_score = sharpe
    trade_count = len(trades)
    trade_term = trade_count * delta
    # final composite
    # composite_reward= (alpha* net_score + beta* shr_score - gamma* dd_pen + trade_term)
    # composite_reward-= tot_inact_pen
    composite_reward = 0.0
    if G.use_net_term:
        composite_reward += alpha * net_score
    if G.use_sharpe_term:
        composite_reward += beta * shr_score * 2
    if G.use_drawdown_term:
        composite_reward += gamma * (1 - abs(mdd))
    if G.use_trade_term:
        composite_reward += trade_term * 3
    if G.use_profit_days_term:
        composite_reward += (days_in_pf / 365) * 1000
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
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_duration": avg_duration,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


if __name__ == "__main__":
    from .utils.torch_threads import set_threads
    from .dataset import load_csv_hourly, HourlyDataset
    from .ensemble import EnsembleModel
    from .training import csv_training_thread
    from .utils import get_device
    from .hyperparams import IndicatorHyperparams
    from .bot_app import CONFIG
    import threading

    set_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
    ensure_dependencies()

    data = load_csv_hourly("Gemini_BTCUSD_1h.csv")
    ds_tmp = HourlyDataset(
        data,
        seq_len=24,
        indicator_hparams=IndicatorHyperparams(
            rsi_period=14, sma_period=10, macd_fast=12, macd_slow=26, macd_signal=9
        ),
        atr_threshold_k=1.5,
        train_mode=False,
    )
    n_features = ds_tmp[0][0].shape[1]
    ens = EnsembleModel(device=get_device(), n_models=1, n_features=n_features)
    stop = threading.Event()
    csv_training_thread(
        ens,
        data,
        stop,
        CONFIG,
        use_prev_weights=False,
        max_epochs=1,
    )
    result = robust_backtest(ens, data)
    print(result)
