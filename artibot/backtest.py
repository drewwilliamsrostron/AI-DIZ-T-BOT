"""Backtesting utilities for evaluating strategies."""

# ruff: noqa: F403, F405, E402
import numpy as np
import pandas as pd
import os
from .environment import ensure_dependencies
import talib
import torch
from .constants import FEATURE_DIMENSION
from config import FEATURE_CONFIG
import logging

from .utils import (
    validate_features,
    zero_disabled,
    rolling_zscore,
)
from .dataset import trailing_sma, HourlyDataset
from .hyperparams import IndicatorHyperparams, WARMUP_STEPS
from . import indicators

import artibot.globals as G
from .execution import submit_order
from . import risk
from .metrics import (
    inactivity_exponential_penalty,
    compute_days_in_profit,
    compute_trade_metrics,
    summarise_net_positions,
)
from .utils.reward_utils import sortino_ratio, omega_ratio, calmar_ratio

FIXED_FEATURES = None


def prepare_backtest_data(data: np.ndarray) -> np.ndarray:
    """Validate and return ``data`` for backtesting."""

    expected = len(FEATURE_CONFIG.get("feature_columns", []))
    hp = IndicatorHyperparams()
    fixed = compute_indicators(data, hp)["features"]
    assert fixed.shape[1] == expected, f"wanted {expected}, got {fixed.shape[1]}"
    return fixed


def compute_indicators(
    data_full,
    indicator_hp,
    *,
    with_scaled: bool = False,
    use_ichimoku: bool = False,
):
    """Return computed indicator arrays for ``data_full``.

    The returned feature matrix always has :data:`FEATURE_DIMENSION` columns
    regardless of which indicators are enabled. Missing columns are padded with
    zeros and the accompanying ``mask`` reflects active indicators.

    Parameters
    ----------
    with_scaled:
        When ``True`` also return a z-score normalised array under ``scaled``.
    use_ichimoku:
        Include Ichimoku indicators.
    """

    raw = np.array(data_full, dtype=np.float64)
    closes = raw[:, 4]
    highs = raw[:, 2]
    lows = raw[:, 3]
    volume = raw[:, 5] if raw.shape[1] > 5 else np.zeros_like(closes)

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

    cols: list[np.ndarray] = []
    mask: list[int] = []

    def add_feat(col: np.ndarray, active: bool) -> None:
        nonlocal cols, mask
        cols.append(np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0))
        mask.append(1 if active else 0)

    for c in raw[:, 1:6].T:
        add_feat(c, True)

    sma_arr = trailing_sma(closes, sma_period)
    add_feat(sma_arr, indicator_hp.use_sma)

    rsi_arr = talib.RSI(closes, timeperiod=rsi_period)
    add_feat(rsi_arr, indicator_hp.use_rsi)

    macd_arr, _sig, _hist = talib.MACD(
        closes,
        fastperiod=fast_macd,
        slowperiod=slow_macd,
        signalperiod=sig_macd,
    )
    add_feat(macd_arr, indicator_hp.use_macd)

    ema_arr = indicators.ema(closes, period=getattr(indicator_hp, "ema_period", 20))
    add_feat(ema_arr, indicator_hp.use_ema)

    ema50_arr = indicators.ema(closes, period=50)
    add_feat(ema50_arr, True)

    atr_arr = indicators.atr(highs, lows, closes, period=indicator_hp.atr_period)
    add_feat(atr_arr, indicator_hp.use_atr)

    vp, vn = indicators.vortex(
        highs, lows, closes, period=getattr(indicator_hp, "vortex_period", 14)
    )
    add_feat(vp, indicator_hp.use_vortex)
    add_feat(vn, indicator_hp.use_vortex)

    cmf_arr = indicators.cmf(
        highs, lows, closes, volume, period=getattr(indicator_hp, "cmf_period", 20)
    )
    add_feat(cmf_arr, indicator_hp.use_cmf)

    _up, _lo, mid = indicators.donchian(
        highs, lows, period=getattr(indicator_hp, "donchian_period", 20)
    )
    add_feat(mid, indicator_hp.use_donchian)

    tenkan, _kijun, _a, _b = indicators.ichimoku(highs, lows)
    add_feat(tenkan, bool(use_ichimoku or getattr(indicator_hp, "use_tenkan", False)))

    EXPECTED_DIM = 16
    cur_dim = len(cols)
    if cur_dim < EXPECTED_DIM:
        # pad with zero-arrays so model always sees 16 columns
        pad = [np.zeros_like(raw[:, 4])] * (EXPECTED_DIM - cur_dim)
        cols.extend(pad)
    elif cur_dim > EXPECTED_DIM:
        # truncate extras to keep contract
        cols = cols[:EXPECTED_DIM]

    feats = np.column_stack(cols).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    mask_arr = np.asarray(mask, dtype=np.uint8)

    # ensure constant feature dimension
    if feats.shape[1] < FEATURE_DIMENSION:
        pad = FEATURE_DIMENSION - feats.shape[1]
        feats = np.concatenate(
            [feats, np.zeros((feats.shape[0], pad), dtype=feats.dtype)], axis=1
        )
        mask_arr = np.concatenate([mask_arr, np.zeros(pad, dtype=np.uint8)])
    elif feats.shape[1] > FEATURE_DIMENSION:
        feats = feats[:, :FEATURE_DIMENSION]
        mask_arr = mask_arr[:FEATURE_DIMENSION]

    result = {"features": feats, "mask": mask_arr}
    if with_scaled:
        tmp = feats.copy()
        active = mask_arr.astype(bool)
        from sklearn.impute import KNNImputer

        imputer = KNNImputer(n_neighbors=5)
        tmp[:, active] = imputer.fit_transform(tmp[:, active])
        tmp = zero_disabled(tmp, active)
        tmp = rolling_zscore(tmp, window=50, mask=active)
        tmp = zero_disabled(tmp, active)
        tmp = np.nan_to_num(tmp, nan=0.0, posinf=0.0, neginf=0.0)
        result["scaled"] = tmp.astype(np.float32)

    return result


def update_indicator_toggles(window: np.ndarray, hp: IndicatorHyperparams) -> None:
    """Mutate ``hp`` based on simple market heuristics.

    Parameters
    ----------
    window:
        Recent OHLCV rows with shape ``(N, >=6)`` used to compute statistics.
    hp:
        ``IndicatorHyperparams`` instance to update in-place.
    """

    closes = window[:, 4].astype(float)
    vols = window[:, 5] if window.shape[1] > 5 else np.zeros_like(closes)

    vol_ratio = np.std(closes) / (np.mean(closes) + 1e-9)
    hp.use_atr = bool(vol_ratio > 0.015)
    hp.use_cmf = bool(vols.mean() > 0 and vols[-1] > vols.mean() * 1.2)
    hp.use_sma = bool(closes[-1] > closes.mean())


###############################################################################
# robust_backtest
###############################################################################
def robust_backtest(
    ensemble,
    data_full,
    indicators=None,
    *,
    indicator_hp=None,
    dynamic_indicators: bool = False,
    cluster_count: int | None = None,
) -> dict:
    """Run a simplified backtest and return key metrics.

    Parameters
    ----------
    ensemble:
        Ensemble providing ``vectorized_predict`` and indicator settings.
    data_full:
        Full OHLCV history used for backtesting. Must contain raw OHLCV rows
        with at least five columns ``[ts, open, high, low, close, ...]``.
    indicators:
        Optional dictionary with precomputed indicator arrays.
    indicator_hp:
        ``IndicatorHyperparams`` instance overriding ``ensemble.indicator_hparams``.
        When ``None`` (default) the ensemble's stored settings are used.
    cluster_count:
        Explicit number of market regimes. When ``None`` the ensemble size
        determines the count.
    """
    # Ensure ``data_full`` is a NumPy array for shape checks
    if isinstance(data_full, list):
        data_full = np.array(data_full)

    cols = np.asarray(data_full).shape[1]
    if cols < 5 or cols > 6:
        raise ValueError("robust_backtest expects raw OHLCV rows")

    start_date = int(data_full[0][0]) if len(data_full) else 0
    end_date = int(data_full[-1][0]) if len(data_full) else 0

    # Log incoming feature dimension for debugging
    print(f"[BACKTEST] Input feature dimension: {data_full.shape[1]}")
    from .constants import FEATURE_DIMENSION

    if data_full.shape[1] != FEATURE_DIMENSION:
        # silently handled inside compute_indicators
        pass

    prepare_backtest_data(data_full)

    base_hp = indicator_hp or getattr(
        ensemble, "indicator_hparams", IndicatorHyperparams()
    )
    hp_dyn = (
        base_hp if not dynamic_indicators else IndicatorHyperparams(**vars(base_hp))
    )

    if dynamic_indicators:
        windows_list = []
        mask = None
        for i in range(23, len(data_full)):
            window = np.asarray(data_full[i - 23 : i + 1], dtype=float)
            update_indicator_toggles(window, hp_dyn)
            indic = compute_indicators(
                window,
                hp_dyn,
                with_scaled=True,
                use_ichimoku=getattr(hp_dyn, "use_ichimoku", False),
            )
            mask = indic["mask"] if mask is None else mask
            windows_list.append(indic["scaled"])
        extd = np.stack(windows_list, axis=0)
        validate_features(extd[0], enabled_mask=mask)
    else:
        indic = compute_indicators(
            data_full,
            hp_dyn,
            with_scaled=True,
            use_ichimoku=getattr(hp_dyn, "use_ichimoku", False),
        )
        extd = indic["scaled"]
        mask = indic["mask"]
        assert extd.shape[1] == mask.size
        validate_features(extd, enabled_mask=mask)
    if len(data_full) < 24:
        return {
            "net_pct": 0.0,
            "trades": 0,
            "effective_net_pct": 0.0,
            "equity_curve": [],
        }

    LEVERAGE = 2  # reduce draw-down pressure
    commission_rate = 0.0006  # 0.06 % taker fee (Phemex) ✓
    FUNDING_RATE = 0.0001
    device = ensemble.device

    hp = base_hp

    # (5) If meta-agent is adjusting threshold, store it in ensemble or define a separate variable.
    # For a simpler demonstration, we keep using GLOBAL_THRESHOLD, but you could do:
    # threshold = ensemble.dynamic_threshold if ensemble.dynamic_threshold is not None else GLOBAL_THRESHOLD
    # or pass it in the function signature.

    # ------------------------------------------------------------------
    # Composite Reward Weights
    # ------------------------------------------------------------------
    # ``alpha`` and ``gamma`` are legacy weights for optional net-profit and
    # draw-down terms.  The main reward now uses risk metrics weighted by
    # ``G.beta`` (Sharpe), ``G.theta`` (Sortino), ``G.phi`` (Omega) and
    # ``G.chi`` (Calmar) in line with recent research on risk-adjusted
    # reinforcement learning.
    alpha = 2.0
    gamma = 4.0
    delta = 0.1

    def _price_with_noise(side: str, price: float) -> float:
        return submit_order(lambda **kw: kw["price"], side, 0.0, price, delay=0.0)

    raw_data = np.array(data_full, dtype=np.float64)
    if raw_data[:, 0].max() > 1_000_000_000_000:
        raw_data[:, 0] //= 1000

    timestamps = raw_data[:, 0]

    # --------------------------------------------------------------------- #
    # Fast vectorised regime labelling – fits K-Means **once** instead of    #
    # refitting for every bar.  Cuts O(n²) down to O(n).                     #
    # --------------------------------------------------------------------- #
    prices = raw_data[:, 4]  # close column
    num_models = len(getattr(ensemble, "models", []))
    n_clust = cluster_count or (num_models if num_models > 1 else 1)
    try:
        from artibot.regime import classify_market_regime_batch

        regime_labels, regime_probs = classify_market_regime_batch(
            prices, n_clusters=n_clust
        )
        regime_probs = np.asarray(regime_probs)
        regimes = (
            regime_labels.tolist()
            if isinstance(regime_labels, np.ndarray)
            else regime_labels
        )
    except Exception:
        regimes = [0] * len(prices)
        regime_probs = np.zeros((len(prices), n_clust))

    transitions: list[int] = []
    if regimes:
        prev_r = regimes[0]
        for idx, r in enumerate(regimes[1:], start=1):
            if r != prev_r:
                transitions.append(int(timestamps[idx]))
                prev_r = r

    from numpy.lib.stride_tricks import sliding_window_view

    windows = (
        extd
        if dynamic_indicators
        else sliding_window_view(extd, (24, extd.shape[1])).squeeze()
    )
    log = logging.getLogger(__name__)
    log.info(
        "[TRACE] backtest window shape=%s (rule-engine expects 6 core bars)",
        windows.shape,
    )
    windows = np.clip(windows, -50.0, 50.0)
    windows = np.nan_to_num(windows, nan=0.0, posinf=0.0, neginf=0.0)
    assert (
        windows.shape[2] == mask.size
    ), f"Expected {mask.size} features, got {windows.shape[2]}"
    windows_t = torch.tensor(windows, dtype=torch.float32, device=device)
    regime_subset = regimes[-windows.shape[0] :]
    probs_subset = regime_probs[-windows.shape[0] :]
    pred_indices, _, avg_params = ensemble.vectorized_predict(
        windows_t,
        batch_size=512,
        regime_labels=regime_subset,
        regime_probs=probs_subset,
    )
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
    net_positions: list[float] = []
    current_position = 0.0
    pos = {
        "size": 0.0,
        "side": None,
        "entry_price": 0.0,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "entry_time": None,
        "regime": None,
    }

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
                if pos["side"] == "long":
                    exit_price = _price_with_noise("sell", exit_price)
                    proceeds = pos["size"] * exit_price
                    comm_exit = proceeds * commission_rate
                    profit = proceeds - (pos["size"] * pos["entry_price"]) - comm_exit
                    new_bal = bal + profit
                    if new_bal < 0:
                        log.warning(
                            "[BACKTEST] equity negative after long exit; clamping to zero (bal=%.2f, profit=%.2f)",
                            bal,
                            profit,
                        )
                    bal = max(0.0, new_bal)
                else:
                    exit_price = _price_with_noise("buy", exit_price)
                    comm_exit = abs(pos["size"]) * exit_price * commission_rate
                    profit = (
                        abs(pos["size"]) * (pos["entry_price"] - exit_price) - comm_exit
                    )
                    hrs = (cur_t - pos["entry_time"]) / 3600.0
                    funding = abs(pos["size"]) * pos["entry_price"] * FUNDING_RATE * hrs
                    profit -= funding
                    new_bal = bal + profit
                    if new_bal < 0:
                        log.warning(
                            "[BACKTEST] equity negative after short exit; clamping to zero (bal=%.2f, profit=%.2f)",
                            bal,
                            profit,
                        )
                    bal = max(0.0, new_bal)
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
                        "regime": pos.get("regime"),
                    }
                )
                pos = {
                    "size": 0.0,
                    "side": None,
                    "entry_price": 0.0,
                    "stop_loss": 0.0,
                    "take_profit": 0.0,
                    "entry_time": None,
                    "regime": None,
                }
        if pos["size"] == 0 and pred in (0, 1):
            atr_val = atr_i if not np.isnan(atr_i) else 1.0
            atr_val = max(1.0, atr_val)
            fill_p = _price_with_noise("buy" if pred == 0 else "sell", cur_p)
            st_dist = sl_m * atr_val
            tp_val = fill_p + tp_m * atr_val if pred == 0 else fill_p - tp_m * atr_val
            pos_sz = risk.position_size(bal, rf, st_dist, fill_p, LEVERAGE)
            if pos_sz <= 0:
                pos_sz = 1.0
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
                        "regime": regimes[i],
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
                        "regime": regimes[i],
                    }
                )
        curr_eq = bal
        if pos["size"] != 0:
            if pos["side"] == "long":
                curr_eq += pos["size"] * (cur_p - pos["entry_price"])
            else:
                curr_eq += abs(pos["size"]) * (pos["entry_price"] - cur_p)
        eq_curve.append((cur_t, curr_eq))
        current_position = pos["size"]
        net_positions.append(current_position)

    if pos["size"] != 0:
        final_price = closes[-1]
        if pos["side"] == "long":
            exit_price = _price_with_noise("sell", final_price)
            comm_exit = pos["size"] * exit_price * commission_rate
            pf = pos["size"] * (exit_price - pos["entry_price"]) - comm_exit
            new_bal = bal + pf
            if new_bal < 0:
                log.warning(
                    "[BACKTEST] equity negative on final long exit; clamping to zero (bal=%.2f, profit=%.2f)",
                    bal,
                    pf,
                )
            bal = max(0.0, new_bal)
        else:
            exit_price = _price_with_noise("buy", final_price)
            comm_exit = abs(pos["size"]) * exit_price * commission_rate
            pf = abs(pos["size"]) * (pos["entry_price"] - exit_price) - comm_exit
            hrs = (timestamps[-1] - pos["entry_time"]) / 3600.0
            fund = abs(pos["size"]) * pos["entry_price"] * FUNDING_RATE * hrs
            pf -= fund
            new_bal = bal + pf
            if new_bal < 0:
                log.warning(
                    "[BACKTEST] equity negative on final short exit; clamping to zero (bal=%.2f, profit=%.2f)",
                    bal,
                    pf,
                )
            bal = max(0.0, new_bal)
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
                "regime": pos.get("regime"),
            }
        )
        eq_curve[-1] = (int(timestamps[-1]), bal)
        current_position = 0.0
        net_positions.append(current_position)

    final_profit = bal - init_bal
    net_pct = (final_profit / init_bal) * 100.0
    if not np.isfinite(net_pct):
        net_pct = 0.0
    # Clamp extreme values so a runaway trade does not dominate training
    eff_net_pct = float(np.clip(net_pct, -1000.0, 1000.0))

    exposure_stats = summarise_net_positions(net_positions)

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
        sortino = torch.tensor(0.0)
        omega = torch.tensor(0.0)
        calmar = 0.0
    else:
        dret = eqdf["balance"].pct_change().dropna()
        mu = dret.mean()
        sigma = dret.std()
        sharpe = (mu * 252) / (sigma * np.sqrt(252)) if sigma > 1e-12 else 0.0
        rollmax = eqdf["balance"].cummax()
        dd = (eqdf["balance"] - rollmax) / rollmax
        mdd = dd.min()
        dret_t = torch.tensor(dret.values, dtype=torch.float32)
        sortino = sortino_ratio(dret_t)
        omega = omega_ratio(dret_t)
        period_days = max(1, int((end_date - start_date) / 86400))
        calmar = calmar_ratio(net_pct, float(mdd), period_days)

    # Calculate performance broken down by regime using trade history
    cluster_perf: dict[int, dict] = {}
    if trades:
        for reg in {t.get("regime") for t in trades if t.get("regime") is not None}:
            reg_trades = [t for t in trades if t.get("regime") == reg]
            if not reg_trades:
                continue
            rets = np.array([t.get("return", 0.0) for t in reg_trades], dtype=float)
            net_pct_reg = (np.prod(rets + 1.0) - 1.0) * 100.0
            avg_ret = float(rets.mean())
            vol_ret = float(rets.std())
            sharpe_reg = (
                (avg_ret / (vol_ret + 1e-9)) * (len(rets) ** 0.5)
                if len(rets) > 1
                else avg_ret / (vol_ret + 1e-9)
            )
            cluster_perf[int(reg)] = {
                "net_pct": net_pct_reg,
                "sharpe": sharpe_reg,
                "trades": len(reg_trades),
            }

    metrics = compute_trade_metrics(trades)
    for reg, stats in cluster_perf.items():
        logging.info(
            f"REGIME_{reg}_PERF net_pct={stats['net_pct']:.2f}%, Sharpe={stats['sharpe']:.2f}, trades={stats.get('trades', 'N/A')}"
        )

    if (cluster_count or len(cluster_perf)) and len(cluster_perf) > 1:
        best_sharpe = (
            max(v["sharpe"] for v in cluster_perf.values()) if cluster_perf else sharpe
        )
        best_net = (
            max(v["net_pct"] for v in cluster_perf.values())
            if cluster_perf
            else net_pct
        )
        if sharpe < best_sharpe or net_pct < best_net:
            logging.warning(
                "PERFORMANCE WARNING: Multi-regime strategy underperforms a single regime. Regime switching may have degraded performance."
            )
    win_rate = metrics["win_rate"]
    profit_factor = metrics["profit_factor"]
    avg_duration = metrics["avg_duration"]
    avg_win = metrics["avg_win"]
    avg_loss = metrics["avg_loss"]

    # Example re-scaling to emphasise positive performance
    sharpe_score = float(np.tanh(sharpe / 2.0))
    sortino_score = float(np.tanh(sortino.item() / 2.0))
    omega_score = float(np.tanh((omega.item() - 1.0) / 2.0))
    calmar_score = float(np.tanh((calmar - 1.0) / 2.0))

    net_score = net_pct
    trade_count = len(trades)
    trade_term = trade_count * delta
    # ------------------------------------------------------------------
    # Composite reward based solely on risk-adjusted metrics.  Recent
    # literature recommends combining Sharpe, Sortino, Omega and Calmar
    # ratios to capture complementary aspects of downside risk
    # (see raw.githubusercontent.com for references).  Net profit and
    # draw-down penalties are disabled by default.  Each ratio is
    # transformed with ``tanh`` and clipped to [-1, 1] so no single
    # metric dominates the reward signal.
    # ------------------------------------------------------------------
    composite_reward = 0.0
    if G.use_net_term:
        composite_reward += alpha * net_score
    if G.use_sharpe_term:
        composite_reward += G.beta * sharpe_score
    if G.use_drawdown_term and not G.use_calmar_term:
        # Penalise large draw-downs exponentially beyond 10%.
        dd_pen = np.exp(max(abs(mdd) - 0.10, 0) * 10) - 1
        composite_reward -= gamma * dd_pen
    if G.use_sortino_term:
        composite_reward += G.theta * sortino_score
    if G.use_omega_term:
        composite_reward += G.phi * omega_score
    if G.use_calmar_term:
        composite_reward += G.chi * calmar_score
    if G.use_trade_term:
        composite_reward += trade_term * 3
    if G.use_profit_days_term:
        composite_reward += (days_in_pf / 365) * 10
    composite_reward -= tot_inact_pen
    composite_reward = float(np.clip(composite_reward, -2.0, 2.0))
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
        "sortino": float(sortino),
        "omega": float(omega),
        "calmar": float(calmar),
        "exposure": exposure_stats,
        "cluster_performance": cluster_perf,
        "regime_transitions": transitions,
        # flag promoting results from a complete dataset span
        # (1337 days or more qualifies as a full data run)
        "full_data_run": (end_date - start_date) >= 1337 * 86400,
    }


if __name__ == "__main__":
    from .utils.torch_threads import set_threads
    from .dataset import load_csv_hourly, HourlyDataset
    from .ensemble import EnsembleModel
    from .training import csv_training_thread
    from artibot.core.device import get_device
    from .hyperparams import IndicatorHyperparams
    from .bot_app import CONFIG
    import threading

    set_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
    ensure_dependencies()

    data = load_csv_hourly("Gemini_BTCUSD_1h.csv", cfg=CONFIG)
    ds_tmp = HourlyDataset(
        data,
        seq_len=24,
        indicator_hparams=IndicatorHyperparams(),
        atr_threshold_k=1.5,
        train_mode=False,
    )
    n_features = ds_tmp[0][0].shape[1]
    ens = EnsembleModel(
        device=get_device(),
        n_models=1,
        n_features=n_features,
        warmup_steps=WARMUP_STEPS,
    )
    stop = threading.Event()
    csv_training_thread(
        ens,
        data,
        stop,
        CONFIG,
        use_prev_weights=False,
        max_epochs=1,
    )
    result = robust_backtest(ens, data, indicator_hp=ens.indicator_hparams)
    if result.get("trades", 0) == 0:
        logging.info("IGNORED_EMPTY_BACKTEST: 0 trades in result")
    else:
        G.push_backtest_metrics(result)
    print(result)
