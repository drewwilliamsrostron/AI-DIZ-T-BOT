"""Utility functions for computing performance metrics."""

import numpy as np
import pandas as pd

DAY_SEC = 24 * 60 * 60

# ruff: noqa: F403, F405


###############################################################################
# Yearly Stats (unchanged)
###############################################################################
def compute_yearly_stats(equity_curve, trades, initial_balance=100.0):
    """Aggregate yearly return statistics from an equity curve."""
    if not equity_curve:
        return pd.DataFrame(), "No data"
    eq_df = pd.DataFrame(equity_curve, columns=["timestamp", "balance"])
    if eq_df["timestamp"].max() > 1_000_000_000_000:
        eq_df["timestamp"] //= 1000
    eq_df["dt"] = pd.to_datetime(eq_df["timestamp"], unit="s")
    eq_df.set_index("dt", inplace=True)
    eq_df = eq_df.resample("1D").last().dropna()
    if eq_df.empty:
        return pd.DataFrame(), "No data"
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        if trades_df["entry_time"].max() > 1_000_000_000_000:
            trades_df["entry_time"] //= 1000
        trades_df["entry_dt"] = pd.to_datetime(trades_df["entry_time"], unit="s")
        trades_df.set_index("entry_dt", inplace=True)
    year_rows = []
    grouped = eq_df.groupby(eq_df.index.year)
    for year, grp in grouped:
        if len(grp) < 2:
            continue
        start_balance = grp["balance"].iloc[0]
        end_balance = grp["balance"].iloc[-1]
        net_pct_year = (
            100.0
            * (end_balance - start_balance)
            / (start_balance if start_balance != 0 else 1e-8)
        )
        daily_returns = grp["balance"].pct_change().dropna()
        mu = daily_returns.mean()
        sigma = daily_returns.std()
        sharpe_year = (mu * 252) / (sigma * np.sqrt(252)) if sigma > 1e-12 else 0.0
        rolling_max = grp["balance"].cummax()
        dd = (grp["balance"] - rolling_max) / rolling_max
        mdd_year = dd.min() if not dd.empty else 0.0

        if not trades_df.empty:
            try:
                trades_in_year = trades_df.loc[str(year)]
                num_trades_year = len(trades_in_year) if not trades_in_year.empty else 0
            except KeyError:
                num_trades_year = 0
        else:
            num_trades_year = 0
        year_rows.append(
            {
                "Year": year,
                "NetPct": net_pct_year,
                "Sharpe": sharpe_year,
                "MaxDD": mdd_year,
                "Trades": num_trades_year,
            }
        )
    if not year_rows:
        return pd.DataFrame(), "No yearly data"
    dfy = pd.DataFrame(year_rows).set_index("Year")
    table_str = dfy.to_string(float_format=lambda x: f"{x: .2f}")
    return dfy, table_str


###############################################################################
# inactivity_exponential_penalty (unchanged)
###############################################################################
def inactivity_exponential_penalty(gap_in_secs, max_penalty=100.0):
    """Return penalty that grows with months of inactivity."""
    month_secs = 30 * 24 * 3600
    months = int(gap_in_secs // month_secs)
    base = 0.01
    penalty = 0.0
    for i in range(months):
        penalty += base * (2**i)
    return penalty


###############################################################################
# compute_days_in_profit (unchanged)
###############################################################################
def compute_days_in_profit(
    equity_curve,
    init_balance,
    *,
    day_seconds: float = DAY_SEC,
):
    """Return days the equity stays above ``init_balance``.

    Parameters
    ----------
    equity_curve : sequence[tuple[float, float]]
        ``(timestamp, balance)`` pairs in ascending order.
    init_balance : float
        The starting capital level to compare against.
    day_seconds : float, optional
        Number of seconds that constitute one day.  Defaults to ``DAY_SEC``
        (24 hours).  Override for non-24 hour markets.

    The previous implementation used a Python loop over ``equity_curve``.
    This version converts the input to :class:`numpy.ndarray` and relies on
    vectorised operations for the intersection logic.  It provides identical
    results while avoiding Python-level iteration.
    """

    if len(equity_curve) < 2:
        return 0.0

    arr = np.asarray(equity_curve, dtype=float)
    t = arr[:, 0]
    b = arr[:, 1]

    above = b >= init_balance
    dt = np.diff(t)

    full = dt[above[:-1] & above[1:]]

    cross_up = (~above[:-1]) & above[1:]
    cross_down = above[:-1] & (~above[1:])
    slope = b[1:] - b[:-1]
    frac = np.zeros_like(slope)
    nz = slope != 0
    frac[nz] = (init_balance - b[:-1][nz]) / slope[nz]
    frac = np.clip(frac, 0.0, 1.0)

    partial_up = dt[cross_up] * (1 - frac[cross_up])
    partial_down = dt[cross_down] * frac[cross_down]

    total_s = np.sum(full) + np.sum(partial_up) + np.sum(partial_down)
    return float(total_s) / day_seconds


###############################################################################
# compute_trade_metrics
###############################################################################


def compute_trade_metrics(trades):
    """Return win rate, profit factor and average duration for given trades.

    The implementation now leverages :mod:`numpy` arrays for all calculations
    to minimise Python overhead on large trade lists.
    """

    if not trades:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_duration": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    returns = np.fromiter((t.get("return", 0.0) for t in trades), dtype=float)
    durations = np.fromiter((t.get("duration", 0.0) for t in trades), dtype=float)

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = float(wins.size) / float(returns.size)
    profit = wins.sum()
    loss = -losses.sum()

    if loss > 1e-12:
        profit_factor = float(profit / loss)
    elif profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    avg_duration = float(durations.mean()) if durations.size else 0.0
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float((-losses).mean()) if losses.size else 0.0

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_duration": avg_duration,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def nuclear_key_condition(sharpe: float, max_dd: float, profit_factor: float) -> bool:
    """Return ``True`` when metrics allow live trading."""

    return sharpe >= 1.5 and max_dd >= -0.20 and profit_factor >= 1.5
