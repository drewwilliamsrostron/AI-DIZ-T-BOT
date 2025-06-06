"""Utility functions for computing performance metrics."""

from .globals import *

###############################################################################
# Yearly Stats (unchanged)
###############################################################################
def compute_yearly_stats(equity_curve, trades, initial_balance=100.0):
    if not equity_curve:
        return pd.DataFrame(), "No data"
    eq_df = pd.DataFrame(equity_curve, columns=["timestamp","balance"])
    eq_df["dt"] = pd.to_datetime(eq_df["timestamp"], unit="s")
    eq_df.set_index("dt", inplace=True)
    eq_df = eq_df.resample("1D").last().dropna()
    if eq_df.empty:
        return pd.DataFrame(), "No data"
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["entry_dt"] = pd.to_datetime(trades_df["entry_time"], unit="s")
        trades_df.set_index("entry_dt", inplace=True)
    year_rows = []
    grouped = eq_df.groupby(eq_df.index.year)
    for year, grp in grouped:
        if len(grp) < 2:
            continue
        start_balance = grp["balance"].iloc[0]
        end_balance = grp["balance"].iloc[-1]
        net_pct_year = 100.0 * (end_balance - start_balance) / (start_balance if start_balance!=0 else 1e-8)
        daily_returns = grp["balance"].pct_change().dropna()
        mu = daily_returns.mean()
        sigma = daily_returns.std()
        sharpe_year = (mu*252)/(sigma*np.sqrt(252)) if sigma>1e-12 else 0.0
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
        year_rows.append({
            "Year": year,
            "NetPct": net_pct_year,
            "Sharpe": sharpe_year,
            "MaxDD": mdd_year,
            "Trades": num_trades_year
        })
    if not year_rows:
        return pd.DataFrame(), "No yearly data"
    dfy = pd.DataFrame(year_rows).set_index("Year")
    table_str = dfy.to_string(float_format=lambda x:f"{x: .2f}")
    return dfy, table_str

###############################################################################
# inactivity_exponential_penalty (unchanged)
###############################################################################
def inactivity_exponential_penalty(gap_in_secs, max_penalty=100.0):
    month_secs = 30*24*3600
    months = int(gap_in_secs//month_secs)
    base = 0.01
    penalty = 0.0
    for i in range(months):
        penalty += base*(2**i)
        if penalty>max_penalty:
            penalty = max_penalty
            break
    return penalty

###############################################################################
# compute_days_in_profit (unchanged)
###############################################################################
def compute_days_in_profit(equity_curve, init_balance):
    total_s = 0.0
    if len(equity_curve)<2:
        return 0.0
    for i in range(1,len(equity_curve)):
        t_prev, b_prev = equity_curve[i-1]
        t_curr, b_curr = equity_curve[i]
        dt = t_curr - t_prev
        if b_prev>=init_balance and b_curr>=init_balance:
            total_s+= dt
        elif b_prev<init_balance and b_curr<init_balance:
            pass
        else:
            if b_curr!=b_prev:
                f = (init_balance - b_prev)/(b_curr - b_prev)
                if b_prev<init_balance and b_curr>=init_balance:
                    total_s += (1 - f)*dt
                else:
                    total_s += f*dt
    return total_s/86400.0

