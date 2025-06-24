min_sharpe = -1.0
max_drawdown = -0.5


def is_viable(strat):
    return (
        strat.get("sharpe", 0) > -2.0
        and strat.get("max_dd", 0) > -0.7
        and strat.get("profit_factor", 0) > 0.8
    )
