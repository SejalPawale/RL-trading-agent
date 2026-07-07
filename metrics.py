import numpy as np

def compute_step_returns(equity_curve):
    equity = np.asarray(equity_curve, dtype=np.float64)
    if len(equity) < 2:
        return np.array([], dtype=np.float64)
    prev = equity[:-1]
    curr = equity[1:]
    valid = prev != 0
    returns = np.zeros(len(prev), dtype=np.float64)
    returns[valid] = (curr[valid] - prev[valid]) / prev[valid]
    return returns

def buy_and_hold_equity_curve(close_prices, initial_balance):
    prices = np.asarray(close_prices, dtype=np.float64)
    if len(prices) == 0 or prices[0] == 0:
        return [initial_balance]
    normalized = prices / prices[0]
    return (normalized * initial_balance).tolist()

def total_return(equity_curve):
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return 0.0
    return (equity_curve[-1] / equity_curve[0]) - 1.0

def sharpe_ratio(equity_curve, periods_per_year=252 * 24, risk_free_rate=0.0):
    rets = compute_step_returns(equity_curve)
    if len(rets) == 0:
        return 0.0
    excess = rets - (risk_free_rate / periods_per_year)
    std = np.std(excess)
    if std == 0:
        return 0.0
    return np.sqrt(periods_per_year) * np.mean(excess) / std

def max_drawdown(equity_curve):
    equity = np.asarray(equity_curve, dtype=np.float64)
    if len(equity) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(equity)
    drawdowns = (equity - running_peak) / running_peak
    return float(np.min(drawdowns))

def win_rate(trade_pnls):
    if not trade_pnls:
        return 0.0
    wins = sum(1 for pnl in trade_pnls if pnl > 0)
    return wins / len(trade_pnls)

def avg_trade_pnl(trade_pnls):
    if not trade_pnls:
        return 0.0
    return float(np.mean(trade_pnls))

def profit_factor(trade_pnls):
    if not trade_pnls:
        return 0.0
    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss

def summarize_performance(equity_curve, trade_pnls):
    return {
        "final_equity": float(equity_curve[-1]) if equity_curve else 0.0,
        "total_return": total_return(equity_curve),
        "sharpe_ratio": sharpe_ratio(equity_curve),
        "max_drawdown": max_drawdown(equity_curve),
        "win_rate": win_rate(trade_pnls),
        "avg_trade_pnl": avg_trade_pnl(trade_pnls),
        "profit_factor": profit_factor(trade_pnls),
        "num_trades": len(trade_pnls)
    }