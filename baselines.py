import numpy as np

from configs import BASE_CONFIG
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnvironment
from metrics import summarize_performance, buy_and_hold_equity_curve


def get_buy_and_hold_metrics():
    test_df, _ = load_and_preprocess_data(BASE_CONFIG["test_csv"], BASE_CONFIG["default_indicators"])
    equity_curve = buy_and_hold_equity_curve(test_df["close"].values, BASE_CONFIG["initial_balance"])
    metrics = summarize_performance(equity_curve, trade_pnls=[])
    metrics["time_in_market"] = 1.0  # always holding
    return metrics, equity_curve


def run_random_agent(seed=None):
    test_df, feature_cols = load_and_preprocess_data(BASE_CONFIG["test_csv"], BASE_CONFIG["default_indicators"])

    env = ForexTradingEnvironment(
        df=test_df,
        feature_cols=feature_cols,
        window_size=BASE_CONFIG["window_size"],
        initial_balance=BASE_CONFIG["initial_balance"],
        lot_size=BASE_CONFIG["lot_size"],
        transaction_cost=BASE_CONFIG["transaction_cost"],
        slippage=BASE_CONFIG["slippage"],
        stop_loss_options=BASE_CONFIG["stop_loss_options"],
        take_profit_options=BASE_CONFIG["take_profit_options"],
        reward_mode=BASE_CONFIG["reward_mode"]
    )

    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        action = int(rng.integers(0, env.action_space.n))
        obs, reward, terminated, truncated, info = env.step(action)

    metrics = summarize_performance(env.equity_curve, env.trade_pnls)
    metrics["time_in_market"] = env.get_time_in_market()
    return metrics, env.equity_curve