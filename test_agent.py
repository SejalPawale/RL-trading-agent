# test_agent.py

from stable_baselines3 import PPO

from configs import BASE_CONFIG
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnvironment
from metrics import summarize_performance

test_df, feature_cols = load_and_preprocess_data(
    BASE_CONFIG["test_csv"],
    BASE_CONFIG["default_indicators"]
)

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

model = PPO.load("models/base_ppo_forex_agent")

obs, info = env.reset()
terminated = False
truncated = False

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

summary = summarize_performance(env.equity_curve, env.trade_pnls)

print("Test summary:")
for k, v in summary.items():
    print(f"{k}: {v}")