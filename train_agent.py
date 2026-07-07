# train_agent.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from configs import BASE_CONFIG
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnvironment

def make_env():
    train_df, feature_cols = load_and_preprocess_data(
        BASE_CONFIG["train_csv"],
        BASE_CONFIG["default_indicators"]
    )

    return ForexTradingEnvironment(
        df=train_df,
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

env = DummyVecEnv([make_env])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=BASE_CONFIG["timesteps"])
model.save("models/base_ppo_forex_agent")
print("Base model saved to models/base_ppo_forex_agent")