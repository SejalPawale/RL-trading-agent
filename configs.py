BASE_CONFIG = {
    "train_csv": "data/eurusd_train.csv",
    "test_csv": "data/eurusd_test.csv",
    "initial_balance": 10_000.0,
    "window_size": 30,
    "lot_size": 10_000,
    "transaction_cost": 0.00005,
    "slippage": 0.00002,
    "stop_loss_options": [0.0010, 0.0015, 0.0020],
    "take_profit_options": [0.0010, 0.0015, 0.0020],
    "default_indicators": ["rsi", "ma20", "ma50", "atr", "ma20_slope"],
    "timesteps": 20_000,
    "reward_mode": "pnl"
}

EXPERIMENTS = [
    {
        "name": "exp_base_10k_pnl",
        "indicators": ["rsi", "ma20", "ma50", "atr", "ma20_slope"],
        "reward_mode": "pnl",
        "timesteps": 10_000,
        "window_size": 30
    },
    {
        "name": "exp_base_30k_pnl",
        "indicators": ["rsi", "ma20", "ma50", "atr", "ma20_slope"],
        "reward_mode": "pnl",
        "timesteps": 30_000,
        "window_size": 30
    },
    {
        "name": "exp_macd_bb_30k_pnl",
        "indicators": ["rsi", "ma20", "ma50", "atr", "ma20_slope", "macd", "bb_width"],
        "reward_mode": "pnl",
        "timesteps": 30_000,
        "window_size": 30
    },
    {
        "name": "exp_macd_bb_30k_risk_adj",
        "indicators": ["rsi", "ma20", "ma50", "atr", "ma20_slope", "macd", "bb_width"],
        "reward_mode": "risk_adjusted",
        "timesteps": 30_000,
        "window_size": 30
    },
    {
        "name": "exp_macd_bb_50k_drawdown_penalty",
        "indicators": ["rsi", "ma20", "ma50", "atr", "ma20_slope", "macd", "bb_width"],
        "reward_mode": "drawdown_penalty",
        "timesteps": 50_000,
        "window_size": 30
    }
]