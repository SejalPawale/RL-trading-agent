# walk_forward.py

import os
import csv
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from configs import BASE_CONFIG, EXPERIMENTS
from indicators import load_raw, add_indicators
from trading_env import ForexTradingEnvironment
from metrics import summarize_performance, buy_and_hold_equity_curve

os.makedirs("results/walk_forward", exist_ok=True)

# Keep this to 1-2 configs — each one trains N_BLOCKS-1 separate models.
FOLD_EXPERIMENTS = ["exp_base_30k_pnl", "exp_macd_bb_50k_drawdown_penalty"]
N_BLOCKS = 5  # yields N_BLOCKS - 1 folds, each with a bigger training set than the last


def build_combined_raw():
    train_raw = load_raw(BASE_CONFIG["train_csv"])
    test_raw = load_raw(BASE_CONFIG["test_csv"])
    combined = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)
    return combined


def make_folds(df, n_blocks=N_BLOCKS):
    block_size = len(df) // n_blocks
    folds = []
    for i in range(1, n_blocks):
        train_end = i * block_size
        test_end = (i + 1) * block_size if i + 1 < n_blocks else len(df)
        train_slice = df.iloc[:train_end].reset_index(drop=True)
        test_slice = df.iloc[train_end:test_end].reset_index(drop=True)
        folds.append((train_slice, test_slice))
    return folds


def make_env(df, feature_cols, exp):
    return ForexTradingEnvironment(
        df=df,
        feature_cols=feature_cols,
        window_size=exp.get("window_size", BASE_CONFIG["window_size"]),
        initial_balance=BASE_CONFIG["initial_balance"],
        lot_size=BASE_CONFIG["lot_size"],
        transaction_cost=BASE_CONFIG["transaction_cost"],
        slippage=BASE_CONFIG["slippage"],
        stop_loss_options=BASE_CONFIG["stop_loss_options"],
        take_profit_options=BASE_CONFIG["take_profit_options"],
        reward_mode=exp.get("reward_mode", BASE_CONFIG["reward_mode"])
    )


def run_fold(exp, train_raw, test_raw, fold_idx):
    train_df, train_features = add_indicators(train_raw, exp["indicators"])
    test_df, test_features = add_indicators(test_raw, exp["indicators"])

    window_size = exp.get("window_size", BASE_CONFIG["window_size"])
    if len(train_df) <= window_size or len(test_df) <= window_size:
        print(f"Skipping fold {fold_idx} for {exp['name']}: not enough rows after indicator warmup")
        return None

    train_env = DummyVecEnv([lambda: make_env(train_df, train_features, exp)])
    model = PPO("MlpPolicy", train_env, verbose=0, seed=0)
    model.learn(total_timesteps=exp["timesteps"])

    test_env = make_env(test_df, test_features, exp)
    obs, _ = test_env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

    metrics = summarize_performance(test_env.equity_curve, test_env.trade_pnls)

    bh_curve = buy_and_hold_equity_curve(test_df["close"].values, BASE_CONFIG["initial_balance"])
    bh_metrics = summarize_performance(bh_curve, trade_pnls=[])

    return {
        "experiment": exp["name"],
        "fold": fold_idx,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "agent_total_return": metrics["total_return"],
        "agent_sharpe": metrics["sharpe_ratio"],
        "agent_max_drawdown": metrics["max_drawdown"],
        "agent_num_trades": metrics["num_trades"],
        "bh_total_return": bh_metrics["total_return"],
        "bh_sharpe": bh_metrics["sharpe_ratio"],
        "bh_max_drawdown": bh_metrics["max_drawdown"],
    }


def main():
    exp_lookup = {exp["name"]: exp for exp in EXPERIMENTS}
    combined_raw = build_combined_raw()
    folds = make_folds(combined_raw)
    print(f"Built {len(folds)} walk-forward folds from {len(combined_raw)} total rows")

    all_rows = []
    for name in FOLD_EXPERIMENTS:
        exp = exp_lookup[name]
        for fold_idx, (train_raw, test_raw) in enumerate(folds):
            print(f"Running {name}, fold {fold_idx} ...")
            row = run_fold(exp, train_raw, test_raw, fold_idx)
            if row is not None:
                all_rows.append(row)
                print(row)

    output_csv = "results/walk_forward/fold_results.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved fold-level results to {output_csv}")

    summary_rows = []
    for name in FOLD_EXPERIMENTS:
        exp_rows = [r for r in all_rows if r["experiment"] == name]
        if not exp_rows:
            continue
        summary = {"experiment": name, "num_folds": len(exp_rows)}
        for key in ["agent_total_return", "agent_sharpe", "agent_max_drawdown",
                    "bh_total_return", "bh_sharpe", "bh_max_drawdown"]:
            values = [r[key] for r in exp_rows]
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
        summary_rows.append(summary)
        print(summary)

    summary_csv = "results/walk_forward/fold_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved fold summary to {summary_csv}")


if __name__ == "__main__":
    main()