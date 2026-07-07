# experiment_runner.py

import os
import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from configs import BASE_CONFIG, EXPERIMENTS
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnvironment
from metrics import summarize_performance

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/equity_curves", exist_ok=True)

# Now covers all 5 experiments, not just the top 2.
SEED_STABILITY_TARGETS = [exp["name"] for exp in EXPERIMENTS]
SEEDS = [0, 1, 2]


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


def run_single_experiment(exp, seed=None, save_curve=True):
    train_df, train_features = load_and_preprocess_data(BASE_CONFIG["train_csv"], exp["indicators"])
    test_df, test_features = load_and_preprocess_data(BASE_CONFIG["test_csv"], exp["indicators"])

    train_env = DummyVecEnv([lambda: make_env(train_df, train_features, exp)])
    model = PPO("MlpPolicy", train_env, verbose=0, seed=seed)
    model.learn(total_timesteps=exp["timesteps"])

    model_name = exp["name"] if seed is None else f"{exp['name']}_seed{seed}"
    model.save(f"models/{model_name}")

    test_env = make_env(test_df, test_features, exp)
    obs, _ = test_env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

    metrics = summarize_performance(test_env.equity_curve, test_env.trade_pnls)
    metrics["time_in_market"] = test_env.get_time_in_market()

    if save_curve:
        np.save(f"results/equity_curves/{model_name}.npy", np.array(test_env.equity_curve))

    row = {
        "experiment": model_name,
        "indicators": ",".join(exp["indicators"]),
        "reward_mode": exp["reward_mode"],
        "timesteps": exp["timesteps"],
        "seed": seed if seed is not None else "default",
        **metrics
    }
    return row


def run_main_experiments():
    rows = []
    for exp in EXPERIMENTS:
        print(f"Running {exp['name']} ...")
        row = run_single_experiment(exp)
        rows.append(row)
        print(row)

    output_csv = "results/experiment_results.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {output_csv}")


def run_seed_stability():
    exp_lookup = {exp["name"]: exp for exp in EXPERIMENTS}
    rows = []

    for name in SEED_STABILITY_TARGETS:
        exp = exp_lookup[name]
        seed_metrics = []

        for seed in SEEDS:
            print(f"Running {name} with seed {seed} ...")
            row = run_single_experiment(exp, seed=seed, save_curve=(seed == SEEDS[0]))
            seed_metrics.append(row)

        metric_keys = [k for k in seed_metrics[0].keys()
                       if k not in ("experiment", "indicators", "reward_mode", "timesteps", "seed")]

        summary_row = {
            "experiment": name,
            "indicators": ",".join(exp["indicators"]),
            "reward_mode": exp["reward_mode"],
            "timesteps": exp["timesteps"],
            "num_seeds": len(SEEDS)
        }
        for key in metric_keys:
            values = [m[key] for m in seed_metrics]
            summary_row[f"{key}_mean"] = float(np.mean(values))
            summary_row[f"{key}_std"] = float(np.std(values))

        rows.append(summary_row)
        print(summary_row)

    output_csv = "results/seed_stability_results.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved seed stability results to {output_csv}")


def main():
    run_main_experiments()
    run_seed_stability()


if __name__ == "__main__":
    main()