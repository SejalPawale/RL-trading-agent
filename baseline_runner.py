# baseline_runner.py

import os
import csv
import numpy as np

from baselines import get_buy_and_hold_metrics, run_random_agent

os.makedirs("results", exist_ok=True)
os.makedirs("results/equity_curves", exist_ok=True)

RANDOM_SEEDS = [0, 1, 2, 3, 4]


def main():
    rows = []

    bh_metrics, bh_curve = get_buy_and_hold_metrics()
    np.save("results/equity_curves/buy_and_hold.npy", np.array(bh_curve))
    rows.append({"baseline": "buy_and_hold", **bh_metrics})
    print("buy_and_hold:", bh_metrics)

    random_runs = []
    for seed in RANDOM_SEEDS:
        metrics, curve = run_random_agent(seed=seed)
        random_runs.append(metrics)
        if seed == RANDOM_SEEDS[0]:
            np.save("results/equity_curves/random_agent_example.npy", np.array(curve))

    metric_keys = random_runs[0].keys()
    random_summary = {"baseline": "random_agent"}
    for key in metric_keys:
        values = [r[key] for r in random_runs]
        random_summary[f"{key}_mean"] = float(np.mean(values))
        random_summary[f"{key}_std"] = float(np.std(values))
    rows.append(random_summary)
    print("random_agent (mean of 5 seeds):", random_summary)

    output_csv = "results/baseline_results.csv"
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved baseline results to {output_csv}")


if __name__ == "__main__":
    main()