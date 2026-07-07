import csv
import numpy as np
import matplotlib.pyplot as plt

SEED_RESULTS_PATH = "results/seed_stability_results.csv"
BASELINE_RESULTS_PATH = "results/baseline_results.csv"
OUTPUT_PATH = "results/seed_comparison.png"

METRIC = "total_return"  # change to "sharpe_ratio", "max_drawdown", etc. to plot other metrics


def load_seed_results():
    names, means, stds = [], [], []
    with open(SEED_RESULTS_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["experiment"])
            means.append(float(row[f"{METRIC}_mean"]))
            stds.append(float(row[f"{METRIC}_std"]))
    return names, means, stds


def load_baseline(name, metric):
    with open(BASELINE_RESULTS_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["baseline"] == name:
                # buy_and_hold has a plain column; random_agent has a _mean column
                if metric in row and row[metric] not in ("", None):
                    return float(row[metric]), 0.0
                mean_key, std_key = f"{metric}_mean", f"{metric}_std"
                if mean_key in row:
                    return float(row[mean_key]), float(row.get(std_key, 0.0) or 0.0)
    return None, None


def main():
    names, means, stds = load_seed_results()
    bh_mean, bh_std = load_baseline("buy_and_hold", METRIC)
    rand_mean, rand_std = load_baseline("random_agent", METRIC)

    all_names = names + ["buy_and_hold", "random_agent"]
    all_means = means + [bh_mean, rand_mean]
    all_stds = stds + [bh_std, rand_std]

    colors = ["#4C72B0"] * len(names) + ["#55A868", "#C44E52"]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(all_names, all_means, yerr=all_stds, capsize=5, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel(METRIC)
    plt.title(f"{METRIC} across experiments (mean ± std over {3} seeds) vs baselines")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()