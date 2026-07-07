# plot_walk_forward.py

import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = "results/walk_forward/fold_results.csv"
OUTPUT_PATH = "results/walk_forward/fold_comparison.png"


def main():
    data = defaultdict(list)
    with open(INPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["experiment"]].append(row)

    experiments = list(data.keys())
    fig, axes = plt.subplots(len(experiments), 1, figsize=(10, 4 * len(experiments)), squeeze=False)

    for idx, exp_name in enumerate(experiments):
        rows = sorted(data[exp_name], key=lambda r: int(r["fold"]))
        folds = [f"fold {r['fold']}" for r in rows]
        agent_returns = [float(r["agent_total_return"]) for r in rows]
        bh_returns = [float(r["bh_total_return"]) for r in rows]

        x = np.arange(len(folds))
        width = 0.35

        ax = axes[idx][0]
        ax.bar(x - width / 2, agent_returns, width, label="agent", color="#4C72B0")
        ax.bar(x + width / 2, bh_returns, width, label="buy_and_hold", color="#55A868")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(folds)
        ax.set_ylabel("total_return")
        ax.set_title(exp_name)
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()