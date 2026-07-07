import os
import glob
import numpy as np
import matplotlib.pyplot as plt

CURVE_DIR = "results/equity_curves"
OUTPUT_PATH = "results/equity_curves_comparison.png"


def main():
    curve_files = sorted(glob.glob(os.path.join(CURVE_DIR, "*.npy")))
    if not curve_files:
        print("No equity curves found. Run experiment_runner.py and baseline_runner.py first.")
        return

    plt.figure(figsize=(12, 6))
    for path in curve_files:
        name = os.path.splitext(os.path.basename(path))[0]
        curve = np.load(path)
        style = "--" if name in ("buy_and_hold", "random_agent_example") else "-"
        plt.plot(curve, style, label=name, linewidth=1.5)

    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.title("Equity Curves: Experiments vs Baselines")
    plt.legend(fontsize=8, loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()