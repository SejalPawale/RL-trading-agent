# RL Trading Agent

A reinforcement learning (PPO) agent trained to trade EUR/USD, evaluated with baseline comparisons, multi-seed variance analysis, and walk-forward validation.

## Project Summary

This project applies reinforcement learning to sequential decision-making on noisy financial time series: a Proximal Policy Optimization (PPO) agent learns to make trading decisions (hold / go long / go short, with configurable stop-loss and take-profit levels) on historical EUR/USD price data, using a custom OpenAI Gymnasium environment. Rather than stopping at a single training run, the project systematically studies how feature choices, reward design, and training duration affect out-of-sample behavior, testing whether the agent's performance is real, reproducible, and better than doing nothing.

## Data Source

The hourly EUR/USD OHLCV data (`data/EURUSD_Candlestick_1_Hour_BID_*.csv`) is sourced from [Dukascopy's Historical Data Feed](https://www.dukascopy.com/swiss/english/marketwatch/historical/).

**Headline finding:** across 5 feature/reward configurations and 3 random seeds each, no configuration reliably outperformed a naive buy-and-hold baseline. Two configurations showed a statistically consistent edge over a random-action agent (evidence the policy learns *something* non-trivial), but this was not enough to overcome transaction costs and market drift. Adding technical indicators (MACD, Bollinger Bands) increased result variance without improving mean performance.

## Why This Result Matters

What this project is actually evaluating, is the methodology needed to find out whether a strategy works: baseline comparison against buy-and-hold and a random agent, seed-variance testing to separate signal from noise, and walk-forward validation to check results hold across different time periods. That process is the core deliverable here, independent of whether the strategy itself turned out to be profitable.

## Environment

`ForexTradingEnvironment` (Gymnasium-compatible):
- **Observation:** a rolling window (default 30 steps) of technical indicator values, plus current position, entry price, balance ratio, and equity ratio.
- **Action space:** `hold / buy / sell`, each combined with a choice of stop-loss and take-profit distance from a configurable set of options.
- **Reward modes:** `pnl` (raw realized profit/loss), `risk_adjusted` (penalizes losing trades), `drawdown_penalty` (penalizes being in a drawdown).
- **Exit mechanism:** positions close only when price hits the stop-loss or take-profit level (no discretionary "close now" action) - a noted limitation, see below.

## Methodology

1. **Baseline experiments** — 5 configurations varying indicator sets, reward shaping, and training length (`configs.py` → `EXPERIMENTS`).
2. **Baseline comparisons** — buy-and-hold and a random-action agent, run on the same test data, to give the RL results a reference point.
3. **Seed-variance testing** — each configuration retrained across 3 random seeds to check whether results are reproducible or artifacts of a lucky/unlucky run.
4. **Walk-forward validation** — the best-performing configurations retested across multiple rolling train/test splits of the full dataset, to check the finding holds across different time periods rather than one fixed split.

## Results

### Total return, mean ± std across seeds

| Configuration | Indicators | Reward Mode | Total Return |
|---|---|---|---|
| Buy-and-hold | — | — | **-1.9%** |
| exp_base_10k_pnl | rsi, ma20, ma50, atr, ma20_slope | pnl | -16.4% ± 5.5% |
| exp_base_30k_pnl | rsi, ma20, ma50, atr, ma20_slope | pnl | -10.5% ± 1.2% |
| exp_macd_bb_30k_pnl | + macd, bb_width | pnl | -18.7% ± 10.8% |
| exp_macd_bb_30k_risk_adj | + macd, bb_width | risk_adjusted | -14.4% ± 14.2% |
| exp_macd_bb_50k_drawdown_penalty | + macd, bb_width | drawdown_penalty | -9.3% ± 6.7% |
| Random agent | — | — | -16.3% ± 6.1% |

See `results/seed_comparison.png` and `results/equity_curves_comparison.png` for the full visual comparison, and `results/walk_forward/fold_comparison.png` for the cross-period validation.

### Reading the results

- **No configuration beat buy-and-hold.** This holds across every seed tested.
- **Two configurations (`base_30k_pnl`, `macd_bb_50k_drawdown_penalty`) consistently beat the random agent** with relatively low variance - evidence the policy learns non-random structure, just not enough to overcome costs and drift.
- **The other three configurations are statistically indistinguishable from random** - their standard deviation is as large as, or larger than, the gap between their mean and random's mean.
- **Adding MACD/Bollinger Band indicators did not reliably help**, and in most cases increased variance.

## Repository Structure

```
configs.py              # Base config + list of 5 experiment configurations
indicators.py           # Loads OHLC data, computes technical indicators
trading_env.py          # Custom Gymnasium environment
train_agent.py          # Trains a single baseline PPO model
test_agent.py           # Evaluates a saved model on the test set
experiment_runner.py    # Runs all 5 experiments + multi-seed stability checks
baselines.py            # Buy-and-hold and random-agent baseline logic
baseline_runner.py      # Runs baseline comparisons, saves results
walk_forward.py         # Rolling train/test split validation
metrics.py              # Performance metrics (Sharpe, drawdown, profit factor, etc.)
plot_results.py         # Equity curve comparison plot
plot_seed_comparison.py # Bar chart of mean ± std per config vs. baselines
plot_walk_forward.py    # Fold-by-fold agent vs. buy-and-hold plot
results/                # CSVs and plots generated by the above scripts
models/                 # Saved PPO models
```

## Running the Project

```bash
pip install -r Requirements.txt

# Full experiment suite: 5 configs + multi-seed stability (20 training runs total)
python experiment_runner.py

# Baseline comparisons (buy-and-hold, random agent)
python baseline_runner.py

# Walk-forward validation across rolling time splits
python walk_forward.py

# Generate plots
python plot_results.py
python plot_seed_comparison.py
python plot_walk_forward.py
```

## Limitations

- **No discretionary exit action.** Positions close only via stop-loss/take-profit; the agent cannot choose to exit early. This structurally limits what any configuration can learn, independent of indicators or reward shaping.
- **Training length is modest** (10k–50k timesteps) relative to typical RL-trading literature (often 500k+); it's unknown whether substantially longer training would close the gap to buy-and-hold.
- **Single instrument, single data source** (EUR/USD). Findings may not generalize to other currency pairs or asset classes.

## What I'd Try Next

- Add an explicit "close position" action to the environment and re-run the full experiment suite.
- Train the best-performing configuration for 300k–500k+ timesteps to test whether the gap to buy-and-hold narrows.
- Compare PPO against a second algorithm (e.g. A2C, DQN) on the same environment.

## Tech Stack

Python, Stable-Baselines3 (PPO), Gymnasium, pandas, NumPy, matplotlib.