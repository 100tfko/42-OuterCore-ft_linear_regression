# Linear Regression — Agent Guide

Single-feature linear regression model: predicting car prices from mileage

## Setup

```bash
./setup.sh                 # creates .venv/, installs deps
source .venv/bin/activate
```

## Entrypoints

| Script | Purpose |
|---|---|---|
| `setup.py` | Shared constants: dataset paths, hyperparameters |
| `tools.py` | Shared helpers: load_csv, stats, normalization, gradient descent |
| `describe.py` | Statistical summary of mileage and price |
| `scatter_plot.py` | Scatter plot of mileage vs price |
| `train.py` | Train linear regression, save θ0 and θ1 |
| `predict.py` | Load weights, predict price for given mileage |

- Run any script directly: `python describe.py`
- Predict: `python predict.py` → prompts for mileage

## Key conventions

- **Pandas for CSV read only** — all stats/math from scratch (no sklearn/numpy for core logic).
- **Constants**: dataset paths, hyperparameters (learning rate, iterations) live in `setup.py`.
- **Features**: 1 feature (mileage) + 1 bias = 2 weights total.
- **Normalization**: z-score. train: compute μ/σ. predict: apply saved μ/σ. std=0 → return 0.
- **Sigmoid**: not needed (linear regression). Use direct linear hypothesis: h(x) = θ0 + θ1*x
- **Loss**: Mean Squared Error (MSE) for regression.
- **Weights file**: JSON with θ0, θ1, mean, std.
- **Sample std**: divide by n-1.

## Build status

### ✅ Completed

- **setup.py** — shared constants: dataset path, hyperparameters
- **tools.py** — all helpers:
  - `load_csv`, `count`, `mean`, `std` (sample, n-1), `min`, `max`
  - `normalize` (z-score, train/inference, std=0 edge case)
  - `gradient_descent` (batch GD for linear regression)
  - MSE loss function
  - All functions have docstrings

### ❌ Not yet built

- describe.py, scatter_plot.py, train.py, predict.py

## Next up: describe.py

## Hyperparameters (train.py)

- Learning rate α=0.01, iterations=1000, initial theta=[0, 0].
- Loss (MSE) must decrease each iteration; oscillate → reduce α; still dropping at end → increase iterations.

## Output

- `model.json` (from train): contains θ0, θ1, mileage_mean, mileage_std
- Predictions: stdout (price for given mileage)

## Teaching preferences

- Never give direct code answers — explain purpose, point out mistakes, indicate correct syntax, let me write the code
