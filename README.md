# ft_linear_regression

A linear regression project implementing gradient descent from scratch to predict car prices from mileage.

Hypothesis: `estimatePrice(mileage) = θ0 + θ1 * mileage`

## Setup

Before running any scripts, set up the Python virtual environment:

```bash
./setup.sh
```

**What `setup.sh` does:**
- Creates a Python virtual environment (`.venv/`) isolated from your system Python
- Installs all required packages listed in `requirements.txt` (pandas, matplotlib)
- Requires Python 3.10 or newer

**Activate the virtual environment:**
```bash
source .venv/bin/activate
```

Once activated, your terminal prompt will show `(.venv)`. All subsequent `python` commands will use packages from `.venv/`.

**To deactivate later:**
```bash
deactivate
```

## Quick Start

### 1. Train the Model
```bash
python train.py datasets/data.csv
```
**Output:** `model.json`
- Contains: `theta0`, `theta1`, `mileage_mean`, `mileage_std`
- The model is trained on normalized mileage (z-score); the saved mean/std allow predictions to be normalized identically

### 2. Predict a Price
```bash
python predict.py model.json
```
**Output:** Interactive prompt
- Enter a mileage when asked
- Prints the estimated price based on the trained linear model

### 3. Explore the Data (Bonus)
```bash
python plot_data.py datasets/data.csv
```
**Output:** Matplotlib window with a scatter plot of mileage vs price

### 4. Plot the Regression Line (Bonus)
```bash
python plot_line.py datasets/data.csv model.json
```
**Output:** Matplotlib window with the original data points plus the fitted regression line

### 5. Evaluate the Model (Bonus)
```bash
python evaluate_model.py datasets/data.csv model.json
```
**Output:** Console metrics
- `MSE`: Mean Squared Error
- `RMSE`: typical prediction error in price units
- `R²`: how much variance the model explains (1.0 = perfect, 0.0 = no better than predicting the mean)

## Files

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `setup.py` | Shared constants (hyperparameters, dataset path) | — | — |
| `tools.py` | Shared helpers: stats, normalization, gradient descent, metrics | — | — |
| `train.py` | Train the linear model | `data.csv` | `model.json` |
| `predict.py` | Interactive price prediction | `model.json` | Console output |
| `plot_data.py` | Scatter plot of raw data | `data.csv` | Matplotlib window |
| `plot_line.py` | Data + regression line | `data.csv` + `model.json` | Matplotlib window |
| `evaluate_model.py` | Precision metrics | `data.csv` + `model.json` | Console output |

## Architecture

- **1 feature:** mileage (km) + 1 bias = 2 weights (θ0, θ1)
- **Normalization:** z-score on training mileage; same mean/std applied at prediction time
- **Optimization:** batch gradient descent, 10000 iterations, learning rate α=0.01
- **Loss:** Mean Squared Error, tracked per iteration during training

## Key Implementation Details

- All ML logic from scratch (no numpy/sklearn for core algorithms)
- Pandas used for CSV reading only
- Gradient descent updates θ0 and θ1 simultaneously each iteration
- `std == 0` edge case handled in normalization (returns 0)

---

**Requires:** Python 3.10+, pandas, matplotlib
