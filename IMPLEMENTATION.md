# Linear Regression — Implementation Plan

## Overview

Build a simple linear regression model to predict car prices from mileage.

Hypothesis: `estimatePrice(mileage) = θ0 + θ1 * mileage`

---

## File Structure

```
project/
├── setup.py              # Shared constants: dataset path, hyperparameters
├── tools.py              # Shared helpers (load, stats, math, normalization, gradient descent)
├── train.py              # Train linear regression, save θ0 and θ1 [MANDATORY]
├── predict.py            # Load weights, predict price for given mileage [MANDATORY]
├── describe.py           # Statistical summary of mileage and price [BONUS]
├── scatter_plot.py       # Scatter plot of mileage vs price + regression line [BONUS]
├── histogram.py          # Price distribution histogram [BONUS]
├── datasets/
│   ├── data.csv          # Training dataset (mileage, price)
```

---

## Phase 1: `setup.py` — Shared Constants

### 1.1 Define constants

```python
DATASET_PATH = "datasets/data.csv"
MODEL_PATH = "model.json"
LEARNING_RATE = 0.01
ITERATIONS = 1000
```

---

## Phase 2: `tools.py` — Shared Utilities

### 2.1 Data Loading

```python
def load_csv(path: str) -> Optional[pd.DataFrame]
```

- Validate file exists, parse with `pd.read_csv`, check not empty
- Return `None` on failure

### 2.2 Core Statistics (from scratch)

```python
def count(col: list[float]) -> int
def mean(col: list[float]) -> float
def std(col: list[float]) -> float       # Sample std (divide by n-1)
def min(col: list[float]) -> float
def max(col: list[float]) -> float
```

### 2.3 Normalization

```python
def normalize(values: list[float], mean: Optional[float] = None, std: Optional[float] = None) -> tuple[list[float], float, float]
```

- If mean/std provided: apply them (prediction)
- If not: compute from values (training), return them alongside normalized values
- Handle std=0 edge case (constant column → return 0)

### 2.4 Linear Regression Building Blocks

```python
def hypothesis(mileage: float, theta0: float, theta1: float) -> float
def mse_loss(y_true: list[float], y_pred: list[float]) -> float
def gradient_descent(X: list[float], y: list[float], theta0: float, theta1: float, alpha: float, iterations: int) -> tuple[float, float, list[float]]
```

- `hypothesis`: θ0 + θ1 * mileage
- `mse_loss`: mean squared error = (1/m) * Σ(y_true[i] - y_pred[i])²
- `gradient_descent`: 
  - Loop iterations times
  - For each iteration:
    - Compute predictions for all samples
    - Compute gradients using the formulas:
      - tmpθ0 = learningRate * (1/m) * Σ(estimatePrice(mileage[i]) - price[i])
      - tmpθ1 = learningRate * (1/m) * Σ(estimatePrice(mileage[i]) - price[i]) * mileage[i]
    - Update θ0 and θ1 simultaneously
    - Track MSE loss
  - Return final θ0, θ1, and loss history

---

## Phase 3: `train.py` — Training [MANDATORY]

### 3.1 Pipeline

```
1. Load dataset
2. Extract mileage and price columns as lists[float]
3. Normalize mileage (z-score), save μ and σ
4. Initialize θ0 = 0, θ1 = 0
5. Run gradient descent (cross-entropy loss)
6. Save θ0, θ1, μ, σ to model.json
7. Print final loss and parameters
```

### 3.2 Model File Format (JSON)

```json
{
    "theta0": 0.5,
    "theta1": 0.15,
    "mileage_mean": 150000.5,
    "mileage_std": 50000.2
}
```

---

## Phase 4: `predict.py` — Prediction [MANDATORY]

### 4.1 Pipeline

```
1. Load model.json
2. Prompt user for mileage
3. Normalize mileage using saved μ and σ
4. Compute estimatePrice = θ0 + θ1 * (normalized_mileage)
5. Print estimated price
6. Optionally loop to predict multiple cars
```

### 4.2 Example Interaction

```
Enter a mileage: 120000
Estimated price: $25,500.00
```

---

## Hyperparameter Tuning

| Parameter | Starting value | Notes |
|---|---|---|
| Learning rate (α) | 0.01 | Adjust based on loss curve |
| Iterations | 1000 | Increase until convergence |
| Initial theta | [0, 0] | Both start at zero |

Loss (MSE) should decrease each iteration. If it oscillates or increases, reduce α. If it's still dropping at the last iteration, increase iterations.

---

## Phase 5: `describe.py` — Data Exploration [BONUS]

### 5.1 Pipeline

```
1. Load CSV
2. Extract mileage column (numeric)
3. Extract price column (numeric)
4. For each column: compute count, mean, std, min, max
5. Print formatted summary table
```

- Takes dataset path from `setup.py` constant
- No CLI args needed
- Format: clean table with statistics

---

## Phase 6: `scatter_plot.py` — Visualization [BONUS]

### 6.1 Pipeline

```
1. Load dataset
2. Extract mileage and price columns
3. Create scatter plot (mileage on x-axis, price on y-axis)
4. Optionally: plot regression line (if model.json exists)
```

- Visualize data distribution
- Show correlation between features
- Answer: is there a linear relationship?

---

## Phase 7: `histogram.py` — Price Distribution [BONUS]

### 7.1 Pipeline

```
1. Load dataset
2. Extract price column
3. Plot histogram of price distribution
4. Optionally: overlay with normal distribution
```

- Understand price range and variance
- Detect outliers
- Visualize data quality

---

## Bonus Features

1. **Plot regression line**: After training, re-plot scatter with fitted line y = θ0 + θ1*x
2. **Precision program**: Compute R² score or RMSE on test/validation set
3. **Loss visualization**: Plot MSE loss curve over iterations to verify convergence
