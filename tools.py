import pandas as pd
import math
from pathlib import Path
from typing import Optional
from setup import MAX_ITERATIONS


def load_csv(path: str) -> Optional[pd.DataFrame]:
    """Load a CSV file and return a DataFrame, or None on failure."""
    file = Path(path)
    if not file.exists():
        print(f"File '{path}' doesn't exist")
        return None

    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError,
            pd.errors.ParserError,
            ValueError,
            ) as e:
        print(f"Error reading '{path}': {e}")
        return

    if df.empty:
        print(f"File {path} is empty")
        return None

    return df


def fill_nan(col: list[Optional[float]]) -> list[float]:
    """Replace None entries with the column mean."""
    values = [x for x in col if x is not None]
    mean_val = sum(values) / len(values)
    return [v if v is not None else mean_val for v in col]


def count(col: list[float]) -> int:
    """Return the number of elements in a column."""
    return len(col)


def mean(col: list[float]) -> float:
    """Return the arithmetic mean of a column."""
    return (sum(col) / len(col))


def std(col: list[float]) -> float:
    """Return the sample standard deviation (n-1)."""
    mean_value = mean(col)
    variance = sum([(x - mean_value) ** 2 for x in col]) / (len(col) - 1)
    return math.sqrt(variance)


def percentile(col: list[float], p: float) -> float:
    """Return the p-th percentile with linear interpolation."""
    sorted_col = sorted(col)
    rank = p / 100 * (len(col) - 1)
    k = math.floor(rank)
    d = rank - k
    if d == 0:
        return sorted_col[k]
    result = sorted_col[k] + d * (sorted_col[k + 1] - sorted_col[k])
    return result


def min(col: list[float]) -> float:
    """Return the minimum value in a column."""
    min_value = float('inf')
    for value in col:
        if value < min_value:
            min_value = value
    return min_value


def max(col: list[float]) -> float:
    """Return the maximum value in a column."""
    max_value = -float('inf')
    for value in col:
        if value > max_value:
            max_value = value
    return max_value


def encode_best_hand(col: list[str]) -> list[int]:
    """Map 'Right' to 1 and 'Left' to 0."""
    return [1 if v == 'Right' else 0 for v in col]


def sigmoid(z: float) -> float:
    """Logistic sigmoid with overflow protection."""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        return math.exp(z) / (1 + math.exp(z))



def predict_proba(x: list[list[float]], thetas: list[float]) -> list[float]:
    """Compute sigmoid scores for all rows given thetas."""
    probs = []
    for row in x:
        scalar = 0
        for theta, feature in zip(thetas, row):
            scalar += theta * feature
        sig_result = sigmoid(scalar)
        probs.append(sig_result)
    return probs


def normalize(X: list[list[float]],
              means=None,
              stds=None) -> tuple[list[list[float]], list[float], list[float]]:
    """Z-score normalize X. Compute or apply given means/stds."""
    if means is None or stds is None:
        means_list = []
        std_list = []
        number_features = len(X[0])
        for j in range(number_features):
            col = []
            for i in range(len(X)):
                col.append(X[i][j])
            col_mean = mean(col)
            col_std = std(col)
            means_list.append(col_mean)
            std_list.append(col_std)
        means = means_list
        stds = std_list

    X_norm = [[0.0] * len(X[0]) for _ in range(len(X))]
    for j in range(len(X[0])):
        for i in range(len(X)):
            if stds[j] == 0:
                X_norm[i][j] = 0
            else:
                X_norm[i][j] = (X[i][j] - means[j]) / stds[j]
    return (X_norm, means, stds)

def argmax(lst: list[float]) -> int:
    """Return the index of the largest element."""
    max_idx = 0
    max_val = lst[0]
    for i in range(1, len(lst)):
        if lst[i] > max_val:
            max_val = lst[i]
            max_idx = i
    return max_idx

def correlation(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if a == a and b == b]

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    if (min(xs) == max(xs) or min(ys) == max(ys) or len(pairs) < 2):
        return 0

    mx = mean(xs)
    my = mean(ys)

    cov = sum((xs[i] - mx) * (ys[i] - my)
              for i in range(len(xs))) / (len(xs) - 1)

    return cov / (std(xs) * std(ys))

def hypothesis(mileage: float, theta0: float, theta1: float) -> float:
    """Return the estimated price for a determined mileage."""
    estimate = theta0 + (theta1 * mileage)
    return estimate
    # h(x) = θ0 + θ1 * x

def mse_loss(y_true: list[float], y_pred: list[float]) -> float:
    """Returns the Mean Squared Error for the prediction model."""
    squared_errors = 0
    for i in range(len(y_true)):
        error = y_true[i] - y_pred[i]
        squared_errors += error ** 2
    mse = squared_errors / len(y_true)
    return mse

# (1/m) * Σ(y_true[i] - y_pred[i])²

#   could do a list comprehension like this:
#   squared_errors = [(y_true[i] - y_pred[i])**2 for i in range(len(y_true))]
#   mse = sum(squared_errors) / len(y_true)


def gradient_descent(mileages: list[float], prices: list[float], theta0: float, theta1: float, alpha: float, iterations: int) -> tuple[float, float, list[float]]:
    """Gradient descent method to adjust thetas of the model."""
    loss_history = []
    for iteration in range(iterations):
        tmp0 = 0
        for i in range(len(prices)):
            tmp0 += hypothesis(mileages[i], theta0, theta1) - prices[i]
        tmp0 = tmp0 * alpha * 1/len(prices)

        tmp1 = 0
        for i in range(len(prices)):
            tmp1 += (hypothesis(mileages[i], theta0, theta1) - prices[i]) * mileages[i]
        tmp1 = tmp1 * alpha * 1/len(prices)

        theta0 -= tmp0
        theta1 -= tmp1

            # Compute and track loss
        predictions = [hypothesis(m, theta0, theta1) for m in mileages]
        loss = mse_loss(prices, predictions)
        loss_history.append(loss)
    return (theta0, theta1, loss_history)
    

    # Returns: (final_theta0, final_theta1, loss_history)
    # Simultaneously update θ0 and θ1 using the formulas from the spec

def r_squared(y_true: list[float], y_pred: list[float]) -> float:
    """Return the R² score: 1 - SS_res / SS_tot."""
    y_mean = mean(y_true)
    ss_res = 0
    ss_tot = 0
    for i in range(len(y_true)):
        ss_res += (y_true[i] - y_pred[i]) ** 2
        ss_tot += (y_true[i] - y_mean) ** 2
    if ss_tot == 0:
        return 0
    return 1 - ss_res / ss_tot

def rmse(y_true: list[float], y_pred: list[float]) -> float:
    """Return the Root Mean Squared Error."""
    return math.sqrt(mse_loss(y_true, y_pred))

def normalize_single_value(value: float, mean: float, std: float) -> float:
    """Normalize a single value."""
    if std == 0:
        return 0
    return (value - mean) / std