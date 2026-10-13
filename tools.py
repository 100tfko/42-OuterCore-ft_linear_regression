import pandas as pd
import math
from pathlib import Path
from typing import Optional
from setup import MAX_ITERATIONS, EPSILON, ALPHA


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


def mean(col: list[float]) -> float:
    """Return the arithmetic mean of a column."""
    return (sum(col) / len(col))


def std(col: list[float]) -> float:
    """Return the sample standard deviation (n-1)."""
    mean_value = mean(col)
    variance = sum([(x - mean_value) ** 2 for x in col]) / (len(col) - 1)
    return math.sqrt(variance)


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


def gradient_descent(
        mileages: list[float],
        prices: list[float],
        theta0: float,
        theta1: float
) -> tuple[float, float, list[float]]:
    """Gradient descent method to adjust thetas of the model."""
    loss_history = []
    previous_loss = float('inf')
    for iteration in range(MAX_ITERATIONS):
        tmp0 = 0
        for i in range(len(prices)):
            error = hypothesis(mileages[i], theta0, theta1) - prices[i]
            tmp0 += error
        tmp0 = tmp0 * ALPHA * 1/len(prices)

        tmp1 = 0
        for i in range(len(prices)):
            error = hypothesis(mileages[i], theta0, theta1) - prices[i]
            tmp1 += error * mileages[i]
        tmp1 = tmp1 * ALPHA * 1/len(prices)

        theta0 -= tmp0
        theta1 -= tmp1

        # Compute and track loss
        predictions = [hypothesis(m, theta0, theta1) for m in mileages]
        loss = mse_loss(prices, predictions)
        loss_history.append(loss)

        if previous_loss - loss < EPSILON:
            break          # improvement too small → converged
        previous_loss = loss
    return (theta0, theta1, loss_history)

    # Returns: (final_theta0, final_theta1, loss_history)
    # Simultaneously update θ0 and θ1 using the formulas from the spec


def correlation(x: list[float], y: list[float]) -> float:
    """Return the Pearson correlation coefficient r between x and y."""
    mx = mean(x)
    my = mean(y)
    numerator = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denominator = math.sqrt(
        sum((xi - mx) ** 2 for xi in x) *
        sum((yi - my) ** 2 for yi in y)
    )
    if denominator == 0:
        return 0
    return numerator / denominator


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
