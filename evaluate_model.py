import sys
import json
from tools import (
    load_csv,
    hypothesis,
    normalize_single_value,
    mse_loss,
    rmse,
    r_squared
)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python evaluate_model.py <dataset_file_name.csv> <model.json>",
              file=sys.stderr)
        sys.exit(1)
    data = load_csv(sys.argv[1])
    if data is None:
        print("Error: Failed to load dataset.", file=sys.stderr)
        sys.exit(1)
    try:
        with open(sys.argv[2], 'r') as weights_file:
            weights_data = json.load(weights_file)
    except FileNotFoundError:
        print("Error: weights file not found.", file=sys.stderr)
        sys.exit(1)

    # Extract theta0, theta1, mean, std from weights_data
    theta0 = weights_data['theta0']
    theta1 = weights_data['theta1']
    mean_mileage = weights_data['mileage_mean']
    std_mileage = weights_data['mileage_std']

    mileages = [float(value) for value in data["km"].tolist()]
    prices = [float(value) for value in data["price"].tolist()]

    predicted_prices = []
    for mileage in mileages:
        normalized_mileage = normalize_single_value(mileage, mean_mileage, std_mileage)
        predicted_prices.append(hypothesis(normalized_mileage, theta0, theta1))

    mse = mse_loss(prices, predicted_prices)
    print(f"MSE:  {mse:,.2f}")
    print(f"RMSE: {rmse(prices, predicted_prices):,.2f}")
    print(f"R²:   {r_squared(prices, predicted_prices):.4f}")