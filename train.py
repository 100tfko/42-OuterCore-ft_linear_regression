import json
import sys
from setup import MAX_ITERATIONS, ALPHA
from tools import (
    load_csv,
    gradient_descent,
    mean,
    std,
    normalize_single_value
)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train.py <dataset_file_name.csv>",
              file=sys.stderr)
        sys.exit(1)
    data = load_csv(sys.argv[1])
    if data is None:
        print("Error: Failed to load dataset.", file=sys.stderr)
        sys.exit(1)

    mileages = [float(value) for value in data["km"].tolist()]
    prices = [float(value) for value in data["price"].tolist()]

    mileage_mean = mean(mileages)
    mileage_std = std(mileages)
    normalized_mileages = [
        normalize_single_value(mileage, mileage_mean, mileage_std)
        for mileage in mileages
    ]
    theta0 = 0.0
    theta1 = 0.0

    theta0, theta1, loss_history = gradient_descent(normalized_mileages, prices, theta0, theta1, ALPHA, MAX_ITERATIONS)

    model_data = {
    "theta0": theta0,
    "theta1": theta1,
    "mileage_mean": mileage_mean,
    "mileage_std": mileage_std
    }

    with open("model.json", "w") as file:
        json.dump(model_data, file, indent=4)