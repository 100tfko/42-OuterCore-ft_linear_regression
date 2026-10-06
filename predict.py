import json
import sys
from tools import (
    hypothesis,
    normalize_single_value
)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python predict.py <model.json>", file=sys.stderr)
        sys.exit(1)

    # Load model from argument
    try:
        with open(sys.argv[1], 'r') as weights_file:
            weights_data = json.load(weights_file)
    except FileNotFoundError:
        print("Error: weights file not found.", file=sys.stderr)
        sys.exit(1)

    # Extract theta0, theta1, mean, std from weights_data
    theta0 = weights_data['theta0']
    theta1 = weights_data['theta1']
    mean_mileage = weights_data['mileage_mean']
    std_mileage = weights_data['mileage_std']

    # Prompt user for mileage
    mileage = float(input("Enter a mileage: "))

    # Normalize
    normalized_mileage = normalize_single_value(mileage, mean_mileage, std_mileage)

    # Predict
    price = hypothesis(normalized_mileage, theta0, theta1)

    # Print
    print(f"Estimated price: ${price:,.2f}")